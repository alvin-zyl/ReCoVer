## Muon code from Moonlight
## https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
import torch
from functools import partial
import math
import warnings
from nanotron.optim.polar_express import PolarExpress, FastApplyPolarExpress

# ---------------------------------------------------------------------------
# Remez-optimized NS-3 coefficients for MuDam (adam_preconditioned).
#
# Each entry is a list of 3 (a,b,c) tuples — one per NS step.
# The composed polynomial phi3(phi2(phi1(sigma))) is minimaxed over the
# empirical mudam SV interval [sigma_lo, sigma_hi] measured from training logs.
#
# Methodology: scipy Nelder-Mead minimax on linspace(sigma_lo, sigma_hi, 800),
# warm-started from multiple random seeds. max_err is the final Chebyshev error
# on 2000-point verification grid; compare against std NS5 max_err = 0.318.
#
# Note: step-1 polynomial maps sigma -> negative values (a < 0) before step-2
# corrects back. This is mathematically valid: A_{k+1} = X_{k+1} X_{k+1}^T
# depends on phi(sigma)^2, not |phi(sigma)|, so the composition is exact.
# Intermediate magnitudes stay within [-1.1, 1.2], well within bfloat16 range.
# ---------------------------------------------------------------------------

# Early training — adam_preconditioned (steps ~0-25%, sigma_hi~0.88), max_err=0.0604
_MUDAM_NS3_EARLY = [
    (-3.6696, 10.0034, -7.0261),  # step 1
    (-3.2889,  9.9988, -8.6186),  # step 2
    ( 3.9926,-10.0015,  9.4264),  # step 3
]

def _apply_ns3_abc(G, abc_list):
    """Apply a sequence of asymmetric NS steps with the given (a,b,c) per step."""
    assert len(G.shape) >= 2
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in abc_list:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.mT
    return X

def mudam_ns3_early(G, steps=3):
    """
    3-step NS for MuDam with Remez-optimized coefficients for early training.

    Coefficients minimaxed over sigma in [0.025, 0.88] — the empirical 5th/95th
    percentile energy range of the Adam-preconditioned gradient SV distribution
    during the first ~25% of training (from moment-log analysis).

    max_err (Chebyshev) = 0.060 vs 0.318 for standard NS5 — ~5x improvement.
    `steps` is accepted for API compatibility but always runs exactly 3 steps.
    """
    return _apply_ns3_abc(G, _MUDAM_NS3_EARLY)


def jiacheng(G, steps):
    """
    Jiacheng optimized polynomials
    """
    assert len(G.shape) >= 2
    abc_list = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    if steps > len(abc_list):
        steps = len(abc_list)
    for a, b, c in abc_list[:steps]:
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X


def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X


def svd_exact_polar(G, _, cutoff=None, reverse=False):
    """
    Exact polar factorization via SVD
    """
    assert len(G.shape) >= 2
    U, Sigma, Vh = torch.linalg.svd(G.to(torch.float32), full_matrices=False)
    if cutoff is None:
        return (U @ Vh).to(G.dtype)
    else:
        Sigma = ((Sigma / Sigma.max()) >= cutoff).to(
            G.dtype
        )  # zero out small singular values
        if reverse:
            Sigma = 2 * Sigma - 1
        return (U @ torch.diag(Sigma) @ Vh).to(G.dtype)


def apply_post_polar_norm(
    u: torch.Tensor, norm_mode: str, eps: float = 1e-8
) -> torch.Tensor:
    """Apply row/col L2 normalization to the polar-factorized update u.

    Matches the soft_polar_ns convention:
      - eps is placed *inside* the sqrt: sqrt(sum_sq + 1e-7)
      - target_norm = min(m, n) ** 0.5  (Frobenius norm of a true polar factor)
      - Frobenius rescaling is applied once at the end

    Modes
    -----
    "col"     : col-wise norm, then Frobenius rescale
    "row"     : row-wise norm, then Frobenius rescale
    "col_row" : col-wise then row-wise, then Frobenius rescale
    "row_col" : row-wise then col-wise, then Frobenius rescale
    "none"    : no-op
    """
    if norm_mode in ("none", None, ""):
        return u

    # min(m, n) ** 0.5 — Frobenius norm of a true UV^T polar factor
    target_norm = min(u.size(-2), u.size(-1)) ** 0.5

    # First normalization
    if norm_mode in ("col", "col_row"):
        u = u / torch.sqrt((u * u).sum(dim=-2, keepdim=True) + 1e-7)
    elif norm_mode in ("row", "row_col"):
        u = u / torch.sqrt((u * u).sum(dim=-1, keepdim=True) + 1e-7)
    else:
        raise ValueError(
            f"Unknown norm_mode '{norm_mode}'. "
            "Choose from: 'none', 'col', 'row', 'col_row', 'row_col'."
        )

    # Second normalization (combined modes only)
    if norm_mode == "col_row":
        u = u / torch.sqrt((u * u).sum(dim=-1, keepdim=True) + 1e-7)
    elif norm_mode == "row_col":
        u = u / torch.sqrt((u * u).sum(dim=-2, keepdim=True) + 1e-7)

    # Frobenius rescaling — one single rescale at the end
    # u = u * (target_norm / (u.norm(dim=(-2, -1), keepdim=True) + 1e-7))

    return u


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        param_groups,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        rms_scaling=True,
        nuclear_scaling=False,
        polar_method="Keller",
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        polar_args={},
        muon_mode="sgd",
        norm_mode="none",
    ):
        """
        Accepts standard PyTorch param groups. Each group should have a "use_muon" bool
        flag (set upstream in helpers.py) indicating whether to apply Muon or AdamW.
        Per-group lr and weight_decay are respected from the param groups.

        Arguments:
            polar_method: The name of the polar factorization method to use (e.g., "NewtonSchultz", "Keller", "Pole") where PolE = PolarExpress
        """
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            rms_scaling=rms_scaling,
            nuclear_scaling=nuclear_scaling,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            muon_mode=muon_mode,
            use_muon=False,
            norm_mode=norm_mode,
        )

        super().__init__(param_groups, defaults)

        # Instantiate the polar factorization method
        self.polar_factorizer = self._initialize_polar_factorizer(
            polar_method, polar_args
        )

    def _initialize_polar_factorizer(self, polar_method, polar_args):
        """Initialize the polar factorization method based on the provided name and parameters."""
        if polar_method == "Keller":
            return zeropower_via_newtonschulz5  # Use the method directly
        elif polar_method == "Jiacheng":
            return jiacheng
        elif polar_method == "polarexpress":
            return PolarExpress
        elif polar_method == "fast_polarexpress":
            return partial(FastApplyPolarExpress, restart_interval=3, shift_eps=1e-3)
        elif polar_method == "mudam-ns3-early":
            return mudam_ns3_early
        elif polar_method == "svd-exact":
            return partial(
                svd_exact_polar,
                cutoff=polar_args.get("svd_cutoff", None),
                reverse=polar_args.get("svd_reverse", False),
            )
        else:
            raise ValueError(f"Unknown polar method: {polar_method}")

    def adjust_lr_for_muon(
        self,
        lr,
        rms_scaling,
        nuclear_scaling,
        param_shape,
        grad,
        grad_sign,
    ):
        scale = 1.0
        if rms_scaling:
            fan_out, fan_in = param_shape[:2]
            scale *= math.sqrt(fan_out / fan_in)
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        return lr * scale

    def step(self, closure=None):
        """Perform a single optimization step.
        Args:
        closure (Callable, optional): A closure that reevaluates the model
            and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group.get("weight_decay", 0.0)

            if group["use_muon"]:
                ############################
                #           Muon           #
                ############################
                momentum = group["momentum"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    muon_mode = group.get("muon_mode", "sgd")

                    if muon_mode == "sgd":
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(g)
                        if group["nesterov"]:
                            g = g.add(buf, alpha=momentum)
                        else:
                            g = buf

                    elif muon_mode == "adam_preconditioned":
                        beta1, beta2 = group["adamw_betas"]
                        eps = group["adamw_eps"]
                        if "moment1" not in state:
                            state["moment1"] = torch.zeros_like(g)
                            state["moment2"] = torch.zeros_like(g)
                        state["moment1"].lerp_(g, 1 - beta1)
                        state["moment2"].lerp_(g.square(), 1 - beta2)
                        g = state["moment1"] / (state["moment2"].sqrt() + eps)
                    
                    elif muon_mode == "adafactor_preconditioned":
                        # Adafactor-preconditioned MuDam: replaces full O(m*n) second moment with
                        # factored row/col vectors v_r [m,1] and v_c [1,n], approximating
                        # V_{ij} ≈ v_r[i] * v_c[j] / sum(v_r).
                        #
                        # Memory vs adam_preconditioned:
                        #   adam:      moment1 [m,n] + moment2 [m,n]  → 2*m*n
                        #   adafactor: moment1 [m,n] + v_r [m] + v_c [n] → m*n + m + n  (beta1 > 0)
                        #              v_r [m] + v_c [n]                  → m + n        (beta1 == 0)
                        beta1, beta2 = group["adamw_betas"]
                        eps = group["adamw_eps"]
                        if "v_r" not in state:
                            state["v_r"] = torch.zeros(g.size(0), 1, dtype=torch.float32, device=g.device)
                            state["v_c"] = torch.zeros(1, g.size(1), dtype=torch.float32, device=g.device)
                            if beta1 > 0:
                                state["moment1"] = torch.zeros_like(g)
                        # Update factored second moment (float32 for numerical stability)
                        g_sq = g.float().square()
                        state["v_r"].mul_(beta2).add_(g_sq.sum(dim=-1, keepdim=True), alpha=1 - beta2)
                        state["v_c"].mul_(beta2).add_(g_sq.sum(dim=-2, keepdim=True), alpha=1 - beta2)
                        # Reconstruct approximate second moment via outer product, Adafactor normalisation
                        v_r = state["v_r"]  # [m, 1], float32
                        v_c = state["v_c"]  # [1, n], float32
                        v_hat = v_r * v_c / (v_r.sum() + eps)  # [m, n]
                        # First moment (optional; skip when beta1 == 0 to save another m*n buffer)
                        if beta1 > 0:
                            state["moment1"].lerp_(g, 1 - beta1)
                            numerator = state["moment1"].float()
                        else:
                            numerator = g.float()
                        g = (numerator / (v_hat.sqrt() + eps)).to(g.dtype)

                    elif muon_mode == "muon_adam_var":
                        beta2 = group["adamw_betas"][1]
                        eps = group["adamw_eps"]
                        if "moment1" not in state:
                            state["moment1"] = torch.zeros_like(g)
                            state["moment2"] = torch.zeros_like(g)
                        state["moment1"].lerp_(g, 1 - momentum)
                        state["moment2"].lerp_(g.square(), 1 - beta2)
                        g = state["moment1"] / (state["moment2"].sqrt() + eps)

                    elif muon_mode == "two_moment":
                        beta1, beta2 = group["adamw_betas"]
                        eps = group["adamw_eps"]
                        if "muon_step" not in state:
                            state["muon_step"] = 0
                            state["moment1"] = torch.zeros_like(g)
                            state["moment2"] = torch.zeros_like(g)
                        state["muon_step"] += 1
                        t = state["muon_step"]
                        state["moment1"].lerp_(g, 1 - beta1)
                        state["moment2"].lerp_(g.square(), 1 - beta2)
                        m_hat = state["moment1"] / (1 - beta1**t)
                        v_hat = state["moment2"] / (1 - beta2**t)
                        g = m_hat / (v_hat.sqrt() + eps)

                    else:
                        raise ValueError(f"Unknown muon_mode: {muon_mode}")

                    u = self.polar_factorizer(g, group["ns_steps"])

                    # Apply post-polar normalization if enabled
                    norm_mode = group.get("norm_mode", "none")
                    if norm_mode != "none":
                        u = apply_post_polar_norm(u, norm_mode=norm_mode)

                    adjusted_lr = self.adjust_lr_for_muon(
                        lr,
                        group["rms_scaling"],
                        group["nuclear_scaling"],
                        p.shape,
                        g.bfloat16(),
                        u,
                    )

                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(u, alpha=-adjusted_lr)

            else:
                ############################
                #       AdamW backup       #
                ############################
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    state["step"] += 1
                    step = state["step"]
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(g, alpha=-lr / scale)

        return loss
