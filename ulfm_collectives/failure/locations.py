"""Failure-injection locations.

Each location names the phase of the training step in which the targeted
rank is killed. What matters for recovery is how far the step's
cross-replica gradient reduction has got:

  location         the rank dies                             when the survivors notice
  forward          before a microbatch's forward pass        no gradient of this step has been
                                                             reduced across replicas yet
  backward         in backward, right after a gradient       the same, except with pipeline
                   bucket was snapshotted and queued         parallelism: a rank finishes its
                   (gradient hooks are deferred: a rank      own backward before earlier stages
                   reduces only after its own backward)      do, so later stages of the failed
                                                             replica may have reduced already;
                                                             those buckets are rolled back
  grad-sync:first  in the cross-replica gradient             the buckets reduced so far contain
  grad-sync:mid    all-reduce, right after the first /       the failed replica's gradients and
  grad-sync:last   middle / last bucket completed            are rolled back to their snapshots
  optimizer        after the step's last cross-replica       the step is complete; the failure is
                   collective (loss logging), before the     found at the next step's consensus
                   local optimizer.step()                    and gradient sync

The names used in the paper's runs are accepted as aliases.
"""

GRAD_SYNC_POSITIONS = ("first", "mid", "last")

LOCATIONS = (
    "forward",
    "backward",
    *(f"grad-sync:{pos}" for pos in GRAD_SYNC_POSITIONS),
    "optimizer",
)

LOCATION_ALIASES = {
    "pre-forward": "forward",
    "post-deferred-hook-firing": "backward",
    "post-allreduce": "grad-sync:first",
}


def canonical_location(location: str) -> str:
    """Map an alias (e.g. the paper's ``post-allreduce``) to its canonical name."""
    return LOCATION_ALIASES.get(location, location)


def validate_location(location: str) -> None:
    if canonical_location(location) not in LOCATIONS:
        raise ValueError(
            f"unknown failure location {location!r}; expected one of "
            f"{', '.join(LOCATIONS)} (aliases: {', '.join(LOCATION_ALIASES)})"
        )
