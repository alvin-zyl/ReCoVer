import os
import torch
from setuptools import setup, find_packages
from torch.utils import cpp_extension

sources = ["csrc/ProcessGroupULFM.cpp", "csrc/ULFMLogging.cpp", "csrc/bindings.cpp"]
mpi_home = os.environ.get("MPI_HOME")
if not mpi_home:
    raise RuntimeError("MPI_HOME environment variable is not set. Please set it to your OpenMPI (with ULFM support) install prefix.")
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/csrc/", f"{mpi_home}/include"]
library_dirs = [f"{mpi_home}/lib", torch_lib_dir]
# Embed runtime paths so libc10.so / libmpi.so are found without LD_LIBRARY_PATH
extra_link_args = [
    f"-Wl,-rpath,{torch_lib_dir}",
    f"-Wl,-rpath,{mpi_home}/lib",
]

# FORCE_CUDA=1 builds the CUDA extension on a machine without a visible GPU
# (e.g. inside `docker build`).
if torch.cuda.is_available() or os.environ.get("FORCE_CUDA") == "1":
    module = cpp_extension.CUDAExtension(
        name="ulfm_collectives._C",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_link_args=extra_link_args,
    )
else:
    module = cpp_extension.CppExtension(
        name="ulfm_collectives._C",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_link_args=extra_link_args,
    )

setup(
    name="recover",
    version="1.0.0",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=find_packages(exclude=["dist", "build", "*.egg-info"]),
    entry_points={"console_scripts": ["recover = recover.cli:main"]},
)
