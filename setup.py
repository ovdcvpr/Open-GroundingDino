import glob
import os
import subprocess
import sys
import ensurepip

from pkg_resources import get_distribution, DistributionNotFound
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext

def ensure_torch_installed():
    try:
        get_distribution('torch')
    except DistributionNotFound:
        print("Torch not found. Installing Torch...")
        ensurepip.bootstrap()  # Ensures pip is available in the environment
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch'])


def check_cuda_availability():
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that CUDA is installed and properly configured.")
    except ImportError:
        raise RuntimeError("Torch is not installed. Cannot check for CUDA availability.")


class CustomInstallCommand(install):
    def run(self):
        ensure_torch_installed()
        check_cuda_availability()
        install.run(self)


class CustomBuildExtCommand(build_ext):
    def run(self):
        # raise Exception
        ensure_torch_installed()
        check_cuda_availability()
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

        def get_extensions():
            this_dir = os.path.dirname(os.path.abspath(__file__))
            extensions_dir = os.path.join(this_dir, "open_groundingdino/models/GroundingDINO/ops/src")

            main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
            source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
            source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

            sources = main_file + source_cpu
            extension = CppExtension
            extra_compile_args = {"cxx": []}
            define_macros = []

            if torch.cuda.is_available() and CUDA_HOME is not None:
                extension = CUDAExtension
                sources += source_cuda
                define_macros += [("WITH_CUDA", None)]
                extra_compile_args["nvcc"] = [
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ]
            else:
                raise RuntimeError('CUDA is not available or CUDA_HOME is not set.')

            sources = [os.path.join(extensions_dir, s) for s in sources]
            include_dirs = [extensions_dir]
            ext_modules = [
                extension(
                    "MultiScaleDeformableAttention",
                    sources,
                    include_dirs=include_dirs,
                    define_macros=define_macros,
                    extra_compile_args=extra_compile_args,
                )
            ]
            return ext_modules

        self.distribution.ext_modules = get_extensions()
        build_ext.run(self)


with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name="open_groundingdino",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/ovdcvpr/Open-GroundingDino",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention integrated into Open GroundingDINO",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.11',
    include_package_data=True,
    install_requires=install_requires,
    packages=find_packages(
        include=[
            'open_groundingdino',
            'open_groundingdino.*',
            'open_groundingdino.tools.*',
            'open_groundingdino.groundingdino.util.*',
            'open_groundingdino.groundingdino.*',
            'open_groundingdino.groundingdino',
        ],
        exclude=("configs", "tests",)
    ),
    package_dir={
        'open_groundingdino.groundingdino.util': 'open_groundingdino/groundingdino/util'
    },
    package_data={
        '': ['models/GroundingDINO/ops/src/cpu/**/*'],  # Adjust if needed
    },
    cmdclass={
        'install': CustomInstallCommand,
        'build_ext': CustomBuildExtCommand,
    },
    entry_points={},
)
