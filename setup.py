import os
import sys
import platform
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
    from pybind11.setup_helpers import Pybind11Extension


class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            for ext in self.extensions:
                ext.extra_compile_args = [
                    "/O2", "/GL", "/arch:AVX2", "/fp:fast", "/EHsc",
                    "/D_USE_MATH_DEFINES", "/DNDEBUG", "/std:c++17"
                ]
                ext.extra_link_args = ["/LTCG"]
        else:
            for ext in self.extensions:
                ext.extra_compile_args = [
                    "-O3", "-march=native", "-ffast-math", "-funroll-loops",
                    "-flto", "-fPIC", "-DNDEBUG", "-std=c++17"
                ]
                ext.extra_link_args = ["-flto"]
        build_ext.build_extensions(self)


def get_cpp_sources():
    src_dir = Path("cpp_core/src")
    return [str(p) for p in src_dir.glob("*.cpp")]


ext_modules = [
    Pybind11Extension(
        "drone_core",
        sorted(get_cpp_sources()),
        include_dirs=["cpp_core/include"],
        language="c++",
    ),
]

setup(
    name="helix-drone",
    version="1.0.0",
    author="HelixDrone Team",
    description="High-Performance Quadrotor Physics Engine with LSTM-TD3 Agent",
    long_description=Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/chele-s/HelixDrone-HybridCore",
    packages=find_packages(where="python_src"),
    package_dir={"": "python_src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "torch>=2.0",
        "pybind11>=2.10",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "mypy"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="quadrotor drone reinforcement-learning lstm td3 physics-simulation",
)
