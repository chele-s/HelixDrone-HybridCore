import os
import sys
import struct
import subprocess
import sysconfig
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11>=2.10"])
    from pybind11.setup_helpers import Pybind11Extension


def detect_simd_capability():
    is_64bit = struct.calcsize("P") * 8 == 64
    arch = os.environ.get("HELIX_SIMD", "auto").lower()
    if arch != "auto":
        return arch
    if sys.platform == "linux" and is_64bit:
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            if "avx2" in cpuinfo:
                return "avx2"
            if "avx" in cpuinfo:
                return "avx"
            if "sse4_1" in cpuinfo:
                return "sse4"
        except Exception:
            pass
        return "avx"
    if sys.platform == "win32" and is_64bit:
        return "avx2"
    return "sse2"


class HelixBuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type

        simd = detect_simd_capability()

        for ext in self.extensions:
            if compiler_type == "msvc":
                ext.extra_compile_args = self._msvc_flags(simd)
                ext.extra_link_args = ["/LTCG", "/OPT:REF", "/OPT:ICF"]
            elif compiler_type in ("unix", "mingw32"):
                ext.extra_compile_args = self._gcc_clang_flags(simd)
                ext.extra_link_args = ["-flto", "-Wl,-O1"]
            else:
                ext.extra_compile_args = ["-O2", "-std=c++17"]

        build_ext.build_extensions(self)

    @staticmethod
    def _msvc_flags(simd):
        flags = [
            "/O2", "/Ob3", "/GL", "/fp:fast", "/EHsc",
            "/D_USE_MATH_DEFINES", "/DNDEBUG",
            "/std:c++17", "/permissive-",
            "/Gy", "/Gw",
        ]
        simd_map = {"avx2": "/arch:AVX2", "avx": "/arch:AVX", "sse4": "/arch:SSE2", "sse2": "/arch:SSE2"}
        flags.append(simd_map.get(simd, "/arch:AVX2"))
        return flags

    @staticmethod
    def _gcc_clang_flags(simd):
        flags = [
            "-O3", "-DNDEBUG", "-std=c++17",
            "-ffast-math", "-funroll-loops", "-flto",
            "-fPIC", "-fvisibility=hidden",
            "-fno-semantic-interposition",
            "-fomit-frame-pointer",
            "-D_USE_MATH_DEFINES",
        ]
        simd_map = {
            "avx2": ["-mavx2", "-mfma", "-mbmi2"],
            "avx": ["-mavx"],
            "sse4": ["-msse4.1"],
            "sse2": ["-msse2"],
        }
        flags.extend(simd_map.get(simd, ["-mavx"]))
        try:
            result = subprocess.run(
                [sysconfig.get_config_var("CC") or "gcc", "-march=native", "-E", "-x", "c", "/dev/null"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                flags = [f for f in flags if not f.startswith(("-mavx", "-msse", "-mfma", "-mbmi"))]
                flags.append("-march=native")
        except Exception:
            pass
        return flags


def collect_sources():
    src_dir = Path("cpp_core/src")
    return sorted(str(p) for p in src_dir.glob("*.cpp"))


ext_modules = [
    Pybind11Extension(
        "drone_core",
        collect_sources(),
        include_dirs=["cpp_core/include"],
        language="c++",
        define_macros=[("_USE_MATH_DEFINES", "1")],
        cxx_std=17,
    ),
]

readme = ""
if Path("README.md").exists():
    try:
        readme = Path("README.md").read_text(encoding="utf-8")
    except Exception:
        pass

setup(
    name="helix-drone",
    version="2.0.0",
    author="HelixDrone Team",
    description="High-Performance Quadrotor Physics Engine with LSTM-TD3 Agent",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/chele-s/HelixDrone-HybridCore",
    packages=find_packages(where="python_src"),
    package_dir={"": "python_src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": HelixBuildExt},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "torch>=2.0",
        "pybind11>=2.10",
        "PyYAML>=6.0",
        "gymnasium>=0.26",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "mypy"],
        "colab": ["matplotlib", "tensorboard", "tqdm"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="quadrotor drone reinforcement-learning lstm td3 physics-simulation",
)
