from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

# Definir los archivos fuente
cpp_sources = [
    "cpp_core/src/bindings.cpp",
    "cpp_core/src/Quadrotor.cpp",
    "cpp_core/src/PhysicsEngine.cpp",
]

# Definir la extensión
ext_modules = [
    Pybind11Extension(
        "drone_core",
        sorted(cpp_sources),
        include_dirs=["cpp_core/include"],
        define_macros=[("_USE_MATH_DEFINES", None)],
        extra_compile_args=["/O2"] if os.name == "nt" else ["-O3"],
    ),
]

setup(
    name="drone_core",
    version="0.1.0",
    description="High-performance Quadrotor Physics Engine in C++",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
