from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    IS_HIP = bool(IS_HIP_EXTENSION)
else:
    IS_HIP = BUILD_TARGET == "rocm"

if not IS_HIP:
    cc_flag = []
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

if platform.system() == "Windows":
    extra_compile_args = {
        "cxx": ["/O2", "/std:c++17", "/EHsc", "/permissive-", "/Zc:__cplusplus", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"],
        "nvcc": ["-O3", "-std=c++17", "-allow-unsupported-compiler", "-Xcompiler=/std:c++17", "-Xcompiler=/EHsc", "-Xcompiler=/permissive-", "-Xcompiler=/Zc:__cplusplus", "-Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"] + cc_flag,
    }
else:
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17"] + cc_flag,
    }

setup(
    name="o_voxel",
    packages=["o_voxel", "o_voxel.convert", "o_voxel.io"],
    ext_modules=[
        CUDAExtension(
            name="o_voxel._C",
            sources=[
                "src/hash/hash.cu",
                "src/convert/flexible_dual_grid.cpp",
                "src/convert/volumetic_attr.cpp",
                "src/serialize/api.cu",
                "src/serialize/hilbert.cu",
                "src/serialize/z_order.cu",
                "src/io/svo.cpp",
                "src/io/filter_parent.cpp",
                "src/io/filter_neighbor.cpp",
                "src/rasterize/rasterize.cu",
                "src/ext.cpp",
            ],
            include_dirs=[os.path.join(ROOT, "third_party/eigen")],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
