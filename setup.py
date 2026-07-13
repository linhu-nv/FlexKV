import json
import os
import shutil
import subprocess
import sys


from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension


def detect_cuda_arch():
    """Auto-detect GPU compute capability. Returns a semicolon-separated arch list.
    Falls back to a safe default when no GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            archs = set()
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                archs.add(f"{major}.{minor}")
            if archs:
                arch_list = ";".join(sorted(archs))
                print(f"Auto-detected GPU architectures: {arch_list}")
                return arch_list
    except Exception as e:
        print(f"GPU architecture auto-detection failed: {e}")
    # Fallback: common architectures (Ampere + Hopper)
    fallback = "8.0;8.6;9.0"
    print(f"No GPU detected, using fallback architectures: {fallback}")
    return fallback

def get_version():
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        return f.read().strip()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
build_dir = "build"
os.makedirs(build_dir, exist_ok=True)

# Check if we're in debug mode using environment variable
debug = os.environ.get("FLEXKV_DEBUG") == "1"
if debug:
    print("Running in debug mode - Cython compilation disabled")

enable_cfs = os.environ.get("FLEXKV_ENABLE_CFS", "0") == "1"
enable_gds = os.environ.get("FLEXKV_ENABLE_GDS", "0") == "1"
enable_p2p = os.environ.get("FLEXKV_ENABLE_P2P", "0") == "1"
enable_cputest = os.environ.get("FLEXKV_ENABLE_CPUTEST", "0") == "1"
# FLEXKV_ENABLE_METRICS=0: build without Prometheus (no prometheus-cpp dependency)
enable_metrics = os.environ.get("FLEXKV_ENABLE_METRICS", "0") == "1"

# Define C++ extensions (base: no dist/Redis)
cpp_sources = [
    "csrc/bindings.cpp",
    "csrc/transfer.cu",  # Skip CUDA file for now
    "csrc/hash.cpp",
    "csrc/tp_transfer_thread_group.cpp",
    "csrc/transfer_ssd.cpp",
    "csrc/radix_tree.cpp",
    "csrc/eviction_strategy.cpp",
    "csrc/monitoring/metrics_manager.cpp",  # Monitoring support
]

hpp_sources = [
    "csrc/cache_utils.h",
    "csrc/tp_transfer_thread_group.h",
    "csrc/transfer_ssd.h",
    "csrc/radix_tree.h",
    "csrc/eviction_strategy.h",
    "csrc/monitoring/metrics_manager.h",  # Monitoring support
]

# extra_link_args carries ONLY the libraries CMake does not manage: the CUDA
# driver, libc-provided pthread/rt, and the proprietary cuFile (GDS) / hifs
# (CFS) SDKs -- none of which have fetchable source. The managed native deps
# (xxHash, liburing, hiredis, prometheus-cpp) are resolved by CMakeLists.txt and
# injected below from the build/flexkv_pyext.json manifest, so they are NOT
# hard-coded here.
extra_link_args = ["-lcuda", "-lpthread", "-lrt"]

if enable_cputest:
    extra_link_args.remove("-lcuda")
    # Set TORCH_CUDA_ARCH_LIST to avoid IndexError when no GPU is available
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;9.0"

if not enable_metrics:
    print("FLEXKV_ENABLE_METRICS=0: building without Prometheus monitoring")
# Auto-detect GPU architecture if TORCH_CUDA_ARCH_LIST is not explicitly set
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = detect_cuda_arch()
print(f"TORCH_CUDA_ARCH_LIST = {os.environ['TORCH_CUDA_ARCH_LIST']}")

extra_compile_args = ["-std=c++17", "-O3"]
if enable_metrics:
    extra_compile_args.append("-DFLEXKV_ENABLE_MONITORING")
# build/include stages headers CMake copies out of fetched source trees (e.g.
# the hiredis <hiredis/hiredis.h> shim); the manifest also lists it explicitly.
include_dirs = [os.path.abspath(os.path.join(build_dir, "include"))]

# Add rpath to find libraries at runtime. Managed deps are either linked
# statically into c_ext or bundled into flexkv/lib (see CustomBuildExt); the
# $ORIGIN entries below let the loader find the bundled sonames next to the
# extension, and flexkv/__init__.py additionally puts flexkv/lib on
# LD_LIBRARY_PATH at import time.
lib_dir = os.path.join(build_dir, "lib")
if os.path.exists(lib_dir):
    extra_link_args.extend([f"-Wl,-rpath,{lib_dir}", "-Wl,-rpath,$ORIGIN"])
    extra_link_args.append("-Wl,-rpath,$ORIGIN/lib")
    extra_link_args.append("-Wl,-rpath,$ORIGIN/../lib")

if enable_cfs:
    print("ENABLE_CFS = true: compiling and link cfs related content")
    cpp_sources.append("csrc/pcfs/pcfs.cpp")
    hpp_sources.append("csrc/pcfs/pcfs.h")
    extra_link_args.append("-lhifs_client_sdk")
    extra_compile_args.append("-DFLEXKV_ENABLE_CFS")
extra_compile_args.append("-DCUDA_AVAILABLE")

nvcc_compile_args = ["-O3"]
if enable_metrics:
    nvcc_compile_args.append("-DFLEXKV_ENABLE_MONITORING")
if enable_gds:
    print("ENABLE_GDS = true: Compiling and linking GDS content")
    cpp_sources.extend([
        "csrc/gds/gds_manager.cpp",
        "csrc/gds/tp_gds_transfer_thread_group.cpp",
        "csrc/gds/layout_transform.cu",
    ])
    hpp_sources.extend([
        "csrc/gds/gds_manager.h",
        "csrc/gds/tp_gds_transfer_thread_group.h",
        "csrc/gds/layout_transform.cuh",
    ])
    extra_link_args.append("-lcufile")
    extra_compile_args.append("-DFLEXKV_ENABLE_GDS")
    nvcc_compile_args.append("-DFLEXKV_ENABLE_GDS")
if enable_p2p:
    print("ENABLE_P2P = true: Compiling and linking distributed (P2P/Redis) content")
    cpp_sources.extend([
        "csrc/dist/distributed_radix_tree.cpp",
        "csrc/dist/local_radix_tree.cpp",
        "csrc/dist/redis_meta_channel.cpp",
        "csrc/dist/lease_meta_mempool.cpp",
    ])
    extra_compile_args.append("-DFLEXKV_ENABLE_P2P")
if not enable_gds:
    print("ENABLE_GDS = false: Skipping GDS code")
if not enable_p2p:
    print("ENABLE_P2P = false: Skipping distributed (P2P/Redis) code; no libhiredis or Redis deps required")

cpp_extensions = [
    cpp_extension.CUDAExtension(
        name="flexkv.c_ext",
        sources=cpp_sources,
        library_dirs=[os.path.join(build_dir, "lib")],
        include_dirs=include_dirs,
        depends=hpp_sources,
        extra_compile_args={"nvcc": nvcc_compile_args, "cxx": extra_compile_args},
        extra_link_args=extra_link_args,
    ),
]

# Initialize ext_modules with C++ extensions
ext_modules = cpp_extensions

# Only use Cython in release mode
if not debug:
    # Compile Python modules with cythonize
    # Exclude __init__.py files and test files
    python_files = ["flexkv/**/*.py"]
    excluded_files = ["flexkv/**/__init__.py",
                      "flexkv/**/test_*.py",
                      "flexkv/**/benchmark_*.py",
                      "flexkv/benchmark/**/*.py",
                      "flexkv/benchmark/test_kvmanager.py"]
    # Import cython when debug is turned off.
    from Cython.Build import cythonize
    cythonized_modules = cythonize(
        python_files,
        exclude=excluded_files,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "profile": True,
        },
        build_dir=build_dir,  # Direct Cython to use the build directory
    )
    # Add Cython modules to ext_modules
    ext_modules.extend(cythonized_modules)
    print("Release mode: Including Cython compilation")
else:
    print("Debug mode: Skipping Cython compilation")


def _cmake_executable():
    """Resolve a cmake binary: PATH first, then the `cmake` pip package."""
    exe = shutil.which("cmake")
    if exe:
        return exe
    try:
        import cmake as _cmake_mod
        candidate = os.path.join(os.path.dirname(_cmake_mod.__file__), "data", "bin", "cmake")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    raise RuntimeError(
        "cmake was not found. FlexKV's native C++ dependencies are resolved by "
        "CMake at build time. Install it via your distro (e.g. `apt-get install "
        "cmake`) or `pip install cmake`, then rebuild."
    )


class CustomBuildExt(cpp_extension.BuildExtension):
    """Resolve native C++ deps via CMake (discover-or-fetch), feed the resolved
    include/library/link inputs into flexkv.c_ext, then build + bundle."""

    def run(self):
        self._resolve_native_deps()
        super().run()

    def _resolve_native_deps(self):
        """Configure + build the CMake project that discovers or fetches xxHash /
        liburing / hiredis / prometheus-cpp, read back its manifest, and inject
        the results into the flexkv.c_ext extension. CMake is the single source
        of truth for where these live -- setup.py performs no discovery itself."""
        cmake = _cmake_executable()
        build_abs = os.path.abspath(build_dir)
        os.makedirs(build_abs, exist_ok=True)

        configure_cmd = [
            cmake, PROJECT_ROOT,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            f"-DFLEXKV_ENABLE_MONITORING={'ON' if enable_metrics else 'OFF'}",
            f"-DFLEXKV_ENABLE_P2P={'ON' if enable_p2p else 'OFF'}",
        ]
        print(f"[flexkv] Resolving native C++ dependencies via CMake:\n  {' '.join(configure_cmd)}")
        subprocess.check_call(configure_cmd, cwd=build_abs)

        manifest_path = os.path.join(build_abs, "flexkv_pyext.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Build only the fetched targets (system-discovered deps need no build).
        targets = manifest.get("build_targets", [])
        if targets:
            build_cmd = [cmake, "--build", ".", "--config", "Release",
                         "-j", str(os.cpu_count() or 4), "--target", *targets]
            print(f"[flexkv] Building fetched dependencies: {' '.join(targets)}")
            subprocess.check_call(build_cmd, cwd=build_abs)

        # Stash bundled sonames so _bundle_into() can carry them.
        self._bundle_libs = manifest.get("bundle_libs", [])

        # Inject the resolved include/library/link inputs into c_ext only (the
        # cythonized modules must not gain native library dependencies).
        for ext in self.extensions:
            if ext.name != "flexkv.c_ext":
                continue
            ext.include_dirs = list(ext.include_dirs) + list(manifest.get("include_dirs", []))
            ext.library_dirs = list(ext.library_dirs) + list(manifest.get("library_dirs", []))
            ext.libraries = list(ext.libraries) + list(manifest.get("libraries", []))
            if self._bundle_libs:
                ext.runtime_library_dirs = list(ext.runtime_library_dirs or [])
                for rp in ("$ORIGIN", "$ORIGIN/lib"):
                    if rp not in ext.runtime_library_dirs:
                        ext.runtime_library_dirs.append(rp)

    def build_extensions(self):
        # Compile the extensions, then bundle the self-built shared deps next to
        # the built c_ext. For a wheel / non-editable install get_ext_fullpath()
        # points into build_lib, so the bundled sonames land INSIDE the wheel --
        # not merely in the working source tree. (Mirrors radixshmem/python/setup.py.)
        super().build_extensions()
        for ext in self.extensions:
            if ext.name == "flexkv.c_ext":
                ext_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
                self._bundle_into(os.path.join(ext_dir, "lib"))

    def copy_extensions_to_source(self):
        # Editable (-e) install: setuptools copies each built .so from build_lib
        # back into the source package tree. Mirror that for the bundled libs so
        # the in-place flexkv/lib is populated too.
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            if ext.name != "flexkv.c_ext":
                continue
            pkg_dir = os.path.abspath(build_py.get_package_dir("flexkv"))
            self._bundle_into(os.path.join(pkg_dir, "lib"))

    def _bundle_into(self, dest_dir):
        """Copy every runtime shared library the extension needs into dest_dir
        (a flexkv/lib) so the installed package -- wheel or editable -- is
        self-contained. Statically-linked / system-discovered deps produce no
        .so here and are simply skipped, leaving dest_dir uncreated."""
        def _ensure():
            os.makedirs(dest_dir, exist_ok=True)

        # 1) Shared libs CMake built into build/lib (if any).
        source_lib_dir = os.path.join(build_dir, "lib")
        if os.path.isdir(source_lib_dir):
            for name in os.listdir(source_lib_dir):
                if ".so" not in name:
                    continue
                src = os.path.join(source_lib_dir, name)
                if os.path.isfile(src):
                    _ensure()
                    shutil.copy2(src, os.path.join(dest_dir, name))
                    print(f"Bundled {src} -> {dest_dir}/{name}")

        # 2) Explicitly bundled sonames (e.g. a fetched liburing.so.2). Copy the
        #    real file the symlink points at so it survives independently of the
        #    build tree.
        for src in getattr(self, "_bundle_libs", []) or []:
            if not os.path.exists(src):
                continue
            _ensure()
            dst = os.path.join(dest_dir, os.path.basename(src))
            shutil.copyfile(os.path.realpath(src), dst)
            print(f"Bundled {src} -> {dst}")

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="flexkv",
    description="A global KV-Cache manager for LLM inference",
    version=get_version(),
    packages=find_packages(exclude=("benchmarks", "csrc", "examples", "tests")),
    package_data={
        "flexkv": ["*.so", "lib/*.so", "lib/*.so.*"],
    },
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=ext_modules,  # Now contains both C++ and Cython modules as needed
    cmdclass={
        "build_ext": CustomBuildExt.with_options(
            include_dirs=os.path.join(build_dir, "include"),  # Include directory for xxhash
            no_python_abi_suffix=True,
            build_temp=os.path.join(build_dir, "temp"),  # Temporary build files
        )
    },
    #python_requires=">=3.8",
    python_requires=">=3.6",
)
