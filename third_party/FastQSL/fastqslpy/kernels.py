import os
import ctypes
import glob
import shutil
import cupy
import nvidia

# get the path of the current file
dirname = os.path.dirname(os.path.abspath(__file__))
src_cu_path = os.path.join(dirname, 'cu')

modules = ['TraceBline','TraceBlineAdaptive','TraceBlineAdaptiveStretch','TraceBlineScott']

def _preload_nvrtc():
    for root in getattr(nvidia, "__path__", []):
        lib_dir = os.path.join(root, "cuda_nvrtc", "lib")
        for name in ("libnvrtc-builtins.so.12.9", "libnvrtc.so.12"):
            path = os.path.join(lib_dir, name)
            if os.path.isfile(path):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


_preload_nvrtc()


def _configure_cupy_nvcc():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        candidates = [
            os.path.join("/usr/local", "cuda", "bin", "nvcc"),
            *sorted(os.path.join(path, "bin", "nvcc") for path in glob.glob("/usr/local/cuda-*")),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                nvcc = candidate
                break
    if not nvcc:
        return
    cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    os.environ.setdefault("CUDA_PATH", cuda_root)
    os.environ.setdefault("CUDA_HOME", cuda_root)
    try:
        import cupy.cuda.compiler as cupy_compiler

        cupy_compiler._nvcc = nvcc
    except Exception:
        pass

def _include_options():
    options = ["-I " + src_cu_path, "-I /usr/include", "-I /usr/include/x86_64-linux-gnu"]
    for root in getattr(nvidia, "__path__", []):
        cuda_include = os.path.join(root, "cu13", "include")
        if os.path.isdir(cuda_include):
            options.append("-I " + cuda_include)
        cuda_nvcc_include = os.path.join(root, "cuda_nvcc", "include")
        if os.path.isdir(cuda_nvcc_include):
            options.append("-I " + cuda_nvcc_include)
    return tuple(options)


def _compile_raw_module(filename, function_name):
    print('compiling kernel')
    print(dirname)
    print(__file__)
    print(src_cu_path)
    _configure_cupy_nvcc()
    with open(os.path.join(src_cu_path, filename), "rt") as traceFunc_file:
        code = traceFunc_file.read()
    options = _include_options()
    last_exc = None
    for backend in ("nvcc", "nvrtc"):
        try:
            kwargs = {"jitify": True} if backend == "nvrtc" else {}
            module = cupy.RawModule(code=code, backend=backend, options=options, **kwargs)
            return module.get_function(function_name)
        except Exception as exc:
            last_exc = exc
            print(f"{backend} compilation failed: {exc}")
    raise last_exc

def compileTraceBlineAdaptive():
    return _compile_raw_module("TraceBlineAdaptive.cu", "TraceAllBline")


def compileTraceBlineRK4():
    return _compile_raw_module("TraceBline.cu", "TraceAllBline")
