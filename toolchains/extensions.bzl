"""Pinned accelerator SDK repositories selected for the host operating system."""

load("@rules_cuda//cuda/private:repositories.bzl", "cuda_component", "cuda_toolkit")

_CUDA_VERSION = "12.8.1"
_CUDA_COMPONENTS = {
    "cccl": {
        "archive": "cuda_cccl-linux-x86_64-12.8.90-archive",
        "sha256": "0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b",
        "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz",
        "version": "12.8.90",
    },
    "cudart": {
        "archive": "cuda_cudart-linux-x86_64-12.8.90-archive",
        "sha256": "8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68",
        "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz",
        "version": "12.8.90",
    },
    "nvcc": {
        "archive": "cuda_nvcc-linux-x86_64-12.8.93-archive",
        "sha256": "9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3",
        "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz",
        "version": "12.8.93",
    },
}

_ROCM_VERSION = "7.2.3"
_ROCM_BASE_URL = "https://repo.radeon.com/rocm/apt/7.2.3/"
_ROCM_PACKAGES = [
    ("rocm-core", "pool/main/r/rocm-core/rocm-core_7.2.3.70203-90~22.04_amd64.deb", "778444878422e65c0a36e18a3fa19d96fb6f004f976146633b72abb39316b5e2"),
    ("rocm-llvm", "pool/main/r/rocm-llvm/rocm-llvm_22.0.0.26084.70203-90~22.04_amd64.deb", "4c406e184f88949cea60869949454e5392e1cbd9480c4c87274f7b59e9f810e5"),
    ("rocm-device-libs", "pool/main/r/rocm-device-libs/rocm-device-libs_1.0.0.70203-90~22.04_amd64.deb", "5a8fec7811bf867ca6b0cbf1f61c08eff84d71113d60f12ade120a155b119a77"),
    ("hipcc", "pool/main/h/hipcc/hipcc_1.1.1.70203-90~22.04_amd64.deb", "b12d171bc413f5ba39877eac0f3fb457e8ea5d5dc10289d5eb181e72d3decb3a"),
    ("hip-runtime-amd", "pool/main/h/hip-runtime-amd/hip-runtime-amd_7.2.53211.70203-90~22.04_amd64.deb", "086e7cd1d5b65af98260df75361ce79e6847d97aab62f1f801ea2742855e5168"),
    ("hip-dev", "pool/main/h/hip-dev/hip-dev_7.2.53211.70203-90~22.04_amd64.deb", "ab43396e35fca205cfb6f6c401290d9d32109be160e28ba259034eb2944ed63d"),
    ("comgr", "pool/main/c/comgr/comgr_3.0.0.70203-90~22.04_amd64.deb", "422601e978e73bce4d4e20093272286abca3bc784d4ed4bc53a668a54d7c77ff"),
    ("hsa-rocr", "pool/main/h/hsa-rocr/hsa-rocr_1.18.0.70203-90~22.04_amd64.deb", "d2c571de32740cbc149c26af8899ad14005dcb1e09b88f54ebf4ec5e16f9b73c"),
    ("hsa-rocr-dev", "pool/main/h/hsa-rocr-dev/hsa-rocr-dev_1.18.0.70203-90~22.04_amd64.deb", "01e6a4a12a1717d87bf2befc32164f9acd91dd7bfe8d5ddb730776befd474a53"),
    ("rocprofiler-register", "pool/main/r/rocprofiler-register/rocprofiler-register_0.6.0.70203-90~22.04_amd64.deb", "e52fae675771057fdf9f81fb487c3fffe83b8d1dd72fb82e0d42b8099475d837"),
    ("rocminfo", "pool/main/r/rocminfo/rocminfo_1.0.0.70203-90~22.04_amd64.deb", "b7d5b244adef84c1e0da1a905a9dd2b939391e280544913469a4ac1d11687ea7"),
]

def _rocm_sdk_impl(repository_ctx):
    dpkg_deb = repository_ctx.which("dpkg-deb")
    if dpkg_deb == None:
        fail("the pinned ROCm SDK repository requires dpkg-deb on Linux")
    for name, path, sha256 in _ROCM_PACKAGES:
        archive = name + ".deb"
        repository_ctx.download(
            output = archive,
            sha256 = sha256,
            url = _ROCM_BASE_URL + path,
        )
        result = repository_ctx.execute([dpkg_deb, "-x", archive, "."])
        if result.return_code != 0:
            fail("extracting {} failed:\n{}".format(archive, result.stderr))
        repository_ctx.delete(archive)
    root = "opt/rocm-{}".format(_ROCM_VERSION)
    compiler_directory = None
    for candidate in [
        root + "/llvm/bin",
        root + "/lib/llvm/bin",
        root + "/bin",
    ]:
        if (
            repository_ctx.path(candidate + "/amdclang").exists and
            repository_ctx.path(candidate + "/amdclang++").exists
        ):
            compiler_directory = candidate
            break
    if compiler_directory == None:
        fail(
            "the extracted ROCm {} SDK contains no amdclang/amdclang++ pair".format(
                _ROCM_VERSION,
            ),
        )
    launcher_directory = root + "/toolchain/bin"
    for entry_point in ["amdclang", "amdclang++"]:
        repository_ctx.file(
            launcher_directory + "/" + entry_point,
            """#!/usr/bin/env bash
set -euo pipefail
sdk_root="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd -P)"
entry_point="${{sdk_root}}/{compiler_directory}/{entry_point}"
exec -a "${{entry_point}}" "${{entry_point}}" "$@"
""".format(
                compiler_directory = compiler_directory[len(root) + 1:],
                entry_point = entry_point,
            ),
            executable = True,
        )
    repository_ctx.file(
        "BUILD.bazel",
        """load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [\"//visibility:public\"])

filegroup(name = \"sdk\", srcs = glob([\"{root}/**\"], allow_empty = False))
filegroup(name = \"amdclang\", srcs = [\"{launcher_directory}/amdclang\"])
filegroup(name = \"amdclang++\", srcs = [\"{launcher_directory}/amdclang++\"])

cc_import(
    name = \"amdhip64\",
    shared_library = \"{root}/lib/libamdhip64.so\",
)

cc_library(
    name = \"hip_runtime\",
    hdrs = glob([\"{root}/include/**\"], allow_empty = False),
    includes = [\"{root}/include\"],
    deps = [\":amdhip64\"],
)
""".format(
            launcher_directory = launcher_directory,
            root = root,
        ),
    )

rocm_sdk_repository = repository_rule(
    implementation = _rocm_sdk_impl,
)

def _empty_rocm_sdk_impl(repository_ctx):
    repository_ctx.file(
        "BUILD.bazel",
        """load("@rules_cc//cc:cc_library.bzl", "cc_library")

filegroup(name = \"sdk\", srcs = [])
filegroup(name = \"amdclang\", srcs = [])
filegroup(name = \"amdclang++\", srcs = [])
cc_library(name = \"hip_runtime\")
""",
    )

empty_rocm_sdk_repository = repository_rule(
    implementation = _empty_rocm_sdk_impl,
)

def _accelerator_sdks_impl(module_ctx):
    if module_ctx.os.name.startswith("linux"):
        mapping = {}
        for component, spec in _CUDA_COMPONENTS.items():
            repository_name = "cuda_{}_v{}".format(
                component,
                spec["version"].replace(".", "_"),
            )
            cuda_component(
                name = repository_name,
                component_name = component,
                descriptive_name = component,
                sha256 = spec["sha256"],
                strip_prefix = spec["archive"],
                urls = [spec["url"]],
                version = spec["version"],
            )
            mapping[component] = "@" + repository_name
        cuda_toolkit(
            name = "cuda",
            components_mapping = mapping,
            nvcc_version = _CUDA_COMPONENTS["nvcc"]["version"],
            version = _CUDA_VERSION,
        )
        rocm_sdk_repository(name = "rocm_sdk")
    else:
        # CUDA/HIP are Linux-only in this project. A disabled local CUDA
        # repository keeps non-CUDA targets analyzable on macOS.
        cuda_toolkit(name = "cuda", toolkit_path = "")
        empty_rocm_sdk_repository(name = "rocm_sdk")

accelerator_sdks = module_extension(
    implementation = _accelerator_sdks_impl,
)
