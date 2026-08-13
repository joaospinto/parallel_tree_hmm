#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rocm_path="${ROCM_PATH:-}"
architecture="${TREE_HMM_ROCM_ARCH:-gfx942}"
operation="${1:-build}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "${repo_dir}"
if [[ -z "${rocm_path}" ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ROCm compilation is supported on Linux only" >&2
    exit 2
  fi
  compiler_file="$(bazel cquery --output=files @rocm_sdk//:amdclang | \
    tail -n 1)"
  execution_root="$(bazel info execution_root)"
  if [[ -z "${compiler_file}" ||
        ! -f "${execution_root}/${compiler_file}" ]]; then
    echo "the pinned ROCm SDK did not expose amdclang" >&2
    exit 2
  fi
  compiler="$(realpath "${execution_root}/${compiler_file}")"
  rocm_path="$(cd "$(dirname "${compiler}")/../.." && pwd)"
fi

compiler_directory=""
for candidate in "${rocm_path}/bin" "${rocm_path}/llvm/bin"; do
  if [[ -x "${candidate}/amdclang" && -x "${candidate}/amdclang++" ]]; then
    compiler_directory="${candidate}"
    break
  fi
done
if [[ -z "${compiler_directory}" ]]; then
  echo "ROCm compilers were not found under ${rocm_path}" >&2
  echo "set ROCM_PATH to the ROCm installation root" >&2
  exit 2
fi
cc="${compiler_directory}/amdclang"
cxx="${compiler_directory}/amdclang++"

common=(
  --config=rocm
  --repo_env="CC=${cc}"
  --repo_env="CXX=${cxx}"
  --action_env="ROCM_PATH=${rocm_path}"
  --rocm_arch="${architecture}"
)

case "${operation}" in
  build)
    exec bazel build "${common[@]}" //:rocm_test "$@"
    ;;
  test)
    exec bazel test "${common[@]}" //:rocm_test "$@"
    ;;
  *)
    echo "usage: $0 [build|test] [additional Bazel options]" >&2
    exit 2
    ;;
esac
