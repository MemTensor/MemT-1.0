# SGLang dependency

The server adapter targets SGLang 0.5.16, reference commit
`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` in `sgl-project/sglang`.
Flash memory decode capture additionally needs the bundled source patch. It
provides exact stable C4 selection, capture tensor ownership, and optional
DSV4 deterministic arithmetic. Default graph ownership is resolved separately
from deterministic arithmetic by the MemT launcher.

```bash
git clone https://github.com/sgl-project/sglang.git sglang-source
git -C sglang-source checkout fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1
python tools/apply_sglang_patch.py /path/to/sglang-source
python tools/apply_sglang_patch.py /path/to/sglang-source --apply
uv pip install '.[server]'
export PYTHONPATH=/path/to/sglang-source/python${PYTHONPATH:+:$PYTHONPATH}
```

Run commands from `inference/`. The helper checks applicability before making
changes and does not edit an installed site-packages tree. A failed check
requires reviewing local changes or selecting a fresh checkout. The dependency
patch is distributed with the repository; install instructions do not depend on
private overlay directories. CPU unit checks cannot qualify this patched CUDA
implementation. See [validation scope](VALIDATION.md).

Use a CUDA compiler and headers from the same toolkit when JIT compiling. Record
CUDA_HOME, nvcc version, PyTorch CUDA runtime and driver in GPU test evidence.
A compiler/header mismatch can fail only on first use of a new shape; successful
weight loading alone does not establish readiness. No machine-specific CUDA
path or GPU count is required by this installation recipe.

## Optional deterministic arithmetic

The default launcher selects fast arithmetic. Set `MEMT_NUMERICS=strict` to
request the model-specific deterministic profile. The launcher adds the native
deterministic flag and, for Flash only, resolves its DSV4 profile defaults.
Explicit conflicting environment settings are preserved and reported as errors.
Nano does not require DeepSeek-specific switches. Tensor ownership remains
required in either mode; strict is not needed merely to enable a decode graph.

When `MEMT_NUMERICS=auto` inherits only part of a Flash arithmetic profile, the
reported effective tier is `custom`; it carries no batch-invariance claim.
DeepSeek-specific flags do not change or gate Nano's numerical policy.
