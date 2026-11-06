"""Check or apply the MemT patches to an explicit SGLang source checkout.

The patches are split by responsibility so an operator installs only what they
enable:

  sglang-decode-owners.patch  required by the Flash memory decode graph
  sglang-stable-topk.patch    optional: DSV4 indexer stable top-k
  sglang-dsv4-numerics.patch  optional: strict numerics tier

Nothing is applied implicitly: a checkout is named, and each patch is verified
with `git apply --check` before anything is written.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PATCHES = {
    "decode-owners": "sglang-decode-owners.patch",
    "stable-topk": "sglang-stable-topk.patch",
    "dsv4-numerics": "sglang-dsv4-numerics.patch",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument(
        "which",
        nargs="+",
        choices=sorted(PATCHES) + ["all"],
        help="Patch groups to check or apply",
    )
    parser.add_argument("--apply", action="store_true", help="Apply after a check")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    checkout = args.checkout.expanduser().resolve()
    if not (checkout / "python/sglang").is_dir():
        parser.error("checkout must contain python/sglang")

    manifest = json.loads((root / "patches/manifest.json").read_text())
    groups = sorted(PATCHES) if "all" in args.which else args.which
    for group in groups:
        path = root / "patches" / PATCHES[group]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = manifest["patch_sha256"][PATCHES[group]]
        if digest != expected:
            parser.error(f"{path.name} does not match the manifest")
        command = ["git", "-C", str(checkout), "apply"]
        subprocess.run([*command, "--check", str(path)], check=True)
        if args.apply:
            subprocess.run([*command, str(path)], check=True)
        print(f"{path.name}: {'applied' if args.apply else 'can be applied'}")


if __name__ == "__main__":
    main()
