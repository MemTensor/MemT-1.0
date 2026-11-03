"""Check or apply the bundled patch to an explicit SGLang source checkout."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument(
        "--apply", action="store_true", help="Apply after a successful check"
    )
    args = parser.parse_args()
    checkout = args.checkout.expanduser().resolve()
    patch = Path(__file__).resolve().parents[1] / "patches/sglang-memory-graph.patch"
    if not (checkout / "python/sglang").is_dir():
        parser.error("checkout must contain python/sglang")
    command = ["git", "-C", str(checkout), "apply"]
    subprocess.run([*command, "--check", str(patch)], check=True)
    if args.apply:
        subprocess.run([*command, str(patch)], check=True)
    print("SGLang patch " + ("applied" if args.apply else "can be applied"))


if __name__ == "__main__":
    main()
