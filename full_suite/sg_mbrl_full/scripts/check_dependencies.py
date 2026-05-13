from __future__ import annotations

import importlib.util
import platform
import sys


REQUIRED = ["numpy", "pandas", "matplotlib", "torch", "gymnasium", "stable_baselines3", "tqdm"]


def main() -> None:
    print(f"python={sys.version.split()[0]} platform={platform.platform()}")
    missing = []
    for name in REQUIRED:
        ok = importlib.util.find_spec(name) is not None
        print(f"{name}: {'ok' if ok else 'missing'}")
        if not ok:
            missing.append(name)
    if missing:
        raise SystemExit(f"Missing dependencies: {', '.join(missing)}")


if __name__ == "__main__":
    main()

