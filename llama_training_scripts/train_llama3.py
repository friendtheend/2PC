from __future__ import annotations

import argparse

from .config import add_common_args, apply_cli_overrides, load_config


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Minimal Llama3 training."))
    args = parser.parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)

    from .training import train

    metrics = train(cfg)
    print(metrics)


if __name__ == "__main__":
    main()
