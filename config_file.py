"""Reads train.py options from a TOML file (see config.toml).

Top-level keys are train.py option names (python train.py --help), e.g. num_epochs = 50.
A table named after an architecture, e.g. [unet_edsr], holds options used only when that
architecture is trained; they override the top-level keys. The file's options become argparse
defaults, so the command line overrides them, e.g. --arch sgarnet switches the architecture.
"""
import argparse
import tomllib
from collections.abc import Sequence
from pathlib import Path


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser.add_argument("--config", type=Path, default=None,
                        help="TOML file with options (see config.toml); command-line options override it")
    path = parser.parse_args(argv).config
    if path is None:
        return parser.parse_args(argv)
    if not path.is_file():
        parser.error(f"config file not found: {path}")

    with open(path, "rb") as file:
        config = tomllib.load(file)
    tables = {name: config.pop(name) for name in list(config) if isinstance(config[name], dict)}

    actions = {action.dest: action for action in parser._actions if action.dest not in ("help", "config")}
    architectures = actions["arch"].choices
    for name, options in [("", config), *tables.items()]:
        if name and name not in architectures:
            parser.error(f"unknown table [{name}] in {path}. Tables must be architectures: {', '.join(architectures)}")
        for key in options:
            if key not in actions:
                parser.error(f"unknown option '{key}' in {path}. Known options: {', '.join(sorted(actions))}")

    parser.set_defaults(**config)
    arch = parser.parse_args(argv).arch
    parser.set_defaults(**tables.get(arch, {}))
    args = parser.parse_args(argv)

    # argparse checks choices only for command-line values, so check the file's values here.
    for key in {**config, **tables.get(arch, {})}:
        value, choices = getattr(args, key), actions[key].choices
        if choices is not None and value not in choices:
            parser.error(f"'{key}' in {path} must be one of {', '.join(map(str, choices))}, got {value!r}")
    return args
