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
from typing import Any


def _checked_value(parser: argparse.ArgumentParser, action: argparse.Action, key: str, value: Any, path: str) -> Any:
    """The file's value checked and converted like the same value given on the command line."""
    if action.nargs == 0:  # on/off flags such as compile
        if not isinstance(value, bool):
            parser.error(f"'{key}' in {path} must be true or false, got {value!r}")
        return value

    if isinstance(value, (bool, dict, list)):
        parser.error(f"'{key}' in {path} must be a single number or string, got {value!r}")
    try:
        value = action.type(str(value)) if action.type else str(value)
    except (TypeError, ValueError, argparse.ArgumentTypeError):
        parser.error(f"'{key}' in {path} must be of type {action.type.__name__}, got {value!r}")
    if action.choices is not None and value not in action.choices:
        parser.error(f"'{key}' in {path} must be one of {', '.join(map(str, action.choices))}, got {value!r}")
    return value


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    # A str, not a Path: checkpoints store the arguments, and a Path saved on Linux cannot be loaded on Windows.
    parser.add_argument("--config", default=None,
                        help="TOML file with options (see config.toml); command-line options override it")
    path = parser.parse_args(argv).config
    if path is None:
        return parser.parse_args(argv)
    if not Path(path).is_file():
        parser.error(f"config file not found: {path}")

    with open(path, "rb") as file:
        config = tomllib.load(file)
    tables = {name: config.pop(name) for name in list(config) if isinstance(config[name], dict)}

    actions = {action.dest: action for action in parser._actions if action.dest not in ("help", "config")}
    architectures = actions["arch"].choices
    for name, options in [("", config), *tables.items()]:
        if name and name not in architectures:
            parser.error(f"unknown table [{name}] in {path}. Tables must be architectures: {', '.join(architectures)}")
        if name and "arch" in options:
            parser.error(f"'arch' in table [{name}] of {path}: set arch at the top level")
        for key, value in options.items():
            if key not in actions:
                parser.error(f"unknown option '{key}' in {path}. Known options: {', '.join(sorted(actions))}")
            options[key] = _checked_value(parser, actions[key], key, value, path)

    parser.set_defaults(**config)
    arch = parser.parse_args(argv).arch
    parser.set_defaults(**tables.get(arch, {}))
    return parser.parse_args(argv)
