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


def wrong_type(value: object, action: argparse.Action) -> str | None:
    """Returns what the option needs if a value from the file has the wrong type, otherwise None."""
    # TOML keeps true/false, numbers and text apart, but argparse does not check the type of defaults:
    # compile = "false" (text) would switch compile on, batch_size = 2.5 would be accepted.
    if action.nargs == 0:  # a flag such as --compile
        return None if isinstance(value, bool) else "true or false"
    expected = action.type or str
    accepted = (int, float) if expected is float else expected
    if isinstance(value, bool) or not isinstance(value, accepted):
        return {int: "a whole number", float: "a number", str: "text in quotes"}.get(expected, expected.__name__)
    return None


def parse_args_with_config(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None = None, ignore_unknown: bool = False
) -> argparse.Namespace:
    """Parses the command line, with the options of --config FILE as defaults.

    ignore_unknown: skip keys this parser does not have, so a script can share config.toml with train.py.
    """
    parser.add_argument("--config", type=Path, default=None,
                        help="TOML file with options (see config.toml); command-line options override it")
    # The command line is read 3 times: first to find --config, then to find --arch (it may come from the
    # file or the command line), and last with the file's options and the table of that arch as defaults.
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
        for key in list(options):
            if key not in actions:
                if ignore_unknown:
                    del options[key]
                    continue
                parser.error(f"unknown option '{key}' in {path}. Known options: {', '.join(sorted(actions))}")
            needed = wrong_type(options[key], actions[key])
            if needed is not None:
                parser.error(f"'{key}' in {path} must be {needed}, got {options[key]!r}")

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
