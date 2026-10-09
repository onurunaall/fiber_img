"""Reads train.py options from a TOML file (see configs/).

Each key of the file is a train.py option name without the leading dashes, e.g. num_epochs = 50.
The values go through argparse like typed options, so types and choices are checked the same way,
and options given on the command line override the file.
"""
import argparse
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _option_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Option actions by name: both the dest (start_epoch) and the flag without dashes (start-epoch)."""
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if not action.option_strings or isinstance(action, argparse._HelpAction):
            continue
        actions[action.dest] = action
        for option in action.option_strings:
            actions[option.lstrip("-")] = action
    return actions


def _config_tokens(parser: argparse.ArgumentParser, config: dict[str, Any], path: Path) -> list[str]:
    actions = _option_actions(parser)
    tokens: list[str] = []
    for key, value in config.items():
        action = actions.get(key)
        if action is None or action.dest == "config":
            known = ", ".join(sorted(a.dest for a in set(actions.values()) if a.dest != "config"))
            parser.error(f"unknown option '{key}' in {path}. Known options: {known}")

        flag = action.option_strings[-1]
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                parser.error(f"'{key}' in {path} must be true or false, got {value!r}")
            if value:
                tokens.append(flag)
        elif isinstance(value, (bool, dict, list)):
            parser.error(f"'{key}' in {path} must be a single number or string, got {value!r}")
        else:
            tokens += [flag, str(value)]
    return tokens


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses --config FILE first, then the file's options, then the command line (the command line wins)."""
    parser.add_argument("--config", type=Path, default=None,
                        help="TOML file with options (see configs/); command-line options override it")

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    known, _ = pre_parser.parse_known_args(argv)

    tokens: list[str] = []
    if known.config is not None:
        if not known.config.is_file():
            parser.error(f"config file not found: {known.config}")
        with open(known.config, "rb") as file:
            tokens = _config_tokens(parser, tomllib.load(file), known.config)

    command_line = list(argv) if argv is not None else sys.argv[1:]
    return parser.parse_args(tokens + command_line)
