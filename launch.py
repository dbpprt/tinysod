import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable

import torch
from rich.logging import RichHandler

from src import training as training


@dataclass
class Command:
    description: str
    add_arguments: Callable[[ArgumentParser], None]
    handler: Callable[[Namespace], None] = None


commands = {
    "training": Command(description=None, add_arguments=training.add_arguments, handler=training.main),
    # "inference": Command(description=None, command=None, handler=None),
}


def main() -> None:
    parser = ArgumentParser(prog="tinysod", description="work in progress transformer based salient object detection")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set logging level to debug",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for random number generators. If None, a random seed is used."
    )

    # we first parse the debug argument only to initialize the logger
    # and continue initializing the argument parser later on
    params, _ = parser.parse_known_args()

    logging.basicConfig(
        level="NOTSET" if params.debug else "WARNING",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    logger = logging.getLogger(__name__)

    if params.seed is not None:
        torch.manual_seed(params.seed)
        logger.info(f"Set seed to {params.seed}")

    sub_parser = parser.add_subparsers(dest="command", help="Choose either training/inference.")

    for command_name, command in commands.items():
        command_parser = sub_parser.add_parser(command_name, help=command.description)
        command.add_arguments(command_parser)

    params = parser.parse_args()

    handler = commands[params.command].handler
    if handler is not None:
        logger.info(f"Running command {params.command} with parameters: {params}")
        handler(params)
    else:
        # this should actually be unreachable code
        logger.error(f"Command {params.command} not implemented.")
        logger.info(f"Available commands: {list(commands.keys())}")
        logger.info("Use --help to see the help for each command.")
        exit(1)


if __name__ == "__main__":
    main()
