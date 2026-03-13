import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

from .data import write_submission
from .search import Search, build


def main():
    parser = argparse.ArgumentParser(description="Find timestamped scenes in the Okko catalog.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "predict"):
        command = subcommands.add_parser(name)
        command.add_argument("--data", type=Path, default=Path("data"))
        command.add_argument("--index", type=Path, default=Path("artifacts"))
        if name == "predict":
            command.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    try:
        if args.command == "index":
            count = build(args.data, args.index)
            print(f"Indexed {count} scenes in {args.index}")
        else:
            with closing(Search(args.index)) as search:
                write_submission(args.data, args.out, search.top)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"okko: {error}\n")


if __name__ == "__main__":
    main()
