"""Command-line entry point.

Subcommands:
    seed    Build the synthetic analytics database.
    serve   Run the HTTP API and chat interface.
    ask     Answer one question from the terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys

from config.settings import load_app_config
from database.seed import seed_database
from utils.errors import AnalystError
from utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-analyst", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="build the synthetic database")
    seed.add_argument("--stations", type=int, default=20)
    seed.add_argument("--riders", type=int, default=600)

    serve = subparsers.add_parser("serve", help="run the API and chat UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    ask = subparsers.add_parser("ask", help="answer one question and exit")
    ask.add_argument("question", help="the question to answer")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_app_config(require_api_key=args.command in ("serve", "ask"))
    setup_logging(config.log_level)

    if args.command == "seed":
        counts = seed_database(config.database.path, args.stations, args.riders)
        print(f"Seeded {config.database.path}")
        for table, count in sorted(counts.items()):
            print(f"  {table:<10} {count:>7,} rows")
        return 0

    if args.command == "serve":
        import uvicorn

        from api.app import create_app

        print(f"Chat interface: http://{args.host}:{args.port}")
        uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")
        return 0

    if args.command == "ask":
        from agents.orchestrator import Orchestrator
        from llm.factory import build_llm_client

        config.log_active_config()
        orchestrator = Orchestrator(build_llm_client(config.llm), config)
        try:
            result = orchestrator.ask(args.question)
        except AnalystError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(result.answer)
        print(
            f"\n[{', '.join(result.agents_called) or 'no delegation'}"
            f" | {result.iterations} turns | {result.total_tokens:,} tokens]",
            file=sys.stderr,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
