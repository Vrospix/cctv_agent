"""Command-line entry point for the CCTV frame query agent."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from cctv_agent.agent import CCTVQueryAgent
from cctv_agent.database import FrameDatabase
from cctv_agent.llm_intent import DEFAULT_MODEL, OllamaIntentParser


DEFAULT_DATABASE = Path(__file__).parent / "data" / "cctv_frames.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query simulated Singapore expressway CCTV frames in natural language."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed", help="Generate the simulated frame database.")
    seed_parser.add_argument("--until", type=date.fromisoformat, default=date.today())

    ask_parser = subparsers.add_parser("ask", help="Run one natural-language request.")
    ask_parser.add_argument("request")
    ask_parser.add_argument("--limit", type=int, default=20)

    chat_parser = subparsers.add_parser(
        "chat", help="Start a conversational query session."
    )
    for command_parser in (ask_parser, chat_parser):
        command_parser.add_argument(
            "--local-ai",
            "--llm",
            action="store_true",
            dest="use_local_ai",
            help="Use the optional local Ollama intent fallback.",
        )
        command_parser.add_argument(
            "--model",
            default=DEFAULT_MODEL,
            help="Ollama model used by --local-ai.",
        )
    return parser


def print_response(response) -> None:
    print(response.message)
    for frame in response.frames:
        print(f"{frame.frame_id:>8}  {frame.captured_at.isoformat()}  {frame.camera_name}")


def main() -> None:
    args = build_parser().parse_args()
    database = FrameDatabase(args.database)
    try:
        if args.command == "seed":
            count = database.seed(until=args.until)
            print(f"Database ready: {count:,} frames at {args.database}")
            return

        today = date.today()
        database.ensure_current(today=today)
        intent_parser = (
            OllamaIntentParser(model=args.model) if args.use_local_ai else None
        )
        agent = CCTVQueryAgent(
            database,
            today=today,
            intent_parser=intent_parser,
        )
        if args.command == "ask":
            print_response(agent.ask(args.request, limit=args.limit))
            return

        print("CCTV query agent. Type 'help' for examples or 'quit' to exit.")
        while True:
            try:
                request = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if request.lower() in {"quit", "exit"}:
                break
            if request:
                print_response(agent.ask(request))
    finally:
        database.close()


if __name__ == "__main__":
    main()
