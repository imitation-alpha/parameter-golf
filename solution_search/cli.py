from __future__ import annotations

import argparse
import json

from .engine import SearchEngine


def _cmd_run(args: argparse.Namespace) -> None:
    engine = SearchEngine(args.config)
    runs = engine.run(iterations=args.iterations)
    print(json.dumps(runs[-1], indent=2))


def _cmd_leaderboard(args: argparse.Namespace) -> None:
    engine = SearchEngine(args.config)
    leaderboard = engine.leaderboard(limit=args.limit)
    for item in leaderboard:
        primary = item.get("metrics", {}).get(item.get("primary_metric", ""))
        print(
            "{candidate_id}\tscore={score:.6f}\tprimary={primary}\tfeasible={feasible}\ttitle={title}".format(
                candidate_id=item.get("candidate_id"),
                score=float(item.get("score", 0.0)),
                primary=primary,
                feasible=item.get("feasible"),
                title=item.get("proposal", {}).get("title"),
            )
        )


def _cmd_sync_knowledge(args: argparse.Namespace) -> None:
    engine = SearchEngine(args.config)
    cards = engine.refresh_knowledge(force=args.force)
    print(json.dumps(cards, indent=2))


def _cmd_show_knowledge(args: argparse.Namespace) -> None:
    engine = SearchEngine(args.config)
    print(engine.knowledge.format_cards(limit=args.limit))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-driven solution search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the search loop")
    run_parser.add_argument("--config", required=True, help="Path to campaign JSON config")
    run_parser.add_argument("--iterations", type=int, default=None, help="Override iteration count")
    run_parser.set_defaults(func=_cmd_run)

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Show top feasible candidates")
    leaderboard_parser.add_argument("--config", required=True, help="Path to campaign JSON config")
    leaderboard_parser.add_argument("--limit", type=int, default=10, help="Max rows to print")
    leaderboard_parser.set_defaults(func=_cmd_leaderboard)

    sync_parser = subparsers.add_parser("sync-knowledge", help="Fetch and summarize external knowledge sources")
    sync_parser.add_argument("--config", required=True, help="Path to campaign JSON config")
    sync_parser.add_argument("--force", action="store_true", help="Refresh all sources even if cached")
    sync_parser.set_defaults(func=_cmd_sync_knowledge)

    show_parser = subparsers.add_parser("show-knowledge", help="Show cached knowledge cards")
    show_parser.add_argument("--config", required=True, help="Path to campaign JSON config")
    show_parser.add_argument("--limit", type=int, default=8, help="Max cards to print")
    show_parser.set_defaults(func=_cmd_show_knowledge)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
