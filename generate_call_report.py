"""Generate an HTML report of the most recent Twilio calls and their transcripts.

Usage:
    python generate_call_report.py
    python generate_call_report.py --limit 5 --output reports/latest_calls.html

The script reads from the same SQLite database used by the app (default ``calls.db`` or
``SQLITE_PATH`` env override) and builds a standalone HTML report containing the last N
calls grouped by ``call_sid`` with the full human/AI conversation for each.
"""

from __future__ import annotations

import argparse
import html
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_OUTPUT = "call_report.html"
DEFAULT_LIMIT = 10
SQLITE_PATH = os.getenv("SQLITE_PATH", "calls.db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML report for recent calls")
    parser.add_argument(
        "--db",
        default=SQLITE_PATH,
        help=f"Path to the SQLite database (default: {SQLITE_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of most recent calls to include",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Where to write the HTML report (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def fetch_last_calls(db_path: str, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        call_rows = conn.execute(
            """
            SELECT call_sid, to_number, from_number, instructions, created_at
            FROM call_logs
            WHERE call_sid IS NOT NULL
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        calls: List[Dict[str, Any]] = []
        for row in call_rows:
            turn_rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM call_turns
                WHERE call_sid = ?
                ORDER BY datetime(created_at) ASC
                """,
                (row["call_sid"],),
            ).fetchall()

            calls.append(
                {
                    "call_sid": row["call_sid"],
                    "to_number": row["to_number"] or "",
                    "from_number": row["from_number"] or "",
                    "instructions": row["instructions"] or "",
                    "created_at": row["created_at"],
                    "turns": [
                        {
                            "role": turn_row["role"] or "unknown",
                            "content": turn_row["content"] or "",
                            "created_at": turn_row["created_at"],
                        }
                        for turn_row in turn_rows
                    ],
                }
            )

        return calls
    finally:
        conn.close()


def role_label(role: str) -> str:
    normalized = (role or "").strip().lower()
    if normalized in {"assistant", "ai", "bot"}:
        return "AI"
    if normalized in {"user", "caller", "human"}:
        return "Human"
    return normalized.title() or "Unknown"


def format_content(text: str) -> str:
    if not text:
        return "<em>(empty)</em>"
    return "<br>".join(html.escape(text).splitlines())


def render_html(calls: List[Dict[str, Any]], limit: int) -> str:
    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang=\"en\">")
    parts.append("<head>")
    parts.append("    <meta charset=\"UTF-8\">")
    parts.append("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    parts.append("    <title>Call Transcript Report</title>")
    parts.append("    <style>")
    parts.append("        body { font-family: 'Segoe UI', sans-serif; background: #f4f6fb; margin: 0; padding: 32px; }")
    parts.append("        h1 { margin-top: 0; }")
    parts.append("        .call-card { background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 6px 20px rgba(0,0,0,0.06); }")
    parts.append("        .call-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 12px 0 16px; }")
    parts.append("        .label { font-size: 12px; text-transform: uppercase; color: #707486; letter-spacing: 0.05em; }")
    parts.append("        .value { font-size: 15px; color: #111322; font-weight: 600; }")
    parts.append("        .turn { border-left: 4px solid transparent; padding: 12px 16px; margin-bottom: 8px; background: #f9f9fc; border-radius: 8px; }")
    parts.append("        .turn.human { border-color: #21a179; }")
    parts.append("        .turn.ai { border-color: #4f6bed; }")
    parts.append("        .turn .role { font-size: 13px; font-weight: 700; margin-bottom: 4px; }")
    parts.append("        .turn .timestamp { font-size: 12px; color: #6b6f82; margin-left: 6px; }")
    parts.append("        .empty { font-style: italic; color: #6b6f82; }")
    parts.append("    </style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("    <h1>Recent Call Transcripts</h1>")
    parts.append(
        f"    <p>Showing up to {limit} most recent calls. Total included: {len(calls)}.</p>"
    )

    if not calls:
        parts.append("    <p class=\"empty\">No calls found in the database.</p>")
    else:
        for call in calls:
            turns = call["turns"]
            parts.append("    <section class=\"call-card\">")
            parts.append(f"        <h2>Call SID: {html.escape(call['call_sid'] or '(unknown)')}</h2>")
            parts.append("        <div class=\"call-meta\">")
            parts.append(
                f"            <div><div class=\"label\">From</div><div class=\"value\">{html.escape(call['from_number'] or 'Unknown')}</div></div>"
            )
            parts.append(
                f"            <div><div class=\"label\">To</div><div class=\"value\">{html.escape(call['to_number'] or 'Unknown')}</div></div>"
            )
            parts.append(
                f"            <div><div class=\"label\">Started</div><div class=\"value\">{html.escape(call['created_at'] or 'Unknown')}</div></div>"
            )
            parts.append("        </div>")

            if call["instructions"]:
                parts.append("        <p><strong>Instructions:</strong> " + html.escape(call["instructions"]) + "</p>")

            if not turns:
                parts.append("        <p class=\"empty\">No conversation turns recorded for this call.</p>")
            else:
                for turn in turns:
                    cls = "human" if role_label(turn["role"]) == "Human" else "ai"
                    parts.append(f"        <div class=\"turn {cls}\">")
                    parts.append(
                        f"            <div class=\"role\">{role_label(turn['role'])}"
                        f" <span class=\"timestamp\">{html.escape(turn['created_at'] or '')}</span></div>"
                    )
                    parts.append(f"            <div>{format_content(turn['content'])}</div>")
                    parts.append("        </div>")

            parts.append("    </section>")

    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)


def write_report(output_path: str, html_body: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_body, encoding="utf-8")
    return path.resolve()


def main() -> None:
    args = parse_args()
    db_path = args.db

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found at {db_path}. Run the app once to create it.")

    calls = fetch_last_calls(db_path, args.limit)
    html_doc = render_html(calls, args.limit)
    output_path = write_report(args.output, html_doc)
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
