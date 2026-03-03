"""Add selected call transcripts to a persistent "important" HTML report.

Usage examples:
    python important_call_report.py CA1234abcd
    python important_call_report.py CA1234abcd CA5678efgh --output reports/important.html

Each run:
    * checks that every provided call SID exists in the database
    * stores the running list of important call SIDs in a JSON sidecar
    * regenerates the HTML report containing all stored SIDs in the order added
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_OUTPUT = "important_calls.html"
DEFAULT_STORE = "important_call_ids.json"
SQLITE_PATH = os.getenv("SQLITE_PATH", "calls.db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append call transcripts to the important report")
    parser.add_argument(
        "call_ids",
        nargs="+",
        help="One or more call_sid values to mark as important",
    )
    parser.add_argument(
        "--db",
        default=SQLITE_PATH,
        help=f"Path to the SQLite database (default: {SQLITE_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Important report file to update (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help=f"JSON file tracking selected call IDs (default: {DEFAULT_STORE})",
    )
    return parser.parse_args()


def load_call_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [str(item) for item in raw if isinstance(item, (str, int)) and str(item).strip()]
    except json.JSONDecodeError:
        pass
    return []


def save_call_ids(path: Path, call_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(call_ids), indent=2), encoding="utf-8")


def call_exists(conn: sqlite3.Connection, call_sid: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM call_logs WHERE call_sid = ? LIMIT 1",
        (call_sid,),
    ).fetchone()
    return bool(row)


def collect_calls(conn: sqlite3.Connection, call_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    calls: List[Dict[str, Any]] = []
    missing: List[str] = []

    for call_sid in call_ids:
        row = conn.execute(
            """
            SELECT call_sid, to_number, from_number, instructions, created_at
            FROM call_logs
            WHERE call_sid = ?
            """,
            (call_sid,),
        ).fetchone()

        if not row:
            missing.append(call_sid)
            continue

        turn_rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM call_turns
            WHERE call_sid = ?
            ORDER BY datetime(created_at) ASC
            """,
            (call_sid,),
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

    return calls, missing


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


def render_html(
    calls: Sequence[Dict[str, Any]],
    tracked_ids: Sequence[str],
    missing_ids: Sequence[str],
    updated_at: datetime,
) -> str:
    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang=\"en\">")
    parts.append("<head>")
    parts.append("    <meta charset=\"UTF-8\">")
    parts.append("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    parts.append("    <title>Important Call Transcripts</title>")
    parts.append("    <style>")
    parts.append("        body { font-family: 'Segoe UI', sans-serif; background: #101320; color: #f9fbff; margin: 0; padding: 32px; }")
    parts.append("        h1 { margin-top: 0; }")
    parts.append("        .meta { color: #c7cad6; margin-bottom: 16px; }")
    parts.append("        .call-card { background: #1c2133; border-radius: 14px; padding: 20px; margin-bottom: 24px; box-shadow: 0 18px 30px rgba(0,0,0,0.35); border: 1px solid #2e3651; }")
    parts.append("        .call-card h2 { margin-top: 0; color: #7dcfff; }")
    parts.append("        .call-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 16px 0 18px; }")
    parts.append("        .label { font-size: 12px; text-transform: uppercase; color: #8e94ab; letter-spacing: 0.08em; }")
    parts.append("        .value { font-size: 16px; color: #fefefe; font-weight: 600; }")
    parts.append("        .turn { border-left: 4px solid transparent; padding: 12px 18px; margin-bottom: 10px; background: #14192b; border-radius: 10px; }")
    parts.append("        .turn.human { border-color: #32d297; }")
    parts.append("        .turn.ai { border-color: #8b6eff; }")
    parts.append("        .turn .role { font-size: 13px; font-weight: 700; }")
    parts.append("        .turn .timestamp { font-size: 12px; color: #9aa0bd; margin-left: 8px; }")
    parts.append("        .empty { font-style: italic; color: #9aa0bd; }")
    parts.append("    </style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("    <h1>Important Call Transcripts</h1>")
    parts.append(
        "    <p class=\"meta\">Updated at " + html.escape(updated_at.strftime("%Y-%m-%d %H:%M:%SZ")) + " · Tracking " + str(len(tracked_ids)) + " call(s).</p>"
    )

    if missing_ids:
        parts.append(
            "    <p class=\"meta\">Missing in database: "
            + ", ".join(html.escape(mid) for mid in missing_ids)
            + ".</p>"
        )

    if not calls:
        parts.append("    <p class=\"empty\">No transcripts available yet. Add call IDs to populate this report.</p>")
    else:
        for call in calls:
            turns = call["turns"]
            parts.append("    <section class=\"call-card\">")
            parts.append(f"        <h2>{html.escape(call['call_sid'] or '(unknown)')}</h2>")
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
                parts.append("        <p class=\"empty\">No conversation turns stored for this call.</p>")
            else:
                for turn in turns:
                    label = role_label(turn["role"])
                    css_class = "human" if label == "Human" else "ai"
                    parts.append(f"        <div class=\"turn {css_class}\">")
                    parts.append(
                        f"            <div class=\"role\">{label}<span class=\"timestamp\">{html.escape(turn['created_at'] or '')}</span></div>"
                    )
                    parts.append(f"            <div>{format_content(turn['content'])}</div>")
                    parts.append("        </div>")

            parts.append("    </section>")

    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


def write_report(path: Path, html_body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_body, encoding="utf-8")
    return path.resolve()


def main() -> None:
    args = parse_args()
    call_ids = [cid.strip() for cid in args.call_ids if cid.strip()]
    if not call_ids:
        raise SystemExit("No valid call IDs provided")

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        store_path = Path(args.store)
        tracked_ids = load_call_ids(store_path)

        added: List[str] = []
        skipped: List[str] = []
        for cid in call_ids:
            if not call_exists(conn, cid):
                skipped.append(cid)
                continue
            if cid not in tracked_ids:
                tracked_ids.append(cid)
                added.append(cid)

        save_call_ids(store_path, tracked_ids)

        calls, missing = collect_calls(conn, tracked_ids)
        html_body = render_html(calls, tracked_ids, missing, datetime.utcnow())
        output_path = write_report(Path(args.output), html_body)

        if added:
            print("Added call IDs:", ", ".join(added))
        if skipped:
            print("Skipped (not found):", ", ".join(skipped))
        if missing:
            print("Missing in DB (kept for later):", ", ".join(missing))
        print(f"Important report written to: {output_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
