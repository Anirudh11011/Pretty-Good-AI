import uuid
import aiosqlite
import os

SQLITE_PATH = os.getenv("SQLITE_PATH", "calls.db")


async def init_db():
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS call_logs (
                id TEXT PRIMARY KEY,
                call_sid TEXT,
                to_number TEXT,
                from_number TEXT,
                instructions TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS call_turns (
                id TEXT PRIMARY KEY,
                call_sid TEXT,
                role TEXT,
                content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.commit()


async def log_call(call_sid: str, to_number: str, from_number: str, instructions: str):
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute(
            "INSERT INTO call_logs (id, call_sid, to_number, from_number, instructions) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), call_sid, to_number, from_number, instructions),
        )
        await db.commit()


async def log_turn(call_sid: str, role: str, content: str):
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute(
            "INSERT INTO call_turns (id, call_sid, role, content) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), call_sid, role, content),
        )
        await db.commit()
