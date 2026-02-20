"""Standalone harness to verify Deepgram STT/TTS and Groq LLM integration.

Flow:
1. Use Deepgram TTS to synthesize a spoken version of a test utterance.
2. Stream those mulaw/8k audio bytes back into Deepgram STT.
3. Collect the final transcript and feed it to the CallAgent LLM.
4. Convert the LLM response back to speech with Deepgram TTS.

Run with:  python deepgram_llm_loop_test.py
Requires environment variables for Deepgram and Groq to be set, just like the main app.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Optional

from deepgram_services import DeepgramSTT, DeepgramTTS
from llm_agent import CallAgent

CHUNK_SIZE = 320  # 20 ms at 8 kHz mulaw
TEST_UTTERANCE = "Where is United States located"
OUTPUT_DIR = Path("./debug_artifacts")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("dg_harness")


def chunk_audio(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


async def synthesize_text(text: str, tts: DeepgramTTS) -> bytes:
    logger.info("Synthesizing text to speech (%d chars)", len(text))
    audio = await tts.synthesize_mulaw_8k(text)
    logger.info("Synthesized %d bytes of mulaw audio", len(audio))
    return audio


async def stt_transcribe(audio: bytes, stt: DeepgramSTT) -> Optional[str]:
    await stt.connect()
    logger.info("Deepgram STT connected")

    try:
        for chunk in chunk_audio(audio, CHUNK_SIZE):
            await stt.send_audio(chunk)
            await asyncio.sleep(0.02)
        logger.info("Finished streaming audio; waiting for final transcript")

        while True:
            message = await stt.recv()
            if message.get("_closed") or message.get("_error"):
                logger.warning("Deepgram stream closed/error: %s", message)
                break

            channel = message.get("channel") or {}
            if isinstance(channel, list):
                channel = channel[0] if channel else {}
            if not isinstance(channel, dict):
                logger.debug("Skipping non-dict channel payload: %s", channel)
                continue
            alts = channel.get("alternatives") or []
            transcript = (alts[0].get("transcript") if alts else "") or ""
            if not transcript:
                continue

            if message.get("is_final"):
                logger.info("STT FINAL: %s", transcript)
                return transcript
            logger.info("STT interim: %s", transcript)
    finally:
        await stt.close()

    return None


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tts = DeepgramTTS()
    stt = DeepgramSTT()

    source_audio = await synthesize_text(TEST_UTTERANCE, tts)
    (OUTPUT_DIR / "source_prompt.mulaw").write_bytes(source_audio)

    transcript = await stt_transcribe(source_audio, stt)
    if not transcript:
        logger.error("No transcript received; aborting")
        return

    logger.info("Transcript ready for LLM: %s", transcript)
    agent = CallAgent.create("Answer concise geography questions.")
    response_text = await agent.respond(transcript)
    logger.info("LLM response: %s", response_text)

    response_audio = await synthesize_text(response_text, tts)
    (OUTPUT_DIR / "llm_response.mulaw").write_bytes(response_audio)
    (OUTPUT_DIR / "llm_response.b64").write_text(base64.b64encode(response_audio).decode("ascii"))

    logger.info("Artifacts saved under %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    asyncio.run(main())
