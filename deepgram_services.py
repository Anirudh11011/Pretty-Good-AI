import os
import json
import logging
import aiohttp

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

logger = logging.getLogger("deepgram")
logger.info("Deepgram key loaded: %s", bool(DEEPGRAM_API_KEY))

DEEPGRAM_LISTEN_WSS = (
    "wss://api.deepgram.com/v1/listen?"
    "encoding=mulaw&sample_rate=8000&channels=1&punctuate=true&interim_results=true&vad_events=true"
)

DEEPGRAM_SPEAK_URL = (
    "https://api.deepgram.com/v1/speak?"
    "model=aura-asteria-en&encoding=mulaw&sample_rate=8000"
)


class DeepgramSTT:
    """
    Streaming STT via Deepgram WebSocket.
    """
    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None

    async def connect(self):
        self._session = aiohttp.ClientSession()
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        self._ws = await self._session.ws_connect(DEEPGRAM_LISTEN_WSS, headers=headers)

    async def send_audio(self, audio_bytes: bytes):
        if not self._ws:
            raise RuntimeError("DeepgramSTT not connected")
        await self._ws.send_bytes(audio_bytes)

    async def recv(self) -> dict:
        """
        Returns parsed JSON messages from Deepgram.
        """
        if not self._ws:
            raise RuntimeError("DeepgramSTT not connected")
        msg = await self._ws.receive()
        if msg.type == aiohttp.WSMsgType.TEXT:
            logger.info("Deepgram TEXT message bytes=%d", len(msg.data))
            return json.loads(msg.data)
        if msg.type == aiohttp.WSMsgType.BINARY:
            logger.info("Deepgram BINARY message bytes=%d", len(msg.data))
            return {"_binary": True}
        if msg.type == aiohttp.WSMsgType.CLOSED:
            logger.warning("Deepgram CLOSED event received")
            return {"_closed": True}
        if msg.type == aiohttp.WSMsgType.ERROR:
            logger.error("Deepgram ERROR event: %s", msg.data)
            return {"_error": True}
        return {}

    async def close(self):
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()


class DeepgramTTS:
    """
    TTS via Deepgram HTTP (returns mulaw 8k bytes).
    """
    async def synthesize_mulaw_8k(self, text: str) -> bytes:
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        async with aiohttp.ClientSession() as session:
            async with session.post(DEEPGRAM_SPEAK_URL, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise RuntimeError(f"Deepgram TTS failed: {resp.status} {detail}")
                return await resp.read()
