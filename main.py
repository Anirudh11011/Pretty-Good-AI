import json
import base64
import asyncio
import time
import logging
from typing import Dict, Any, Optional
import contextlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, JSONResponse

from db import init_db, log_call, log_turn
from llm_agent import CallAgent

# IMPORTANT: make sure your import matches your filename.
# If your file is deepgram_services.py, use:
from deepgram_services import DeepgramSTT, DeepgramTTS

# IMPORTANT: make sure your import matches your filename.
# If your file is twilio_service.py, use:
from twilio_service import make_instructions_id, build_twiml_stream, start_outbound_call

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("voicebot")

app = FastAPI()

# Temporary state:
# 1) instructions_id bucket exists before we know CallSid
# 2) moved to call_sid bucket after Twilio "start" event
CALL_STATE: Dict[str, Dict[str, Any]] = {}

tts = DeepgramTTS()


@app.on_event("startup")
async def startup():
    logger.info("Starting up: init_db()")
    await init_db()
    logger.info("Startup complete")


@app.post("/make-call")
async def make_call(payload: Dict[str, Any]):
    to_number = payload.get("to_number")
    instructions = (payload.get("instructions") or "").strip()

    if not to_number or not instructions:
        return JSONResponse({"error": "to_number and instructions are required"}, status_code=400)

    instructions_id = make_instructions_id()

    CALL_STATE[instructions_id] = {"agent": CallAgent.create(instructions)}
    logger.info(f"/make-call: to_number={to_number} instructions_id={instructions_id}")

    try:
        call_sid = start_outbound_call(to_number, instructions_id)
        CALL_STATE[call_sid] = CALL_STATE.pop(instructions_id)
    except Exception:
        logger.exception("Twilio start_outbound_call failed")
        raise

    await log_call(call_sid, to_number, "", instructions)
    logger.info(f"/make-call: call started call_sid={call_sid} instructions_id={instructions_id}")

    return {"status": "calling", "call_sid": call_sid, "instructions_id": instructions_id}


@app.post("/twiml")
async def twiml(request: Request):
    instructions_id = request.query_params.get("instructions_id") or ""
    logger.info(f"/twiml requested: instructions_id={instructions_id}")
    try:
        twiml_xml = build_twiml_stream(instructions_id)
        logger.info(f"/twiml returning TwiML OK: instructions_id={instructions_id} bytes={len(twiml_xml)}")
        return Response(content=twiml_xml, media_type="application/xml")
    except Exception:
        logger.exception(f"/twiml failed: instructions_id={instructions_id}")
        raise


async def send_audio_to_twilio(ws: WebSocket, stream_sid: str, audio_bytes: bytes, chunk_size: int = 320):
    # Twilio expects 20ms frames for 8k mulaw: 160 samples ~= 20ms; your chunk_size=320 is OK in practice.
    total = len(audio_bytes)
    logger.info(f"send_audio_to_twilio: streamSid={stream_sid} bytes={total} chunk_size={chunk_size}")

    sent = 0
    for i in range(0, total, chunk_size):
        chunk = audio_bytes[i : i + chunk_size]
        payload_b64 = base64.b64encode(chunk).decode("utf-8")
        msg = {"event": "media", "streamSid": stream_sid, "media": {"payload": payload_b64}}
        await ws.send_text(json.dumps(msg))
        sent += len(chunk)
        await asyncio.sleep(0.02)

    logger.info(f"send_audio_to_twilio: streamSid={stream_sid} done sent_bytes={sent}")


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()

    instructions_id = ws.query_params.get("instructions_id")
    logger.info(f"WS /media-stream connected. instructions_id={instructions_id}")

    stt = DeepgramSTT()
    try:
        await stt.connect()
        logger.info("DeepgramSTT connected")
    except Exception:
        logger.exception("DeepgramSTT connect failed")
        await ws.close()
        return

    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None

    # For rate-limited logging of media packets
    media_packets = 0
    last_media_log = time.time()

    try:
        async def deepgram_loop():
            nonlocal call_sid, stream_sid
            logger.info("Deepgram loop started")

            while True:
                data = await stt.recv()
                logger.info(f"Deepgram raw keys={list(data.keys())} data={data if len(str(data))<300 else '<<large>>'}")

                if data.get("_closed"):
                    logger.warning("Deepgram ws closed")
                    return
                if data.get("_error"):
                    logger.warning("Deepgram ws error")
                    return

                # Pull transcript fields
                channel = data.get("channel") or {}
                if isinstance(channel, list):
                    channel = channel[0] if channel else {}
                if not isinstance(channel, dict):
                    logger.debug(
                        "Skipping Deepgram channel payload of type %s: %s",
                        type(channel).__name__,
                        channel,
                    )
                    continue

                alts = channel.get("alternatives", [])
                transcript = (alts[0].get("transcript") if alts else "") or ""
                is_final = bool(data.get("is_final"))

                # Log interim/final transcripts so you can SEE if STT works
                if transcript:
                    if transcript:
                        logger.info(f"STT {'FINAL' if is_final else 'interim'}: call_sid={call_sid} text={transcript!r}")

                    else:
                        logger.debug(f"STT interim: call_sid={call_sid} text={transcript!r}")

                # Your logic: only respond when final + known call_sid bucket exists
                if transcript and is_final and call_sid and call_sid in CALL_STATE:
                    agent: CallAgent = CALL_STATE[call_sid]["agent"]

                    await log_turn(call_sid, "user", transcript)

                    # LLM timing
                    t0 = time.time()
                    logger.info(f"LLM invoke: call_sid={call_sid} user_text_len={len(transcript)}")
                    try:
                        response_text = await agent.respond(transcript)
                    except Exception:
                        logger.exception("LLM respond() failed")
                        response_text = "Sorry—there was an internal error."
                    dt = time.time() - t0
                    logger.info(
                        f"LLM done: call_sid={call_sid} latency_s={dt:.2f} assistant_len={len(response_text)}"
                    )

                    await log_turn(call_sid, "assistant", response_text)

                    # TTS timing
                    t1 = time.time()
                    try:
                        audio = await tts.synthesize_mulaw_8k(response_text)
                        logger.info(
                            f"TTS done: call_sid={call_sid} latency_s={time.time()-t1:.2f} audio_bytes={len(audio)}"
                        )
                    except Exception:
                        logger.exception("Deepgram TTS failed")
                        audio = b""

                    if audio and stream_sid:
                        await send_audio_to_twilio(ws, stream_sid, audio)
                    else:
                        logger.warning(
                            f"Skipped sending audio. stream_sid={stream_sid} audio_bytes={len(audio)}"
                        )

        dg_task = asyncio.create_task(deepgram_loop())

        while True:
            raw = await ws.receive_text()
            event = json.loads(raw)
            etype = event.get("event")

            if etype == "start":
                start = event.get("start", {})
                stream_sid = start.get("streamSid")
                call_sid = start.get("callSid")

                logger.info(f"Twilio START: call_sid={call_sid} stream_sid={stream_sid} instructions_id={instructions_id}")

                # ensure state bucket exists for this call
                if call_sid and call_sid not in CALL_STATE and instructions_id and instructions_id in CALL_STATE:
                    CALL_STATE[call_sid] = CALL_STATE.pop(instructions_id)
                    logger.info(
                        f"CALL_STATE moved: instructions_id -> call_sid ({instructions_id} -> {call_sid})"
                    )
                elif call_sid and call_sid in CALL_STATE:
                    logger.debug(f"CALL_STATE already attached for call_sid={call_sid}")
                else:
                    logger.warning(
                        f"CALL_STATE missing agent for call_sid={call_sid} instructions_id={instructions_id}"
                    )

                # greet
                if call_sid and stream_sid:
                    opening = "Hi! I wanted to schedule an appointment"
                    await log_turn(call_sid, "assistant", opening)

                    try:
                        audio = await tts.synthesize_mulaw_8k(opening)
                        logger.info(f"Greeting TTS bytes={len(audio)}")
                        await send_audio_to_twilio(ws, stream_sid, audio)
                    except Exception:
                        logger.exception("Greeting TTS/send failed")

            elif etype == "media":
                payload_b64 = (event.get("media") or {}).get("payload")
                if payload_b64:
                    audio_bytes = base64.b64decode(payload_b64)
                    await stt.send_audio(audio_bytes)

                    # Rate-limited media logging so it doesn't spam
                    media_packets += 1
                    now = time.time()
                    if now - last_media_log >= 2.0:
                        logger.info(
                            f"Twilio MEDIA: call_sid={call_sid} stream_sid={stream_sid} "
                            f"packets_last_2s={media_packets}"
                        )
                        media_packets = 0
                        last_media_log = now

            elif etype == "stop":
                logger.info(f"Twilio STOP: call_sid={call_sid} stream_sid={stream_sid}")
                break

            else:
                logger.debug(f"Twilio EVENT: {etype} keys={list(event.keys())}")

        dg_task.cancel()
        with contextlib.suppress(Exception):
            await dg_task

    except WebSocketDisconnect:
        logger.warning(f"WebSocketDisconnect: call_sid={call_sid} stream_sid={stream_sid}")
    except Exception:
        logger.exception(f"Unhandled error in /media-stream: call_sid={call_sid} stream_sid={stream_sid}")
    finally:
        await stt.close()
        with contextlib.suppress(Exception):
            await ws.close()
        logger.info(f"WS /media-stream closed. call_sid={call_sid} stream_sid={stream_sid}")
