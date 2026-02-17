import json
import base64
import asyncio
from typing import Dict, Any, Optional
import contextlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, JSONResponse

from db import init_db, log_call, log_turn
from llm_agent import CallAgent
from deepgram_service import DeepgramSTT, DeepgramTTS
from twilio_service import make_instructions_id, build_twiml_stream, start_outbound_call

app = FastAPI()

# Temporary state:
# 1) instructions_id bucket exists before we know CallSid
# 2) moved to call_sid bucket after Twilio "start" event
CALL_STATE: Dict[str, Dict[str, Any]] = {}

tts = DeepgramTTS()


@app.on_event("startup")
async def startup():
    await init_db()


@app.post("/make-call")
async def make_call(payload: Dict[str, Any]):
    to_number = payload.get("to_number")
    instructions = (payload.get("instructions") or "").strip()

    if not to_number or not instructions:
        return JSONResponse({"error": "to_number and instructions are required"}, status_code=400)

    instructions_id = make_instructions_id()

    CALL_STATE[instructions_id] = {
        "agent": CallAgent.create(instructions),
    }

    call_sid = start_outbound_call(to_number, instructions_id)

    await log_call(call_sid, to_number, "", instructions)

    return {"status": "calling", "call_sid": call_sid, "instructions_id": instructions_id}


@app.post("/twiml")
async def twiml(request: Request):
    instructions_id = request.query_params.get("instructions_id") or ""
    twiml_xml = build_twiml_stream(instructions_id)
    return Response(content=twiml_xml, media_type="application/xml")


async def send_audio_to_twilio(ws: WebSocket, stream_sid: str, audio_bytes: bytes, chunk_size: int = 320):
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i : i + chunk_size]
        payload_b64 = base64.b64encode(chunk).decode("utf-8")
        msg = {"event": "media", "streamSid": stream_sid, "media": {"payload": payload_b64}}
        await ws.send_text(json.dumps(msg))
        await asyncio.sleep(0.02)


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    instructions_id = ws.query_params.get("instructions_id")

    stt = DeepgramSTT()
    await stt.connect()

    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None

    try:
        async def deepgram_loop():
            nonlocal call_sid, stream_sid
            while True:
                data = await stt.recv()
                if data.get("_closed") or data.get("_error"):
                    return

                channel = data.get("channel", {})
                alts = channel.get("alternatives", [])
                transcript = (alts[0].get("transcript") if alts else "") or ""
                is_final = bool(data.get("is_final"))

                if transcript and is_final and call_sid and call_sid in CALL_STATE:
                    agent: CallAgent = CALL_STATE[call_sid]["agent"]

                    await log_turn(call_sid, "user", transcript)
                    response_text = await agent.respond(transcript)
                    await log_turn(call_sid, "assistant", response_text)

                    audio = await tts.synthesize_mulaw_8k(response_text)
                    if stream_sid:
                        await send_audio_to_twilio(ws, stream_sid, audio)

        dg_task = asyncio.create_task(deepgram_loop())

        while True:
            raw = await ws.receive_text()
            event = json.loads(raw)

            etype = event.get("event")

            if etype == "start":
                start = event.get("start", {})
                stream_sid = start.get("streamSid")
                call_sid = start.get("callSid")

                # move state from instructions_id -> call_sid
                if instructions_id and instructions_id in CALL_STATE and call_sid:
                    CALL_STATE[call_sid] = CALL_STATE.pop(instructions_id)

                # greet
                if call_sid and stream_sid:
                    opening = "Hi! How can I help you today?"
                    await log_turn(call_sid, "assistant", opening)
                    audio = await tts.synthesize_mulaw_8k(opening)
                    await send_audio_to_twilio(ws, stream_sid, audio)

            elif etype == "media":
                payload_b64 = (event.get("media") or {}).get("payload")
                if payload_b64:
                    audio_bytes = base64.b64decode(payload_b64)
                    await stt.send_audio(audio_bytes)

            elif etype == "stop":
                break

        dg_task.cancel()
        with contextlib.suppress(Exception):
            await dg_task

    except WebSocketDisconnect:
        pass
    finally:
        await stt.close()
        with contextlib.suppress(Exception):
            await ws.close()
