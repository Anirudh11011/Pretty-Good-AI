import json
import base64
import asyncio
import time
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import contextlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, JSONResponse

from dotenv import load_dotenv
from db import init_db, log_call, log_turn
from llm_agent import CallAgent
from langsmith import Client

load_dotenv()

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

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "Pretty Good AI"


def _init_langsmith_client() -> Optional[Client]:
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.info("LangSmith logging disabled (no API key provided)")
        return None

    # Ensure LangChain SDK knows which endpoint/project to use when LangSmith is active.
    if not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)

    try:
        client = Client()
        logger.info("LangSmith logging enabled for project '%s'", LANGSMITH_PROJECT)
        return client
    except Exception:
        logger.exception("Failed to initialize LangSmith client")
        return None


LANGSMITH_CLIENT = _init_langsmith_client()

# Temporary state:
# 1) instructions_id bucket exists before we know CallSid
# 2) moved to call_sid bucket after Twilio "start" event
CALL_STATE: Dict[str, Dict[str, Any]] = {}


def create_langsmith_call_run(to_number: str, instructions: str, instructions_id: str) -> Optional[str]:
    if not LANGSMITH_CLIENT:
        return None
    try:
        run = LANGSMITH_CLIENT.create_run(
            name="outbound_call",
            inputs={"to_number": to_number, "instructions": instructions},
            run_type="chain",
            tags=["twilio", "deepgram", "call"],
            metadata={"instructions_id": instructions_id},
        )
        return str(run.id)
    except Exception:
        logger.warning("LangSmith call run creation failed", exc_info=True)
        return None


def patch_langsmith_run(run_id: Optional[str], **kwargs) -> None:
    if not (LANGSMITH_CLIENT and run_id):
        return
    try:
        LANGSMITH_CLIENT.patch_run(run_id=run_id, **kwargs)
    except Exception:
        logger.warning("LangSmith patch_run failed", exc_info=True)


def log_langsmith_user_transcript(run_id: Optional[str], transcript: str) -> None:
    if not (LANGSMITH_CLIENT and run_id and transcript):
        return
    try:
        LANGSMITH_CLIENT.create_run(
            name="caller_transcript",
            inputs={"source": "twilio"},
            outputs={"text": transcript},
            run_type="tool",
            parent_run_id=run_id,
            tags=["user", "stt"],
        )
    except Exception:
        logger.warning("LangSmith user transcript log failed", exc_info=True)


def log_langsmith_assistant_response(
    run_id: Optional[str],
    response_text: str,
    usage: Optional[Dict[str, Any]],
    audio_bytes: int,
    error: Optional[str] = None,
) -> None:
    if not (LANGSMITH_CLIENT and run_id and response_text):
        return

    outputs: Dict[str, Any] = {"text": response_text}
    metadata: Dict[str, Any] = {"audio_bytes": audio_bytes}
    if usage:
        outputs["usage"] = usage
        metadata.update({
            "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
            "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens") or usage.get("token_count"),
        })

    # Remove None entries to keep metadata clean.
    metadata = {k: v for k, v in metadata.items() if v is not None}

    try:
        LANGSMITH_CLIENT.create_run(
            name="assistant_response",
            inputs={"generator": "CallAgent"},
            outputs=outputs,
            run_type="llm",
            parent_run_id=run_id,
            tags=["assistant", "tts"],
            metadata=metadata or None,
            error=error,
        )
    except Exception:
        logger.warning("LangSmith assistant response log failed", exc_info=True)


def finalize_langsmith_run(run_id: Optional[str], status: str, error: Optional[str] = None) -> None:
    if not (LANGSMITH_CLIENT and run_id):
        return
    try:
        LANGSMITH_CLIENT.patch_run(
            run_id=run_id,
            outputs={"status": status},
            error=error,
            end_time=datetime.utcnow(),
        )
    except Exception:
        logger.warning("LangSmith finalize failed", exc_info=True)


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

    agent = CallAgent.create(instructions)
    langsmith_run_id = create_langsmith_call_run(to_number, instructions, instructions_id)

    CALL_STATE[instructions_id] = {
        "agent": agent,
        "langsmith_run_id": langsmith_run_id,
        "instructions": instructions,
        "to_number": to_number,
    }
    logger.info(f"/make-call: to_number={to_number} instructions_id={instructions_id}")

    try:
        call_sid = start_outbound_call(to_number, instructions_id)
        call_state = CALL_STATE.pop(instructions_id)
        CALL_STATE[call_sid] = call_state
        patch_langsmith_run(
            call_state.get("langsmith_run_id"),
            metadata={"call_sid": call_sid, "to_number": to_number},
            outputs={"status": "dialing"},
        )
    except Exception:
        logger.exception("Twilio start_outbound_call failed")
        pending_state = CALL_STATE.pop(instructions_id, None)
        run_id = (pending_state or {}).get("langsmith_run_id") or langsmith_run_id
        finalize_langsmith_run(run_id, status="failed", error="Twilio start failed")
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

            try:
                while True:
                    data = await stt.recv()
                    logger.info(
                        f"Deepgram raw keys={list(data.keys())} data={data if len(str(data))<300 else '<<large>>'}"
                    )

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
                            logger.info(
                                f"STT {'FINAL' if is_final else 'interim'}: call_sid={call_sid} text={transcript!r}"
                            )

                        else:
                            logger.debug(f"STT interim: call_sid={call_sid} text={transcript!r}")

                    # Your logic: only respond when final + known call_sid bucket exists
                    if transcript and is_final and call_sid and call_sid in CALL_STATE:
                        call_state = CALL_STATE[call_sid]
                        agent: CallAgent = call_state["agent"]
                        run_id = call_state.get("langsmith_run_id")

                        await log_turn(call_sid, "user", transcript)
                        log_langsmith_user_transcript(run_id, transcript)

                        # LLM timing
                        t0 = time.time()
                        logger.info(f"LLM invoke: call_sid={call_sid} user_text_len={len(transcript)}")
                        usage: Dict[str, Any] = {}
                        llm_error: Optional[str] = None
                        try:
                            response_text = await agent.respond(transcript)
                            usage = agent.last_usage
                        except Exception as exc:
                            logger.exception("LLM respond() failed")
                            response_text = "Sorry—there was an internal error."
                            llm_error = str(exc)
                            usage = {}
                        dt = time.time() - t0
                        logger.info(
                            f"LLM done: call_sid={call_sid} latency_s={dt:.2f} assistant_len={len(response_text)}"
                        )

                        await log_turn(call_sid, "assistant", response_text)

                        # TTS timing
                        t1 = time.time()
                        tts_error: Optional[str] = None
                        try:
                            audio = await tts.synthesize_mulaw_8k(response_text)
                            logger.info(
                                f"TTS done: call_sid={call_sid} latency_s={time.time()-t1:.2f} audio_bytes={len(audio)}"
                            )
                        except Exception as exc:
                            logger.exception("Deepgram TTS failed")
                            tts_error = str(exc)
                            audio = b""

                        combined_error = None
                        if llm_error or tts_error:
                            combined_error = " | ".join(filter(None, [llm_error, tts_error]))
                        log_langsmith_assistant_response(
                            run_id, response_text, usage, len(audio), error=combined_error
                        )

                        if audio and stream_sid:
                            await send_audio_to_twilio(ws, stream_sid, audio)
                        else:
                            logger.warning(
                                f"Skipped sending audio. stream_sid={stream_sid} audio_bytes={len(audio)}"
                            )
            except asyncio.CancelledError:
                logger.info("Deepgram loop cancelled")
            except Exception:
                logger.exception("Deepgram loop failed")
                raise

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
                call_state = CALL_STATE.pop(call_sid, None)
                finalize_langsmith_run((call_state or {}).get("langsmith_run_id"), status="completed")
                break

            else:
                logger.debug(f"Twilio EVENT: {etype} keys={list(event.keys())}")

        dg_task.cancel()
        with contextlib.suppress(Exception):
            await dg_task

    except WebSocketDisconnect:
        logger.warning(f"WebSocketDisconnect: call_sid={call_sid} stream_sid={stream_sid}")
        call_state = CALL_STATE.pop(call_sid, None) or CALL_STATE.pop(instructions_id or "", None)
        finalize_langsmith_run((call_state or {}).get("langsmith_run_id"), status="disconnected")
    except Exception as exc:
        logger.exception(f"Unhandled error in /media-stream: call_sid={call_sid} stream_sid={stream_sid}")
        call_state = CALL_STATE.pop(call_sid, None) or CALL_STATE.pop(instructions_id or "", None)
        finalize_langsmith_run((call_state or {}).get("langsmith_run_id"), status="error", error=str(exc))
    finally:
        await stt.close()
        with contextlib.suppress(Exception):
            await ws.close()
        logger.info(f"WS /media-stream closed. call_sid={call_sid} stream_sid={stream_sid}")
