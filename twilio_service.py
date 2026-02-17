import os
import uuid
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def make_instructions_id() -> str:
    return uuid.uuid4().hex


def build_twiml_stream(instructions_id: str) -> str:
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL not set")

    ws_url = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/media-stream?instructions_id={instructions_id}"

    vr = VoiceResponse()
    connect = Connect()
    connect.append(Stream(url=ws_url))
    vr.append(connect)
    return str(vr)


def start_outbound_call(to_number: str, instructions_id: str) -> str:
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL not set")

    twiml_url = f"{PUBLIC_BASE_URL}/twiml?instructions_id={instructions_id}"

    call = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        url=twiml_url,
        method="POST",
    )
    return call.sid
