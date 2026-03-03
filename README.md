# Pretty Good AI Voice Assistant

AI-powered outbound caller that routes audio between Twilio, Deepgram, and a Groq-hosted LLM so an assistant can join real phone calls in real time.

## 1. Features
- Scenario-driven instruction builder (Streamlit) for consistent call briefs.
- FastAPI backend that dials callers via Twilio and manages a full STT→LLM→TTS loop.
- Deepgram streaming STT/TTS with low-latency mulaw audio compatible with Twilio media streams.
- LangChain `CallAgent` powered by Groq `llama-3.3-70b-versatile`, including message history per call.
- SQLite storage for call metadata and transcripts, plus LangSmith telemetry to trace LLM latency and token usage.

## 2. Prerequisites
- Python 3.10+
- A Twilio account with voice-capable phone number
- Deepgram API key (for STT/TTS)
- Groq API key (for LLM access)
- LangSmith API key for tracing/logging LLM latency and tokens
- A publicly reachable URL (e.g., via ngrok) so Twilio can POST to your FastAPI instance

## 3. Environment Variables
Create a `.env` file in the project root:

```env
# Twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+15551234567

# Deepgram
DEEPGRAM_API_KEY=dg_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Groq / LangChain
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_TEMPERATURE=0.2

# Public URL for Twilio webhooks
PUBLIC_BASE_URL=https://your-ngrok-subdomain.ngrok.app

# LangSmith / LangChain tracing (required for logging latency and tokens)
LANGSMITH_API_KEY=lsm_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=Pretty Good AI
```

## 4. Installation
```bash
python -m venv venv
venv\Scripts\activate   # PowerShell (Windows)
pip install -r requirements.txt
```

## 5. Running the Backend
To run the full application end-to-end:

1. Start ngrok (so Twilio can reach your FastAPI server):
   ```bash
   ngrok http 8000
   ```
2. Copy the HTTPS URL printed by ngrok and update `PUBLIC_BASE_URL` in your `.env` file.
3. Start the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. With the virtual environment active, start the Streamlit UI:
   ```bash
   streamlit run streamlit_app.py
   ```
   Set `BASE_URL` inside `streamlit_app.py` (or via environment variable) to the FastAPI endpoint (e.g., `http://localhost:8000`). Use the UI to craft instructions and click **Make Call**.

## 6. Placing a Call (End-to-End)
1. Ensure FastAPI is running and exposed publicly via ngrok.
2. Open the Streamlit app, pick a scenario, enter the phone number and task instructions.
3. Press **Make Call**. The backend will:
   - Instantiate a call-specific `CallAgent`.
   - Ask Twilio to dial the recipient and stream audio to `/media-stream`.
   - Relay caller speech to Deepgram STT, then LLM, then synthesize the response via Deepgram TTS.
4. Monitor logs in the FastAPI console for transcripts and status.

## 7. Debugging & Testing
- Check `calls.db` (SQLite) for `call_logs` and `call_turns` tables; use any SQLite client to inspect transcripts.
- Use `python generate_call_report.py --limit N --output call_report.html` to export the most recent N call transcripts into an HTML report. Open `call_report.html` in a browser to review full conversations.
- Use the LangSmith dashboard (with your LangSmith keys configured) to trace LLM calls, latency, and token usage.

## 8. Troubleshooting
- **Twilio returns 11200 / webhook failure**: confirm `PUBLIC_BASE_URL` is HTTPS and reachable; FastAPI logs should show incoming `/twiml`/`/media-stream` requests.
- **Audio glitches / high latency**: verify Deepgram API key and internet latency; ensure server has sufficient CPU for real-time chunking.
- **LLM errors**: check Groq quota and keys; backend falls back to an apology message if LLM fails.
 
## 9. Notes
- For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).
- All call logs can be extracted using the `generate_call_report.py` script, which writes an HTML report to `call_report.html` (or a path you specify with `--output`).
- `important_calls.html` is the selected important call transcript that can be viewed from a browser.
