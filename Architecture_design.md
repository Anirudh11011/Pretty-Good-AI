# System Architecture

## 1. High-Level Overview
- **Goal**: Place an AI-driven outbound call that carries out a user-defined task (e.g., schedule, reschedule, gather information) while sounding natural and respecting strict conversational etiquette.
- **Key Actors**: Streamlit frontend, FastAPI orchestration service, Twilio programmable voice, Deepgram STT/TTS, Groq-hosted LLM, SQLite persistence, and LangSmith telemetry to trace latency and LLM token usage.
- **Core Loop**: User configures call → backend initiates Twilio call → Twilio streams RTP audio over WebSocket → Deepgram performs STT → LangChain agent decides response → Deepgram synthesizes speech → audio streamed back to Twilio in real time.


## 2. Component Breakdown

| Component | Responsibility | Why This Choice |
| --- | --- | --- |
| **[streamlit_app.py](streamlit_app.py)** | Renders scenarios, instructions preview, and collects phone number / task before calling `/make-call`. | Streamlit provides rapid prototyping for internal tooling; scenario templates ensure consistent prompts and reduce hallucinations.
| **FastAPI core ([main.py](main.py))** | Hosts REST/WS endpoints, initializes LangSmith, manages per-call state, glues together Twilio, Deepgram, and the LLM loop. | FastAPI’s async stack is ideal for handling concurrent WebSocket audio streams and IO-heavy AI services.
| **Twilio helper ([twilio_service.py](twilio_service.py))** | Generates instruction IDs, TwiML, and triggers outbound voice calls. | Keeps Twilio-specific logic isolated; making changes to call initiation won’t touch AI logic.
| **Deepgram adapters ([deepgram_services.py](deepgram_services.py))** | Handles streaming STT WebSocket and TTS HTTP calls, always in mulaw/8k. | Using the same vendor/codecs for STT and TTS avoids transcoding latency and audio artifacts.
| **LLM agent ([llm_agent.py](llm_agent.py))** | Wraps Groq `llama-3.3-70b-versatile` inside LangChain `RunnableWithMessageHistory` to preserve conversation context per call. | Runnable-based design keeps memory scoped to each call and makes switching models trivial.
| **Persistence ([db.py](db.py))** | Stores call metadata (`call_logs`) and every conversational turn (`call_turns`). | SQLite is simple, file-based, and good enough for single-host deployments; schema is easy to migrate to Postgres later.


## 3. End-to-End Call Flow

1. **Instruction Capture**
   - Streamlit builds scenario-aware instructions from user input and sends `{to_number, instructions}` to `/make-call`.
   - The backend creates an `instructions_id`, instantiates `CallAgent`, and logs an initial LangSmith run.

2. **Call Bootstrap**
   - Twilio is asked to dial the target and fetch TwiML from `/twiml?instructions_id=...`.
   - The temporary state bucket (keyed by `instructions_id`) is upgraded to the real `CallSid` once Twilio provides it, ensuring instructions aren’t lost before SID assignment.
   - Metadata (numbers, instructions) is persisted via `log_call()` for replay.

3. **Media Streaming**
   - Twilio connects a bi-directional WebSocket to `/media-stream?instructions_id=...` and starts sending mulaw PCM frames.
   - Backend connects to Deepgram STT once per call so that all inbound audio frames can be forwarded immediately.

4. **Turn Handling Loop**
   - Deepgram emits interim and final transcripts. Final transcripts trigger the AI pipeline once deduplicated (`last_user_norm`).
   - The backend logs each user turn to SQLite and LangSmith, then calls `CallAgent.respond()`.
   - Assistant text is stored, LangSmith-logged, and synthesized into mulaw audio via Deepgram TTS.
   - Audio is chunked in 20 ms frames (320 bytes) and streamed back to Twilio; optional silence padding keeps pacing natural after questions.

5. **Run Finalization**
   - When Twilio disconnects or errors occur, the backend closes Deepgram, finalizes LangSmith runs, and leaves a complete transcript in SQLite.

## 4. Key Design Decisions & Rationale

### Per-Call Agent Instances
- `CallAgent.create()` is invoked per call, so each call retains its own `InMemoryChatMessageHistory`. This prevents cross-call leakage and allows concurrent calls without race conditions.

### Dual Deepgram Usage
- Using Deepgram for both directions (STT+TTS) ensures codec compatibility, lowers latency, and simplifies networking to just one vendor (single API key, consistent auth model).

### Streaming WebSocket Architecture
- `/media-stream` keeps a single async loop that reads Deepgram events and writes Twilio media frames. This avoids polling, minimizes buffering, and enables mid-call behaviors (e.g., inserting silence or interrupting audio) without extra infrastructure.

### SQLite Event Logging
- Every turn (`role`, `content`, `timestamp`) is persisted. This makes debugging and QA straightforward, supports replay tools, and creates a foundation for analytics without pulling LangSmith traces.

### LangSmith Integration
- LangSmith is implemented to trace latency and LLM token usage, and to track runs, user transcripts, and assistant responses. When API keys exist, the system automatically performs this tracing. Failures degrade gracefully by simply skipping tracing, so observability never blocks calls.

### Instruction Templates via Streamlit
- Scenario-specific guidelines ensure the assistant receives structured goals, which reduces prompt brittleness and keeps call behavior predictable for each task type.

## 5. Operational Considerations

- **Latency Budget**: TTS synthesis and STT streaming dominate latency. Keeping audio in mulaw/8 kHz and chunking at 20 ms frames maintains real-time responsiveness.
- **Duplicates & Throttling**: The system tracks normalized text (`last_user_norm`, `last_assistant_norm`) to avoid reacting twice to repeated STT finals or sending the same answer multiple times.
- **Error Handling**: Failures in STT, LLM, or TTS are caught, logged, and surfaced through LangSmith plus server logs; assistant falls back to a generic apology when LLM errors.
- **Environment**: `.env` carries keys for Deepgram, Twilio, and Groq; `PUBLIC_BASE_URL` must be internet-accessible so Twilio can reach FastAPI.

This document captures how the current system pieces fit together and the reasoning behind the main architectural decisions. For further questions or modifications, refer to the linked source files above.

## 7. Issues Faced and Solutions

- **Main issue**: During live calls, the assistant often responded before the human finished speaking, leading to partial or fragmented transcripts (for example, repeated phrases like "Pivot Point Orthopedic is…" without the full sentence).
- **Technical root cause**: The Deepgram streaming loop in the FastAPI `/media-stream` endpoint reacted to every STT message marked `is_final=true` as a complete user turn. In practice, Deepgram can emit multiple finals for a single spoken sentence whenever there is a short pause, so the backend treated mid-sentence pauses as full turns and immediately triggered `CallAgent.respond()`, causing the LLM/TTS to talk over the caller and log incomplete text.
- **Mitigations attempted**: I experimented with tuning Deepgram endpointing and VAD behavior (e.g., `vad_events=true`, `vad_sensitivity`, `utterance_end_ms`) and briefly prototyped application-level gating: tracking recent caller speech timestamps, delaying LLM calls until a minimum silence window had elapsed, and injecting extra silence after assistant questions. These changes reduced some barge-ins but also introduced new timing/behavior issues in real calls, so the code was reverted to the simpler, original streaming logic.
- **Current status**: The system still occasionally records partial human utterances in the call log when the caller pauses mid-sentence; a robust fix will likely require a combination of tuned Deepgram endpointing plus an explicit buffering layer that groups multiple final transcripts into a single logical turn before sending them to the LLM.
