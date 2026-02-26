# streamlit_app.py
import json
from textwrap import dedent

import requests
import streamlit as st


def build_instruction_text(
    task: str,
    appointment_type: str,
    doctor_preference: str,
    time_preference: str,
    extra_notes: str,
) -> str:
    task = task.strip()
    if not task:
        return ""

    appt = (appointment_type.strip() or "a general appointment").lower()
    doctor = doctor_preference.strip() or "no specific doctor preference"
    time_pref = time_preference.strip() or "no specific time preference; accept the earliest opening"
    notes = extra_notes.strip() or "None provided"

    instructions = f"""
    You are Anirudh's personal phone assistant calling on their behalf.
    When the call connects, stay silent and listen to the other person first. Once they finish their greeting, respond warmly and naturally.
    Use varied, natural phrasing; do not repeat the same question or statement unless the person explicitly asks you to.
    After asking a question, leave about one second of silence so they have room to answer.

    Primary task:
    - {task}

    Appointment expectations:
    - Appointment type to request: {appt}
    - Preferred time or window: {time_pref}
    - Doctor preference: {doctor}

    If the office asks for details you don't have, explain there is no preference and book a general appointment with any available doctor and time. Accept the earliest option that works unless the task specifies otherwise.
    Confirm the agreed appointment details aloud before ending the call.

    Extra context for you: {notes}
    """

    return dedent(instructions).strip()


st.set_page_config(page_title="AI Voice Assistant Caller", page_icon="📞", layout="centered")

st.title("📞 Real-time AI Voice Assistant (Outbound Caller)")

# ---- Sidebar / config ----
st.sidebar.header("Backend")
default_base_url = "http://localhost:8000"
base_url = st.sidebar.text_input("FastAPI Base URL", value=default_base_url).strip().rstrip("/")

st.sidebar.caption("Example: http://localhost:8000 (local)")

# ---- Main inputs ----
to_number = st.text_input("Phone number to call", placeholder="+14155551234")

st.subheader("What should your assistant do?")
task_request = st.text_area(
    "Describe the task",
    placeholder="Example: Book a follow-up appointment for me around 9pm tonight.",
    height=140,
)

col_a, col_b = st.columns(2)
with col_a:
    appointment_type = st.text_input("Appointment type", placeholder="General check-up")
with col_b:
    time_preference = st.text_input("Preferred time", placeholder="9pm or 'Earliest available'")

doctor_preference = st.text_input("Doctor preference", placeholder="Any doctor is fine")
extra_notes = st.text_area("Extra notes (optional)", placeholder="Add insurance info, reference numbers, etc.", height=100)

instruction_preview = build_instruction_text(
    task_request or "",
    appointment_type or "",
    doctor_preference or "",
    time_preference or "",
    extra_notes or "",
)

if instruction_preview:
    with st.expander("LLM instruction preview", expanded=False):
        st.code(instruction_preview, language="markdown")

col1, col2 = st.columns([1, 1])
with col1:
    call_btn = st.button("📲 Make Call", type="primary", use_container_width=True)
with col2:
    st.button("Clear", use_container_width=True, on_click=lambda: st.session_state.clear())

# ---- Action ----
if call_btn:
    if not base_url:
        st.error("Please set the FastAPI Base URL in the sidebar.")
        st.stop()
    if not to_number.strip():
        st.warning("Please enter the phone number you want to call.")
        st.stop()
    if not instruction_preview:
        st.warning("Please describe the task so the assistant knows what to do.")
        st.stop()

    endpoint = f"{base_url}/make-call"
    payload = {"to_number": to_number.strip(), "instructions": instruction_preview}

    try:
        with st.spinner("Requesting outbound call..."):
            resp = requests.post(endpoint, json=payload, timeout=20)

        # If backend returns non-200, show details
        if not resp.ok:
            st.error(f"Backend error ({resp.status_code})")
            # try to show JSON if possible
            try:
                st.code(resp.json(), language="json")
            except Exception:
                st.code(resp.text)
            st.stop()

        data = resp.json()
        st.success("Call initiated!")
        st.write("Response:")
        st.code(json.dumps(data, indent=2), language="json")

        st.info(
            "If you’re running locally, make sure Twilio can reach your FastAPI server "
            "(e.g., via ngrok) and that your PUBLIC_BASE_URL matches."
        )

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend. Is FastAPI running?")
    except requests.exceptions.Timeout:
        st.error("Backend request timed out.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
