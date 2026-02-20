# streamlit_app.py
import json
import requests
import streamlit as st

st.set_page_config(page_title="AI Voice Assistant Caller", page_icon="📞", layout="centered")

st.title("📞 Real-time AI Voice Assistant (Outbound Caller)")

# ---- Sidebar / config ----
st.sidebar.header("Backend")
default_base_url = "http://localhost:8000"
base_url = st.sidebar.text_input("FastAPI Base URL", value=default_base_url).strip().rstrip("/")

st.sidebar.caption("Example: http://localhost:8000 (local)")

# ---- Main inputs ----
to_number = st.text_input("Phone number to call", placeholder="+14155551234")
instructions = st.text_area(
    "Instructions for the assistant",
    placeholder="Example: Call and book a dentist appointment for next Tuesday afternoon.",
    height=180,
)

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
    if not to_number.strip() or not instructions.strip():
        st.warning("Please enter both a phone number and instructions.")
        st.stop()

    endpoint = f"{base_url}/make-call"
    payload = {"to_number": to_number.strip(), "instructions": instructions.strip()}

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
