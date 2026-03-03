# streamlit_app.py
import json
from textwrap import dedent

import requests
import streamlit as st


SCENARIOS = {
    "schedule": {
        "name": "Schedule Appointment",
        "tagline": "Book a new appointment and lock in the best slot for Anirudh.",
        "placeholder": "Call Dr. Lee's office and schedule a follow-up next week in the afternoon. Earliest slot works.",
        "helper_bullets": [
            "Mention appointment type and reason for the visit.",
            "Share time windows or say 'earliest available' if flexible.",
            "Call out doctor preference or say any provider works.",

        ],
        "instruction_lines": [
            "Confirm the office understood the requested appointment type.",
            "Offer multiple time windows if the first one is unavailable.",
            "Accept the earliest option that fits when no preference is given.",
            "Repeat the confirmed date, time, and provider before ending.",
        ],
        "examples": [
            {
                "title": "Standard booking",
                "text": "Call Midtown Spine Clinic and book a physical therapy follow-up for Anirudh next Tuesday or Thursday after 3pm. Dr. Russell if possible.",
            },
            {
                "title": "Earliest slot",
                "text": "Reach out to Valley Pediatrics and grab the earliest available well visit for Anirudh's nephew. Any doctor is fine and mornings are easier.",
            },
        ],
    },
    "reschedule": {
        "name": "Reschedule or Cancel",
        "tagline": "Move or cancel an existing appointment without losing key details.",
        "placeholder": "Call Dr. Patel's office and move my cleaning on March 2 at 9am to anything next week after 1pm. Cancel if nothing is open.",
        "helper_bullets": [
            "Provide the original appointment date, time, and doctor.",
            "State clearly if you want to reschedule or cancel outright.",
            "Share new availability windows or acceptable alternatives.",
        ],
        "instruction_lines": [
            "Give the receptionist the original appointment info immediately.",
            "If rescheduling, confirm the replacement slot aloud before ending.",
            "If canceling, ask for confirmation numbers or policy details.",
            "Document any follow-up actions, payments, or prep work required.",
        ],
        "examples": [
            {
                "title": "Move appointment",
                "text": "Call Dr. Patel's office and move my cleaning currently on March 2 at 9am to anything next week after 1pm. If they cannot, cancel and note any fees.",
            },
            {
                "title": "Cancel outright",
                "text": "Cancel the dermatology follow-up on April 18 at 11am with Dr. Wong and confirm whether there is a cancellation fee.",
            },
        ],
    },
    "med_refill": {
        "name": "Medication Refill",
        "tagline": "Request a prescription refill or renewal with the pharmacy details ready.",
        "placeholder": "Call CVS on Mission St and refill metoprolol 25mg tablets, 90 count, with pickup later this week.",
        "helper_bullets": [
            "Include medication name, dosage, and quantity needed.",
            "Provide the pharmacy name, location, or delivery preference.",
            "Mention allergies, urgency, or last appointment if relevant.",
            "Note whether provider approval is already on file.",
        ],
        "instruction_lines": [
            "Spell the medication name and dosage clearly.",
            "If location is asked mention Texas as Location and CVS as Pharmacy name.",
        ],
        "examples": [
            {
                "title": "Pharmacy pickup",
                "text": "Call CVS on Mission Street and request a refill for metoprolol succinate 25mg ER tablets, 90 count. Pickup is fine any afternoon this week.",
            },
            {
                "title": "Mail delivery",
                "text": "Contact Kaiser pharmacy and renew my albuterol inhaler prescription. Ask if they can mail it to me and confirm the shipping timeline.",
            },
        ],
    },
    "office_info": {
        "name": "Office Info & Insurance",
        "tagline": "Ask about hours, locations, insurance acceptance, or other logistics.",
        "placeholder": "Call Lakeside Family Medicine to confirm Saturday hours, parking tips, and whether they take Aetna POS.",
        "helper_bullets": [
            "List the specific questions: hours, address, parking, insurance, paperwork.",
            "Add any follow-up info you want so the assistant can ask in one call.",
            "Mention dates or deadlines if the info is time-sensitive.",
        ],
        "instruction_lines": [
            "Ask all requested questions so the call stays efficient.",
            "Request clarification if staff answers ambiguously.",
            "Capture any special instructions like parking codes or ID requirements.",
            "Offer to repeat key details back to ensure accuracy.",
        ],
        "examples": [
            {
                "title": "Hours and insurance",
                "text": "Call Lakeside Family Medicine to ask about Saturday hours, parking directions, and whether they accept Aetna POS. Note any new patient paperwork.",
            },
            {
                "title": "Multiple offices",
                "text": "Reach out to Sunrise Dental and confirm which location has early morning hours and whether MetLife PPO is in-network.",
            },
        ],
    },
    "custom": {
        "name": "Custom / Other",
        "tagline": "Describe any other phone task you want the assistant to handle.",
        "placeholder": "Call the imaging center about my pending MRI results and ask if a doctor needs to sign off before release.",
        "helper_bullets": [
            "Describe the goal in natural language.",
            "List constraints, names, numbers, or references that must be mentioned.",
            "Share what success looks like so the assistant knows when the call is done.",
        ],
        "instruction_lines": [
            "Listen carefully and adapt to what the other person says.",
            "Ask polite clarifying questions if the request is underspecified.",
            "Summarize agreed next steps before hanging up.",
        ],
        "examples": [
            {
                "title": "General purpose",
                "text": "Can You tell what are the facilities available in the hospital and list the names of doctors available for consultation.",
            }
        ],
    },
}

DEFAULT_SCENARIO_KEY = "schedule"
SCENARIO_KEYS = list(SCENARIOS.keys())
DEFAULT_SCENARIO_INDEX = SCENARIO_KEYS.index(DEFAULT_SCENARIO_KEY)
DEFAULT_BASE_URL = "http://localhost:8000"
BASE_URL = DEFAULT_BASE_URL.rstrip("/")


def build_instruction_text(task: str, scenario_key: str) -> str:
    task = task.strip()
    if not task:
        return ""

    scenario = SCENARIOS.get(scenario_key) or SCENARIOS[DEFAULT_SCENARIO_KEY]
    guidelines = "\n    ".join(f"- {line}" for line in scenario["instruction_lines"])

    instructions = f"""
    You are Anirudh's personal phone assistant calling on their behalf.
    When the call connects, stay silent and listen to the other person first. Once they finish their greeting, respond warmly and naturally.
    Use varied, natural phrasing; do not repeat the same question or statement unless the person explicitly asks you to.
    After asking a question, leave about one second of silence so they have room to answer.

    Scenario focus:
    - {scenario['name']}
    - Mission reminder: {scenario['tagline']}

    Primary task:
    - {task}

    Scenario-specific reminders:
    {guidelines}

    Wrap-up checklist:
    - Confirm the outcome or next steps before ending the call.
    - Summarize agreed details aloud for Anirudh.
    """

    return dedent(instructions).strip()


st.set_page_config(page_title="AI Voice Assistant Caller", page_icon="📞", layout="centered")

# ---- Modern CSS (purely UI) ----
st.markdown(
    """
<style>
/* Page max width */
.block-container {
    max-width: 980px;
    padding-top: 2.0rem;
    padding-bottom: 2.0rem;
}

/* Typography */
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.95rem; }
.small-muted-light { color: rgba(0,0,0,0.55); font-size: 0.95rem; }

/* Card */
.card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px 18px 16px 18px;
    background: rgba(255,255,255,0.02);
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset;
    margin-bottom: 14px;
}
.card.light {
    border: 1px solid rgba(0,0,0,0.08);
    background: rgba(0,0,0,0.02);
    box-shadow: 0 1px 0 rgba(0,0,0,0.03) inset;
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.85rem;
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.85);
    margin-right: 8px;
}

/* Helper list styling */
ul.clean { margin: 0.2rem 0 0.2rem 1.1rem; }
ul.clean li { margin: 0.25rem 0; }

/* CTA row spacing */
.cta-wrap {
    margin-top: 6px;
    padding-top: 12px;
}

/* Streamlit input alignment tweaks */
.stTextArea textarea {
    border-radius: 12px !important;
}
.stTextInput input {
    border-radius: 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown(
    """
<div class="card">
\
  <h1 style="margin: 10px 0 6px 0;">AI Voice Assistant Caller</h1>
 
</div>
""",
    unsafe_allow_html=True,
)



# ---- Main layout ----
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown(
        """
<div class="card">
  <h3 style="margin: 0 0 10px 0;">Call details</h3>
  <div class="small-muted">Enter the number you want the assistant to call.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    to_number = st.text_input("Phone number", placeholder="+14155551234")

    st.markdown(
        """
<div class="card">
  <h3 style="margin: 0 0 10px 0;">Call scenario</h3>
  <div class="small-muted">Pick the options from below so the assistant asks the right questions.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    scenario_key = st.radio(
        "Call scenario",
        options=SCENARIO_KEYS,
        index=DEFAULT_SCENARIO_INDEX,
        format_func=lambda key: SCENARIOS[key]["name"],
        horizontal=True,
    )
    scenario_meta = SCENARIOS.get(scenario_key, SCENARIOS[DEFAULT_SCENARIO_KEY])
    st.caption(scenario_meta["tagline"])

    st.markdown(
        f"""
<div class="card">
  <h3 style="margin: 0 0 10px 0;">What should the assistant do?</h3>
  <div class="small-muted">{scenario_meta['tagline']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    task_request = st.text_area(
        "Your request",
        placeholder=scenario_meta["placeholder"],
        height=170,
    )

with right:
    helper_items = scenario_meta.get("helper_bullets", [])
    helper_html = "".join(f"<li>{item}</li>" for item in helper_items) or "<li>Describe the call goal in natural language.</li>"
    st.markdown(
        f"""
<div class="card">
  <h3 style="margin: 0 0 10px 0;">{scenario_meta['name']} tips</h3>
  <div class="small-muted">{scenario_meta['tagline']}</div>
  <ul class="clean">{helper_html}</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    examples = scenario_meta.get("examples", [])
    if examples:
        with st.expander("Example scripts", expanded=False):
            for example in examples:
                title = example.get("title")
                if title:
                    st.markdown(f"**{title}**")
                st.code(example.get("text", ""), language="text")

scenario_meta = SCENARIOS.get(scenario_key, SCENARIOS[DEFAULT_SCENARIO_KEY])

# Scenario-aware instruction builder
instruction_preview = build_instruction_text(task_request or "", scenario_key)

# Preview + actions in a clean footer card
# st.markdown(
#     f"""
# <div class="card cta-wrap">
#   <h3 style="margin: 0 0 10px 0;">Ready to call</h3>
#   <div class="small-muted">Scenario focus: {scenario_meta['name']}.</div>
#   <div class="small-muted">We’ll generate the assistant’s internal instructions from your request.</div>
# </div>
# """,
#     unsafe_allow_html=True,
# )

if instruction_preview:
    preview_label = f"View generated assistant instructions ({scenario_meta['name']})"
    with st.expander(preview_label, expanded=False):
        st.code(instruction_preview, language="markdown")

col1, col2 = st.columns([1, 1], gap="medium")
with col1:
    call_btn = st.button("📲 Make Call", type="primary", use_container_width=True)
with col2:
    st.button("Clear", use_container_width=True, on_click=lambda: st.session_state.clear())

# ---- Action ----
if call_btn:
    if not BASE_URL:
        st.error("Please set DEFAULT_BASE_URL near the top of streamlit_app.py.")
        st.stop()
    if not to_number.strip():
        st.warning("Please enter the phone number you want to call.")
        st.stop()
    if not instruction_preview:
        st.warning("Please describe the task so the assistant knows what to do.")
        st.stop()

    endpoint = f"{BASE_URL}/make-call"
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