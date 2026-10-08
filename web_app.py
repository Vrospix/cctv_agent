"""Streamlit chat interface for the CCTV query agent."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import streamlit as st

from cctv_agent.agent import CCTVQueryAgent
from cctv_agent.database import FrameDatabase
from cctv_agent.llm_intent import DEFAULT_MODEL, OllamaIntentParser
from cctv_agent.models import ConversationState


DATABASE_PATH = Path(__file__).parent / "data" / "cctv_frames.db"
WELCOME = (
    "Ask me to find Singapore expressway CCTV frames. You can refine the camera, "
    "date, time, sorting, or result page in follow-up messages."
)
EXAMPLES = (
    "Show ECP frames today",
    "Show CTE and PIE frames from the last 3 days",
    "Give me 5 random TPE frames this month",
    "Show one frame per hour from every camera except ECP today",
)


def frame_rows(response) -> list[dict[str, object]]:
    return [
        {
            "Frame ID": frame.frame_id,
            "Captured at": frame.captured_at,
            "Camera": frame.camera_name,
        }
        for frame in response.frames
    ]


def submit(message: str, *, use_local_ai: bool) -> None:
    database = FrameDatabase(DATABASE_PATH)
    try:
        today = date.today()
        database.ensure_current(today=today)
        intent_parser = OllamaIntentParser() if use_local_ai else None
        agent = CCTVQueryAgent(
            database,
            today=today,
            intent_parser=intent_parser,
        )
        agent.state = st.session_state.agent_state
        response = agent.ask(message)
        st.session_state.agent_state = agent.state
        st.session_state.filter_description = agent.describe_context()
        response_text = response.message
        if agent.last_intent_source == "local_ai":
            response_text = "Local AI intent interpretation used.\n\n" + response_text
        elif use_local_ai and agent.last_llm_error:
            if agent.last_llm_error == "The local Ollama service is unavailable.":
                ai_note = "Local AI could not be reached, so the deterministic parser was used."
            else:
                ai_note = (
                    "Local AI was reached, but its interpretation did not pass validation. "
                    "The deterministic parser was used instead."
                )
            response_text += "\n\n" + ai_note
    finally:
        database.close()
    st.session_state.messages.append({"role": "user", "content": message, "frames": []})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "frames": frame_rows(response),
        }
    )


st.set_page_config(page_title="CCTV Frame Query Agent", layout="wide")

if "agent_state" not in st.session_state:
    st.session_state.agent_state = ConversationState()
if "filter_description" not in st.session_state:
    st.session_state.filter_description = "No active filters."
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME, "frames": []}
    ]
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

with st.sidebar:
    st.header("CCTV Query Agent")
    st.caption("Deterministic natural-language search over simulated 2026 CCTV frames.")
    use_local_ai = st.toggle(
        "Use local AI fallback",
        value=False,
        help=f"Uses Ollama with {DEFAULT_MODEL}; deterministic validation remains active.",
    )

    if st.button("Start over", use_container_width=True):
        st.session_state.agent_state = ConversationState(
            page_size=st.session_state.agent_state.page_size
        )
        st.session_state.filter_description = "No active filters."
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME, "frames": []}
        ]
        st.session_state.queued_prompt = None
        st.rerun()

    st.subheader("Try an example")
    for index, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example_{index}", use_container_width=True):
            st.session_state.queued_prompt = example

    st.divider()
    st.subheader("Current search")
    st.caption(st.session_state.filter_description)

st.title("Singapore Expressway CCTV Search")
st.caption("Search, refine, count, sort, and page through frame records conversationally.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["frames"]:
            st.dataframe(message["frames"], hide_index=True, use_container_width=True)

typed_prompt = st.chat_input("Ask for CCTV frames...")
prompt = st.session_state.queued_prompt or typed_prompt
if prompt:
    st.session_state.queued_prompt = None
    submit(prompt, use_local_ai=use_local_ai)
    st.rerun()
