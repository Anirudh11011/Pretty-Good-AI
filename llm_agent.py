# llm_agent.py
"""
LangChain (new style) call agent using RunnableWithMessageHistory.

This replaces ConversationBufferMemory with:
- InMemoryChatMessageHistory
- RunnableWithMessageHistory
- MessagesPlaceholder

Interface matches your existing main.py usage:
    agent = CallAgent.create(instructions)
    response_text = await agent.respond(user_text)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

load_dotenv()


@dataclass
class CallAgent:
    instructions: str
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2

    def __post_init__(self) -> None:
        # One history per CallAgent instance (your main.py already creates one agent per call).
        self._history = InMemoryChatMessageHistory()
        self._last_usage: Dict[str, Any] = {}

        # System message includes the user-provided instructions from Streamlit.
        system_text = (
            "You are Anirudh's real-time personal phone assistant. But Don't mention this in call\n"
            "If verification is asked my Date of birth is September 01, 2001\n"
            "Call etiquette (strict turn-taking):\n"
            "- When the call connects, remain silent until the other party completes their initial greeting.\n"
            "- You MUST NOT speak while the other party is speaking.\n"
            "- Only begin speaking after the other party has been silent for at least 1.0 second (1000 ms).\n"
            "- Treat pauses shorter than 1.0 second as the other party still speaking/thinking; continue listening.\n"
            "- Do not interrupt, do not 'acknowledge' mid-sentence (no 'okay', 'sure', 'mm-hmm') while they are speaking.\n"
            "- If you accidentally start speaking while they are speaking, STOP immediately, say: 'Sorry—go ahead.' then remain silent until they finish.\n"
            "- Ask at most ONE question per turn. After asking, stay silent and listen for the full answer.\n"
            "- Confirm critical details (date/time/name/phone/address) only after the other party finishes their turn.\n"
            f"Task briefing:\n{self.instructions}"
        )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        self._llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            # You can set GROQ_API_KEY in your environment/.env
        )

        # Base chain: prompt -> model
        base_chain = self._prompt | self._llm

        # RunnableWithMessageHistory needs a function that returns a history object.
        # We ignore session_id because this agent instance is already per-call.
        def _get_history(_session_id: str) -> BaseChatMessageHistory:
            return self._history

        self._chain = RunnableWithMessageHistory(
            base_chain,
            _get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    @classmethod
    def create(cls, instructions: str) -> "CallAgent":
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        temperature = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
        return cls(instructions=instructions, model_name=model_name, temperature=temperature)

    async def respond(self, user_text: str, session_id: Optional[str] = None) -> str:
        """
        Generate an assistant response while preserving conversation history.

        session_id is optional; if you pass Twilio CallSid, it will still work.
        """
        sid = session_id or "default"
        self._last_usage = {}

        result = await self._chain.ainvoke(
            {"input": user_text},
            config={"configurable": {"session_id": sid}},
        )

        response_metadata = getattr(result, "response_metadata", {}) or {}
        usage = getattr(result, "usage_metadata", None)
        if not usage:
            usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        self._last_usage = usage or {}

        # ChatGroq returns an AIMessage; its text is in .content
        return getattr(result, "content", str(result))

    @property
    def last_usage(self) -> Dict[str, Any]:
        return self._last_usage


#  "When the call connects, remain silent and listen until the other party finishes their initial greeting.\n"
#             "Follow the caller's instructions exactly, staying warm, concise, and professional.\n"
#             "Ask for clarifications one question at a time, avoid repeating yourself unless asked, and confirm important commitments aloud.\n"
#             "Whenever you ask a question, pause silently for about one second to give the other person room to respond.\n\n"
#             "And Don't talk until the other party has finished speaking.\n"