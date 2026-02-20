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
from typing import Optional

from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq


@dataclass
class CallAgent:
    instructions: str
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2

    def __post_init__(self) -> None:
        # One history per CallAgent instance (your main.py already creates one agent per call).
        self._history = InMemoryChatMessageHistory()

        # System message includes the user-provided instructions from Streamlit.
        system_text = (
            "You are a real-time AI voice assistant speaking on a phone call.\n"
            "Follow the user's instructions exactly.\n"
            "Be concise, natural, and conversational.\n"
            "Ask one question at a time when you need details.\n\n"
            f"User instructions:\n{self.instructions}"
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

        result = await self._chain.ainvoke(
            {"input": user_text},
            config={"configurable": {"session_id": sid}},
        )

        # ChatGroq returns an AIMessage; its text is in .content
        return getattr(result, "content", str(result))
