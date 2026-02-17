import os
from dataclasses import dataclass

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


@dataclass
class CallAgent:
    instructions: str
    memory: ConversationBufferMemory
    llm: ChatGroq

    @classmethod
    def create(cls, instructions: str) -> "CallAgent":
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-70b-8192",
            temperature=0.3,
        )
        memory = ConversationBufferMemory(return_messages=True)
        return cls(instructions=instructions, memory=memory, llm=llm)

    async def respond(self, user_text: str) -> str:
        messages = [SystemMessage(content=self.instructions)]

        # include chat history
        if hasattr(self.memory, "chat_memory") and getattr(self.memory.chat_memory, "messages", None):
            messages.extend(self.memory.chat_memory.messages)

        messages.append(HumanMessage(content=user_text))

        ai = await self.llm.ainvoke(messages)
        response_text = (ai.content or "").strip()

        # update memory
        self.memory.chat_memory.add_user_message(user_text)
        self.memory.chat_memory.add_ai_message(response_text)
        return response_text
