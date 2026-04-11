from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool
from langchain.agents import create_agent
from typing import Any, Optional, List

class Agent:
    """
    Reusable wrapper around LangGraph's prebuilt create_react_agent.
    Mirrors the style of the Supervisor wrapper and standardizes agent
    construction: accepts LLM, tools, optional prompt.
    Build returns a StateGraph; compile returns a runnable.
    """

    def __init__(self, agent_name: str = "agent", llm: Optional[BaseChatModel] = None, tools: Optional[List[Tool]] = None, prompt: Optional[str] = None):
        self.agent_name = agent_name
        self.llm = llm
        self.tools = list(tools or [])
        self.prompt = prompt
        self.agent: Optional[Any] = None


    def build(self):
        if self.llm is None:
            raise ValueError("LLM (model) must be provided to build the agent")

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.prompt,
        )

        return self.agent

    def invoke(self, state):
        if not self.agent:
            self.build()
        return self.agent.invoke(state)
    
    def stream(self, state):
        if not self.agent:
            self.build()
        for chunk in self.agent.stream(state):
            yield chunk

