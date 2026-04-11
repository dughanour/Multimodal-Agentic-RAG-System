from typing import TypedDict, Literal
from src.state.state import State
from langgraph.types import Command
from langgraph.graph import END
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.prompts.prompts import MAIN_SUPERVISOR_PROMPT, FINAL_ANSWER_PROMPT

MAX_TURNS = 1

class Supervisor:
    """A class to manage the supervisor agent's state and logic."""
    custom_instructions: str = ""

    def __init__(self,agents: list[str], llm: BaseChatModel):
        self.agents = agents
        self.llm = llm

    def build_supervisor_node(self) -> str:
        """Build a supervisor node for the agents"""
        options = ["FINISH"] + self.agents

        class Router(TypedDict):
            """Worker to route to next. If no workers needed, route to FINISH."""
            next: Literal[*options]
        
        def supervisor_node(state: State) -> Command[Literal[*self.agents, "__end__"]]:
            """An LLM-based router."""
            
            # Check if we've exceeded the maximum number of turns
            if state.get("turns", 0) >= MAX_TURNS:
                # If so, force the generation of the final answer
                final_answer_messages = [
                    SystemMessage(content=FINAL_ANSWER_PROMPT)
                ] + state["messages"]
                final_answer = self.llm.invoke(final_answer_messages)
                return Command(
                    update={"messages": [AIMessage(content=final_answer.content, name="supervisor_final_answer")]},
                    goto=END
                )

            # --- DYNAMIC PROMPT INJECTION ---
            instruction_text = ""
            # If custom instructions are provided, format them into a string.
            if Supervisor.custom_instructions:
                print(f"Applying custom instructions: {Supervisor.custom_instructions}")
                instruction_text = f"\nMust to follow these instructions :{Supervisor.custom_instructions}\n"
            
            # Format the final system prompt with the (potentially empty) instructions
            system_prompt = MAIN_SUPERVISOR_PROMPT.format(
                agents=self.agents,
                custom_instructions=instruction_text
            )
            print(f"System prompt: {system_prompt}")

            # Otherwise, let the LLM decide the next step
            messages = [
                SystemMessage(content=system_prompt),
            ] + state["messages"]
            response = self.llm.with_structured_output(Router).invoke(messages)
            
            if response['next'] == "FINISH":
                print(f"Final answer: {response['next']}")
                
                # The full context, including images from the retrieval_agent_node,
                # is already in the state's messages. We can just pass them to the LLM.
                final_answer_messages = [
                    SystemMessage(content=FINAL_ANSWER_PROMPT)
                ] + state["messages"]

                # Generate the final answer
                final_answer = self.llm.invoke(final_answer_messages)
                return Command(
                    update={"messages": [HumanMessage(content=final_answer.content, name="supervisor_final_answer")]},
                    goto=END
                )

            # Route to the next agent
            print(f"route to: {response['next']}")
            print(f"options: {options}")      
            return Command(goto=response['next'], update={"next": response['next'], "turns": 1})

        return supervisor_node


def configure_supervisor(instructions: str):
    """A function to set the supervisor's custom instructions for the next call."""
    Supervisor.custom_instructions = instructions


        