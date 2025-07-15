# mcp_interface.py

from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field
from typing import List, Any

# Import our single, unified agent factory
from agent_core import create_agent_executor

# --- Pydantic Schemas ---


class ScraperInput(BaseModel):
    task: str = Field(description="The high-level scraping task.")


class ScraperOutput(BaseModel):
    answer: str = Field(description="The final answer or extracted data.")
    # Use 'Any' for intermediate steps as the format can vary
    intermediate_steps: List[Any] = Field(
        description="A log of the agent's actions.")

# --- The MCP Component ---


class AgenticScraper(Runnable[ScraperInput, ScraperOutput]):
    """
    An agentic web scraper that can be configured with any available backend
    via the unified agent factory.
    """

    def __init__(self, backend: str):
        """
        Initializes the scraper by calling the agent factory with the specified backend.
        """
        print(
            f"🚀 Initializing AgenticScraper with requested backend: '{backend}'.")
        # The MCP interface now has a single point of entry to create the agent
        self.agent_executor = create_agent_executor(backend=backend)

    def invoke(self, input: ScraperInput, config: RunnableConfig = None) -> ScraperOutput:
        result = self.agent_executor.invoke({"input": input.task}, config)

        # Access the memory directly for a reliable history
        history = self.agent_executor.memory.chat_memory.messages

        return ScraperOutput(
            answer=result.get("output", "No answer could be determined."),
            intermediate_steps=history
        )
