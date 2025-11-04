"""
Agent Factory Module
Factory for creating different types of agents.
"""

from abc import ABC, abstractmethod
import time

from pydantic_ai import Agent, ModelMessage
from pydantic_ai.result import FinalResult
from pydantic_ai.models.openai import OpenAIChatModel

from utils.logs import log_info
from src.agents.models import BaseDeps


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, model: str = "google/gemini-2.5-flash"):
        self.model_name = model
        self.model = OpenAIChatModel(
            model,
            provider="openrouter",
        )
        self.deps = None
        self.agent = None

    @abstractmethod
    def setup(self):
        """Setup the agent with tools and dependencies"""
        pass

    async def run_async(
        self, user_question: str, message_history: list[ModelMessage] = None
    ) -> FinalResult:
        """Async version of run_sync for use with FastAPI"""

        # 1. Start timing
        start_time = time.time()

        # 2. Run the agent asynchronously
        if message_history is None:
            message_history = []

        result = await self.agent.run(
            user_question, deps=self.deps, message_history=message_history
        )

        # 3. End timing
        end_time = time.time()
        self.deps.inference_time = end_time - start_time
        log_info(f"Inference time: {self.deps.inference_time:.2f} seconds")

        return result


class SimpleAgent(BaseAgent):
    """Simple concrete implementation of BaseAgent."""

    def setup(self):
        """Setup the agent with basic dependencies"""
        self.deps = BaseDeps()
        self.agent = Agent(self.model, deps_type=BaseDeps)


class AgentFactory:
    """Factory class to create agents based on type."""

    @staticmethod
    def create_agent(
        agent_type: str, openrouter_model: str = "gemini-2.5-flash"
    ) -> BaseAgent:
        """Create an agent based on the specified type."""
        if agent_type == "simple":
            return SimpleAgent(model=openrouter_model)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
