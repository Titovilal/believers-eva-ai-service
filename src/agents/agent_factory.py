"""
Agent Factory Module
This module combines the base agent functionality, SQL database agent, document retrieval agent,
and the agent factory into a single file for easier management and usage.
"""

from abc import ABC, abstractmethod
import logging

import time
import os
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
from src.retrievers import Retriever
from .prompts import eva_esg_efrag_sys_prompt
from .tools import retrieve_from_documents
from .models import Deps
from pydantic_ai.messages import ModelMessagesTypeAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class ModelRegistry:
    """Simple registry for available LLM models"""

    # List of available models
    AVAILABLE_MODELS = [
        "google-gla:gemini-2.0-flash",
        "google-gla:gemini-2.5-flash-preview-04-17",
        "openai:gpt-4o",
    ]
    # Dictionary for OpenRouter models with their identifiers
    OPENROUTER_MODELS = [
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-r1",
        # "google/gemini-2.0-flash-001",
        # "qwen/qwen2.5-coder-7b-instruct"
    ]

    @classmethod
    def get_available_models(cls):
        """Return the list of all available models"""
        return cls.AVAILABLE_MODELS + cls.OPENROUTER_MODELS

    @classmethod
    def get_default_model(cls):
        """Get the default model for a specific agent type"""
        return cls.AVAILABLE_MODELS[0]

    @classmethod
    def is_valid_model(cls, model_id):
        """Check if a model ID is valid"""
        return model_id in cls.AVAILABLE_MODELS or model_id in cls.OPENROUTER_MODELS

    @classmethod
    def get_model(cls, model_id):
        """
        Returns either a string model identifier or a configured model object based on the model_id.
        For OpenRouter models, returns an OpenAIModel instance with appropriate configuration.
        """
        # For standard models, just return the model ID string
        if model_id in cls.AVAILABLE_MODELS:
            return model_id

        # For OpenRouter models, return a configured OpenAIModel
        if model_id in cls.OPENROUTER_MODELS:
            return OpenAIModel(
                model_id,
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        raise ValueError(f"Unknown model ID: {model_id}")


class BaseAgent(ABC):
    """Base abstract agent class that defines the common interface for all agents"""

    def __init__(self, model: str = "google-gla:gemini-2.0-flash", max_retry: int = 3):
        self.model_id = model
        self.model = ModelRegistry.get_model(model)
        self.max_retry = max_retry
        self.deps = None
        self.agent = None

    @abstractmethod
    def setup(self):
        """Setup the agent with proper tools and dependencies"""
        ...

    def run_sync(
        self, user_question: str, message_history: list[ModelMessage]
    ) -> AgentRunResult:
        ...
        # logger.info(f"User question: {user_question}")
        # self.deps.user_question = user_question

        # Start timing
        # start_time = time.time()
        # Run the agent
        # result = self.agent.run_sync(
        #    user_question, deps=self.deps, message_history=message_history
        # )
        # End timing
        # end_time = time.time()
        # self.deps.inference_time = end_time - start_time
        # logger.info(f"Inference time: {self.deps.inference_time:.2f} seconds")

        # Log the result
        # logger.info(f"Result: {result.data}")

        # return result


class EFRAGAgent(BaseAgent):
    """Agent for EFRAG-related tasks"""

    def __init__(
        self,
        model: str = "google-gla:gemini-2.0-flash",
        max_retry: int = 3,
        retriever: Retriever = None,
    ):
        self.model_id = model
        self.model = ModelRegistry.get_model(model)
        self.max_retry = max_retry
        self.deps = None
        self.agent = None
        self.retriever = retriever
        self.setup()

    def setup(self):
        """Setup the EFRAG agent with specific tools and dependencies"""
        # Setup logic for EFRAG agent
        self.agent = Agent(
            model=self.model,
            system_prompt=eva_esg_efrag_sys_prompt,
            deps_type=Deps,
            output_type=str,  # FinalResult,
            retries=self.max_retry,
            tools=[retrieve_from_documents],
        )

        self.deps = Deps(retriever=self.retriever)

    def run_sync(
        self, user_question: str, message_history: list[ModelMessage]
    ) -> AgentRunResult:
        logger.info(f"User question: {user_question}")
        self.deps.user_question = user_question

        # Start timing
        start_time = time.time()

        # Run the agent
        result = self.agent.run_sync(
            user_question,
            deps=self.deps,
            message_history=ModelMessagesTypeAdapter.validate_python(message_history),
        )
        # End timing
        end_time = time.time()
        self.deps.inference_time = end_time - start_time
        logger.info(f"Inference time: {self.deps.inference_time:.2f} seconds")

        # Log the result
        logger.info(f"Result: {result.data}")

        return result


class AgentFactory:
    """Factory class for creating different types of agents"""

    @staticmethod
    def create_agent(retriever: Retriever, model: str = None, **kwargs):
        """
        Create an agent based on the specified type

        Args:
            db_type: Type of agent to create ("business", "drilling", "document_retriever")
            model_id: ID of the model to use (optional, will use default if None)
            **kwargs: Additional parameters for the specific agent type

        Returns:
            An instance of the appropriate agent
        """

        # Use default model if none specified
        if model is None:
            model = ModelRegistry.get_default_model()

        # Validate model
        if not ModelRegistry.is_valid_model(model):
            raise ValueError(
                f"Invalid model ID: {model}. Available models: {ModelRegistry.get_available_models()}"
            )

        return EFRAGAgent(model=model, retriever=retriever, **kwargs)

    @staticmethod
    def get_available_models():
        """Return list of available models"""
        return ModelRegistry.get_available_models()

    @staticmethod
    def get_default_model():
        """Return default model"""
        return ModelRegistry.get_default_model()
