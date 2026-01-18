import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        description="The name of the language model to use for the agent's query generation.",
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash",
        description="The name of the language model to use for the agent's reflection.",
    )

    answer_model: str = Field(
        default="gemini-2.5-pro",
        description="The name of the language model to use for the agent's answer.",
    )

    number_of_initial_queries: int = Field(
        default=3,
        description="The number of initial search queries to generate.",
    )

    max_research_loops: int = Field(
        default=2,
        description="The maximum number of research loops to perform.",
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values and convert strings to appropriate types
        values = {}
        for k, v in raw_values.items():
            if v is not None:
                # Convert string values from environment variables to correct type
                field_info = cls.model_fields[k]
                if field_info.annotation in (int, int | None) and isinstance(v, str):
                    values[k] = int(v)
                else:
                    values[k] = v

        return cls(**values)
