from random import uniform
from src.agents.models import BaseDeps
from pydantic_ai import RunContext

from src.utils.logs import log_exception, log_info


async def get_city_temperature(ctx: RunContext[BaseDeps], city: str) -> str:
    """
    Get the current temperature in °C for a specific city.

    Args:
        city (str): The city for which to get the temperature.

    Returns:
        Formatted string with the current temperature value
    """
    log_info(f"Tool called: get_city_temperature for {city}")

    try:
        # Here you can use ctx.deps to retrieve any dependencies if needed
        # Example: inference_time = ctx.deps.inference_time
        # More at https://ai.pydantic.dev/dependencies

        temperature = uniform(0, 40)
        return f"Current temperature in {city}: {temperature:.1f}°C"

    except Exception as e:
        log_exception(f"Error getting current temperature: {str(e)}")
        return f"Error getting current temperature: {str(e)}"
