from dataclasses import dataclass
from typing import Any, Callable, Dict

ToolFunc = Callable[..., Any]

@dataclass
class Tool:
    """
    Declarative description of a callable tool the agent can use.

    Attributes
    ----------
    name:
        Unique identifier used by the LLM to reference this tool.
    description:
        Short, user-facing description of what the tool does.
    parameters:
        JSON-schema-like description of the tool arguments.
    func:
        Python callable that implements the tool.
    timeout_seconds:
        Maximum time to allow the tool to run before timing out.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    func: ToolFunc
    timeout_seconds: float = 10.0

    def validate_args(self, args: Dict[str, Any]) -> None:
        """
        Validate that the provided arguments match the declared schema.

        Raises
        ------
        ValueError
            If required arguments are missing or unknown arguments are present.
        """
        required = self.parameters.get("required", [])
        props = self.parameters.get("properties", {})

        missing = [k for k in required if k not in args]
        if missing:
            raise ValueError(f"Missing required argument(s): {', '.join(missing)}")

        unknown = [k for k in args.keys() if k not in props]
        if unknown:
            raise ValueError(f"Unknown argument(s): {', '.join(unknown)}")


class ToolExecutionError(Exception):
    """Raised when a tool fails or times out."""
