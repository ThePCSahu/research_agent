"""
Config helper module.

Fetches configuration values with the following priority:
  1. .env file (loaded via python-dotenv)
  2. OS environment variables
  3. Raises RuntimeError if not found in either place
"""

import os
from dotenv import load_dotenv, find_dotenv

import logging
logger = logging.getLogger(__name__)

# Load .env file if it exists; searches parent directories if needed.
# override=True ensures .env values take precedence over existing shell variables.
if load_dotenv(find_dotenv(), override=True):
    logger.info("Environment variables loaded from .env")
else:
    logger.warning("No .env file found via find_dotenv()")


# Configuration constants moved to .env


class ConfigError(RuntimeError):
    """Raised when a required configuration key is missing."""


def resolve_env_value(value: str) -> str:
    """Recursively resolve environment variable references in the format ${VAR_NAME}."""
    if value.startswith('${') and value.endswith('}'):
        var_name = value[2:-1]
        resolved = os.environ.get(var_name)
        if resolved is None:
            raise ConfigError(f"Referenced environment variable '{var_name}' not found.")
        return resolve_env_value(resolved)
    else:
        return value


def get_config(key: str) -> str:
    """Return the value for *key* from the environment.

    Resolution order:
      1. Value loaded from the ``.env`` file (already merged into
         ``os.environ`` by ``load_dotenv``).
      2. Value present in OS environment variables.
      3. If the key is absent from both, a ``ConfigError`` (subclass of
         ``RuntimeError``) is raised.

    Parameters
    ----------
    key : str
        The configuration key to look up (case-sensitive).

    Returns
    -------
    str
        The configuration value.

    Raises
    ------
    ConfigError
        If *key* is not found in the .env file or the OS environment.
    """
    value = os.environ.get(key)
    if value is None:
        raise ConfigError(
            f"Required configuration '{key}' is not set. "
            f"Please add it to a .env file or set it as an environment variable."
        )
    return resolve_env_value(value)


def get_config_or_default(key: str, default: str) -> str:
    """Return the value for *key*, falling back to *default* if absent.

    Same resolution as :func:`get_config` but never raises – returns
    *default* instead when the key is missing from both the ``.env``
    file and OS environment variables.

    Parameters
    ----------
    key : str
        The configuration key to look up (case-sensitive).
    default : str
        Value to return when the key is not found.

    Returns
    -------
    str
        The configuration value, or *default*.
    """
    value = os.environ.get(key, default)
    return resolve_env_value(value)
