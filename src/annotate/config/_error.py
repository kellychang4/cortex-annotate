# -*- coding: utf-8 -*-
################################################################################
# annotate/config/_error.py

"""Configuration error types for cortex-annotate.

Provides ConfigError, a specialized exception raised when the project's
config.yaml file contains invalid or missing values.
"""

# Configuration Error ----------------------------------------------------------

class ConfigError(Exception):
    """Exception raised when config.yaml contains invalid or missing values.

    Parameters
    ----------
    section : str
        The dotted config path where the error was found
        (e.g. ``"display.figsize"`` or ``"annotations.`V1 Foveal Point`"``).
        
    message : str
        A human-readable description of the validation failure.
    """
    
    __slots__ = ( )
   
    def __init__(self, section, message): 
        super().__init__(
            f"Invalid `config.{section}`\n"
            f"{message}"
        )