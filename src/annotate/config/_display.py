# -*- coding: utf-8 -*-
################################################################################
# annotate/config/_display.py

"""Display configuration for cortex-annotate.

DisplayConfig parses the optional ``display`` section of config.yaml,
which controls figure-generation parameters (figsize, dpi) and base
annotation style overrides (default_style, active_style). An optional
``layout`` value specifies the default panel orientation.
"""

# Imports ----------------------------------------------------------------------

from functools import partial
from numbers import Real, Integral

from ._error  import ConfigError
from .._style import validate_annotation_style

# Display Configuration --------------------------------------------------------

class DisplayConfig():
    """Display settings parsed from the ``display`` section of config.yaml.

    Controls figure generation parameters and base annotation style
    overrides. All attributes are read-only after construction.

    Parameters
    ----------
    display_yaml : dict or None
        The ``display`` section from config.yaml. If ``None``, all
        attributes default to ``None`` or empty dicts.

    Attributes
    ----------
    figsize : tuple of float or None
        Matplotlib figure size as ``(width, height)``. ``None`` if not
        specified in config.

    dpi : int or None
        Matplotlib DPI for figure generation. ``None`` if not specified.

    default_style : dict
        Base annotation style overrides applied to all annotations.
        Empty dict if not specified.

    active_style : dict
        Style overrides applied only to the currently active
        annotation. Empty dict if not specified.

    layout : str or None
        Default panel layout, either ``"horizontal"`` or
        ``"vertical"``. ``None`` if not specified.
    """

    __slots__ = ( "figsize", "dpi", "active_style", "default_style", "layout" )
    
    def __init__(self, display_yaml):
        # The display section is optional. If None, return empty dictionary.
        if display_yaml is None: display_yaml = {}
        
        # Initialize the figure size.
        self.figsize = self._init_figsize(display_yaml)

        # Initialize the DPI. 
        self.dpi = self._init_dpi(display_yaml)

        # Initialize the active style.
        self.active_style = self._init_style(
            display_yaml, style_name = "active_style", default = {})

        # Initialize the default style.
        self.default_style = self._init_style(
            display_yaml, style_name = "default_style", default = {})
        
        # Initialize the layout option. 
        self.layout = self._init_layout(display_yaml)


    @staticmethod
    def _init_figsize(display_yaml):
        """Validate and normalize the ``figsize`` value.

        Accepts a single number (used for both dimensions), a
        two-element list/tuple, or ``None``.

        Parameters
        ----------
        display_yaml : dict
            The display YAML mapping.

        Returns
        -------
        tuple of float or None
            A ``(width, height)`` tuple, or ``None`` if not specified.
        """
        # Prepare ConfigError arguments for any errors that may arise in this function.
        err = partial(ConfigError, "display.figsize")

        # Extract the figure size from the yaml. 
        figsize = display_yaml.get("figsize", None)

        # If the figure size is None, return None (will use defaults)
        if figsize is None: return None

        # Check that the figure size is not a string.
        if isinstance(figsize, str):
            raise err(f"figsize cannot be a string: {figsize}")

        # If the figure size is a single number, use both for the dimensions.
        if isinstance(figsize, (int, float)):
            figsize = [figsize, figsize]

        # If the figure size is a list/tuple, check there are two dimensions.
        if isinstance(figsize, (list, tuple)):
            # Check that there are two dimensions.
            if len(figsize) != 2:
                raise err(
                    f"figsize must be a number or 2-element list/tuple: "
                    f"{figsize}"
                )
            
            # Check that the figure size elements are positive numbers.
            if not all(isinstance(u, Real) and u > 0 for u in figsize):
                raise err(
                    f"figsize elements must be positive numbers: "
                    f"{figsize}",
                )
        
        # Return the figure size as a tuple in the form (width, height).
        return tuple(figsize)


    @staticmethod
    def _init_dpi(display_yaml):
        """Validate the ``dpi`` value.

        Parameters
        ----------
        display_yaml : dict
            The display YAML mapping.

        Returns
        -------
        int or None
            A positive integer, or ``None`` if not specified.
        """
        # Prepare ConfigError arguments for any errors that may arise in this function.
        err = partial(ConfigError, "display.dpi")

        # Extract the DPI from the yaml.
        dpi = display_yaml.get("dpi", None)

        # If the DPI is None, return None (will use defaults)
        if dpi is None: return None

        # Check that the DPI is a positive integer.
        if not isinstance(dpi, Integral) or dpi < 1:
            raise err(f"dpi must be a positive integer: {dpi}")
        
        # Return the DPI.
        return dpi
    

    @staticmethod
    def _init_style(display_yaml, style_name, default = {}):
        """Validate an annotation style override dict.

        Reads the named key from the display YAML, checks that it is a
        dict, and passes it through
        :func:`~annotate._style.validate_annotation_style`.

        Parameters
        ----------
        display_yaml : dict
            The display YAML mapping.

        style_name : str
            Key to read (``"default_style"`` or ``"active_style"``).

        default : dict, optional
            Fallback value if the key is absent (default empty dict).

        Returns
        -------
        dict
            The validated style dict.
        """
        # Prepare ConfigError arguments for any errors that may arise in this function.
        err = partial(ConfigError, f"display.{style_name}")

        # Extract the style from the yaml.
        style = display_yaml.get(style_name, default)

        # Check that the style is a yaml mapping (dictionary)
        if not isinstance(style, dict):
            raise err(f"{style_name} must be a mapping")
        
        # Try to make sure the style keys are valid
        try: validate_annotation_style(style)
        except RuntimeError as e: raise err(e) from e

        return style

    
    @staticmethod
    def _init_layout(display_yaml):
        """Validate the ``layout`` value.

        Parameters
        ----------
        display_yaml : dict
            The display YAML mapping.

        Returns
        -------
        str or None
            ``"horizontal"``, ``"vertical"``, or ``None`` if not
            specified.
        """
        # Prepare ConfigError arguments for any errors that may arise in this function.
        err = partial(ConfigError, f"display.layout")

        # Extract the layout option from the yaml.
        layout = display_yaml.get("layout", None)

        # If the layout option is None, return None (will use defaults)
        if layout is None: return None

        # Check that the layout option is a string.
        if not isinstance(layout, str):
            raise err(f"layout must be a string: {layout}")
        
        # Check that the layout option is either "horizontal" or "vertical".
        if layout not in ("horizontal", "vertical"):
            raise err(f"layout must be 'horizontal' or 'vertical': {layout}")
        
        # Return the layout option.
        return layout
