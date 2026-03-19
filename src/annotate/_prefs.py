# -*- coding: utf-8 -*-
################################################################################
# annotate/_prefs.py

"""User preferences management for cortex-annotate.
 
PrefsManager handles loading, saving, and accessing user preferences that
persist between annotation sessions. Preferences are organized into three
sections:
 
    display          : general tool layout, sizing, and figure generation
                       parameters (figsize, dpi, image_pixel, layout)
    annotation_style : per-annotation visual properties (color, linewidth, etc.)
    viewer_style     : 3D cortex viewer settings (morph, overlay, etc.)
 
Preferences are stored as a YAML file in the user's save directory. On first
run, defaults are built from the style constants in ``_style.py``, overridden
by any values specified in the ``display`` section of ``config.yaml``.
 
Priority chain (lowest to highest):
 
    Module constants (``DEFAULT_*``)
        → ``config.yaml`` display section (read-only overrides)
            → saved preferences file on disk
                → runtime changes via UI widgets
"""

# Imports ----------------------------------------------------------------------

import yaml
import os.path as op
 
from ._style import (
    DEFAULT_ANNOTATION_STYLE,
    DEFAULT_VIEWER_STYLE,
    VIEWER_STYLE_KEYS,
    validate_annotation_style,
)

# Constants --------------------------------------------------------------------

# Default display preferences.
# NOTE: image_pixel is not included here because it is computed from
# figsize and dpi in _build_defaults(). 
DEFAULT_DISPLAY_PREFS = {
    "figsize" : [4, 4],
    "dpi"     : 128,
    "layout"  : "horizontal",
}
 
# All valid display preference keys (for get_display validation).
# Includes image_pixel which is computed, not stored in the constant above.
DISPLAY_PREFS_KEYS = ( "figsize", "dpi", "image_pixel", "layout" )
 
# Keys that can be modified at runtime via set_display().
_SETTABLE_DISPLAY_KEYS = ( "image_pixel", "layout" )
 
# Preference Manager Class -----------------------------------------------------

class PrefsManager:
    """Manages user preferences that persist between annotation sessions.
 
    Parameters
    ----------
    config : Config
        Parsed configuration object. Used to build defaults from
        ``config.display.default_style``, ``config.display.active_style``,
        ``config.display.figsize``, ``config.display.dpi``, and
        ``config.display.layout``.
 
    paths : PathManager
        Path manager, used to resolve the preferences file location.
 
    preferences_file : str
        Preferences filename (default ``".annot-prefs.yaml"``). Resolved
        against the save path via ``paths.get_preferences_path()``.
 
    Attributes
    ----------
    config : Config
        The configuration object, retained for building defaults.
 
    paths : PathManager
        The path manager.
 
    preferences_file : str
        Absolute filesystem path to the preferences YAML file.
 
    preferences : dict
        In-memory preferences dict with keys ``"display"``,
        ``"annotation_style"``, and ``"viewer_style"``.
    """

    __slots__ = ( "config", "paths", "preferences_file", "preferences" )
 
    def __init__(self, config, paths, preferences_file = ".annot-prefs.yaml"):
        # Store the arguments.
        self.config = config
        self.paths  = paths
 
        # Resolve the preferences file path.
        self.preferences_file = self.paths.get_preferences_path(preferences_file)
 
        # Load the preferences (from file if it exists, otherwise default).
        self.preferences = self._load()

    # Load and Save Methods ----------------------------------------------------

    def _load(self):
        """Load preferences from disk, or build defaults if no file exists.
 
        Returns
        -------
        dict
            The preferences dict with keys ``"display"``,
            ``"annotation_style"``, and ``"viewer_style"``.
        """
        if op.isfile(self.preferences_file):
            with open(self.preferences_file, "rt") as f:
                return yaml.safe_load(f)
        return self._build_defaults()

 
    def _build_defaults(self):
        """Construct the default preferences from constants and config overrides.
 
        Merges module-level defaults with any values specified in the
        ``display`` section of ``config.yaml``. Config values override
        module constants where present; missing config values fall back
        to the module constant.
 
        Returns
        -------
        dict
            A fully populated preferences dict ready for use.
 
        Notes
        -----
        Priority chains (lowest → highest):
 
        Display:
            ``DEFAULT_DISPLAY_PREFS`` → ``config.display`` (matching keys)
 
        Annotation style:
            ``DEFAULT_ANNOTATION_STYLE`` → ``config.display.default_style``
            Active annotation: above chain → ``config.display.active_style``
 
        Viewer style:
            ``DEFAULT_VIEWER_STYLE`` (no config override currently)
        """
        # Initialize the preferences dictionary.
        preferences = {
            "display"          : {},
            "annotation_style" : {},
            "viewer_style"     : DEFAULT_VIEWER_STYLE.copy(),
        }
 
        # Extract the display preferences from the config, if they exist.
        config_display = {}
        for key in ( "figsize", "dpi", "layout" ):
            if getattr(self.config.display, key) is not None:
                config_display[key] = getattr(self.config.display, key)
 
        # Build the display prefs: module defaults + config overrides.
        display_prefs = {
            **DEFAULT_DISPLAY_PREFS.copy(),
            **config_display,
        }
 
        # Calculate the figure size in pixels from the display prefs.
        image_pixel = int(display_prefs["figsize"][0] * display_prefs["dpi"])
        display_prefs["image_pixel"] = image_pixel
 
        # Store the display prefs in the main preferences dict.
        preferences["display"] = display_prefs
 
        # Build the annotation style: module defaults + config overrides.
        annotation_style = {
            **DEFAULT_ANNOTATION_STYLE.copy(),
            **self.config.display.default_style,
        }
 
        # Set each annotation to the base style.
        for annotation in self.config.annotations.keys():
            preferences["annotation_style"][annotation] = annotation_style.copy()
 
        # Set active annotation style (key = None).
        active_style = { **annotation_style, **self.config.display.active_style }
        preferences["annotation_style"][None] = active_style.copy()
 
        # Return the fully built preferences dict.
        return preferences
 

    def save(self):
        """Write the current preferences to disk as YAML.
 
        The file is written to :attr:`preferences_file`. Overwrites
        any existing file.
        """
        with open(self.preferences_file, "wt") as f:
            yaml.dump(self.preferences, f)

    # Display Methods ----------------------------------------------------------
 
    def get_display(self, key = None):
        """Return display preference values.
 
        Parameters
        ----------
        key : str or None
            If ``None``, returns the full display prefs dict.
            If a string, returns the single value for that key.
 
        Returns
        -------
        dict or scalar
            Full display dict if *key* is ``None``, otherwise the
            value for the requested key.
        """
        display = self.preferences["display"]
        if key is None: return display
        if key not in DISPLAY_PREFS_KEYS:
            raise RuntimeError(f"Invalid display preference key: {key}")
        return display.get(key)
 

    def set_display(self, key, value):
        """Set a display preference.
 
        Only ``image_pixel`` and ``layout`` can be modified at runtime.
        The ``figsize`` and ``dpi`` keys are config-level generation
        parameters and cannot be changed after initialization.
 
        Validation constraints:
 
        - ``image_pixel``: positive integer.
        - ``layout``: one of ``("horizontal", "vertical")``.
 
        Parameters
        ----------
        key : str
            Must be one of ``_SETTABLE_DISPLAY_KEYS``
            (``"image_pixel"`` or ``"layout"``).
 
        value : int or str
            The new value for this key.
 
        Returns
        -------
        dict
            The full updated display prefs dict.
        """
        if key not in _SETTABLE_DISPLAY_KEYS:
            raise RuntimeError(
                f"Display key '{key}' is not settable at runtime. "
                f"Settable keys: {_SETTABLE_DISPLAY_KEYS}")
        self.preferences["display"][key] = value
        return self.preferences["display"]
    
    # Annotation Style Methods -------------------------------------------------
 
    def get_annotation_style(self, annotation):
        """Return the full style dict for an annotation.
 
        Parameters
        ----------
        annotation : str or None
            The annotation name, or ``None`` for the active annotation
            style.
 
        Returns
        -------
        dict
            Style dict with keys like ``color``, ``linewidth``, etc. 
        """
        styles = self.preferences["annotation_style"]
        return styles.get(annotation, )
 
 
    def set_annotation_style(self, annotation, updates):
        """Merge style updates into an annotation's style dict.
 
        Validates incoming style options before merging. See 
        ``validate_annotation_style()``.

        Parameters
        ----------
        annotation : str or None
            Annotation name, or ``None`` for the active annotation style.

        updates : dict
            Partial style dict. Colors are normalized to hex.
 
        Returns
        -------
        dict
            The full updated style dict for this annotation.
        """
        # Validate the incoming keys and values.
        validate_annotation_style(updates)
 
        # Merge into existing style.
        styles = self.preferences["annotation_style"]
        current = styles.get(annotation)
        styles[annotation] = { **current, **updates }
        return styles[annotation]
 
   # Viewer Style Methods -----------------------------------------------------
 
    def get_viewer_style(self, key = None):
        """Return viewer style values.
 
        Parameters
        ----------
        key : str or None
            If ``None``, returns the full viewer style dict.
            If a string, returns the single value for that key.
 
        Returns
        -------
        dict or scalar
            Full style dict if *key* is ``None``, otherwise the value
            for the requested key.
        """
        styles = self.preferences["viewer_style"]
        if key is None: return styles
        if key not in VIEWER_STYLE_KEYS:
            raise RuntimeError(f"Invalid viewer style key: {key}")
        return styles.get(key)
 
 
    def set_viewer_style(self, key, value):
        """Set a single viewer style value.
 
        Parameters
        ----------
        key : str
            Must be one of ``VIEWER_STYLE_KEYS``.
 
        value : scalar
            The new value for this key.
 
        Returns
        -------
        dict
            The full updated viewer style dict.
        """
        if key not in VIEWER_STYLE_KEYS:
            raise RuntimeError(f"Invalid viewer style key: {key}")
        styles = self.preferences["viewer_style"]
        styles[key] = value
        return styles
 