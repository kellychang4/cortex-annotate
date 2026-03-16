# -*- coding: utf-8 -*-
################################################################################
# annotate/_prefs.py

"""User preferences management for cortex-annotate.

PrefsManager handles loading, saving, and accessing user preferences that
persist between annotation sessions. Preferences are organized into three
sections:

    display          : general tool layout and sizing (figure_size, layout, etc.)
    annotation_style : per-annotation visual properties (color, linewidth, etc.)
    viewer_style     : 3D cortex viewer settings (morph, overlay, etc.)

Preferences are stored as a YAML file in the user's save directory. On first
run, defaults are built from the style constants in ``_style.py`` and any
overrides in ``config.yaml``.
"""

# Imports ----------------------------------------------------------------------

import yaml
import os.path as op

from ._style import (
    DEFAULT_ANNOTATION_STYLE, ANNOTATION_STYLE_KEYS,
    DEFAULT_VIEWER_STYLE, VIEWER_STYLE_KEYS,
    validate_annotation_style, validate_viewer_style,
)

# Default display preferences (general tool layout and sizing).
DEFAULT_DISPLAY_PREFS = {
    "figsize" : [4, 4],
    "dpi"     : 128,
    "layout"  : "horizontal",
}

DISPLAY_PREFS_KEYS = tuple(DEFAULT_DISPLAY_PREFS.keys())

# Preference Manager Class -----------------------------------------------------

class PrefsManager:
    """Manages user preferences that persist between annotation sessions.

    Parameters
    ----------
    config : Config
        Parsed configuration object, used to build annotation style defaults
        from ``config.display.default_style`` and ``config.display.active_style``.
    preferences_file : str
        Path to the YAML preferences file on disk.

    Attributes
    ----------
    preferences : dict
        In-memory preferences with keys ``"annotation_style"``,
        ``"viewer_style"``, ``"display"``.
    """

    def __init__(self, config, preferences_file = ".annot-prefs.yaml"):
        # Store the config for building defaults. 
        self.config = config
        
        # Store the preferences file paths.
        self.preferences_file = preferences_file

        # Load the preferences (from file if it exists, otherwise default).
        self.preferences = self._load()

    
    def _load(self):
        """Loads preferences from disk, or builds defaults if no file exists."""
        if op.isfile(self.preferences_file):
            with open(self.preferences_file, "rt") as f:
                return yaml.safe_load(f)
        return self._build_defaults()


    def _build_defaults(self):
        """Constructs the default preferences dictionary from style constants 
        and config overrides.

        Display preference priority (lowest → highest):
            DEFAULT_DISPLAY_PREFS → config.display

        Annotation style priority (lowest → highest):
            DEFAULT_ANNOTATION_STYLE → config.display.default_style
        
        Active annotation style priority (lowest → highest):
            DEFAULT_ANNOTATION_STYLE → config.display.default_style → config.display.active_style

        Viewer style priority (lowest → highest):
            DEFAULT_VIEWER_STYLE → config.display.viewer_style
        """
        # Initialize the preferences dict with the defaults.
        preferences = {
            "display": DEFAULT_DISPLAY_PREFS.copy(),
            "annotation_style": {},
            "viewer_style": DEFAULT_VIEWER_STYLE.copy(),
        }

        # Build the annotation style: module defaults + config overrides.
        annotation_style = {
            **DEFAULT_ANNOTATION_STYLE,
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
        """Writes the current preferences to disk as YAML."""
        with open(self.preferences_file, "wt") as f:
            yaml.dump(self.preferences, f)


    # Display Methods ----------------------------------------------------------

    def get_display(self, key = None):
        """Returns display preference values.

        Parameters
        ----------
        key : str or None
            If None, returns the full display prefs dict.
            If a string, returns the single value for that key.

        Returns
        -------
        dict or scalar
            Full display dict, or a single value if key was given.
        """
        display = self.preferences["display"]
        if key is None: return display
        if key not in DISPLAY_PREFS_KEYS:
            raise RuntimeError(f"Invalid display preference key: {key}")
        return display.get(key)


    def set_display(self, key, value):
        """Sets a single display preference value.

        Parameters
        ----------
        key : str
            A display preference key (e.g. 'figure_size', 'layout').
        value : scalar
            The new value for this key.

        Returns
        -------
        dict
            The full updated display prefs dict.
        """
        if key not in DISPLAY_PREFS_KEYS:
            raise RuntimeError(f"Invalid display preference key: {key}")
        display = self.preferences["display"]
        display[key] = value
        return display

    # Annotation Style Methods -------------------------------------------------

    def get_annotation_style(self, annotation):
        """Returns the full style dict for the given annotation.

        Parameters
        ----------
        annotation : str or None
            The annotation name, or None for the active annotation style.

        Returns
        -------
        dict
            Style dict with keys: color, linewidth, linestyle, markersize, 
            visible.
        """
        styles = self.preferences["annotation_style"]
        return styles.get(annotation, {})


    def set_annotation_style(self, annotation, updates):
        """Merge style updates into the given annotation's style dict.

        Validates incoming keys against ``ANNOTATION_STYLE_KEYS`` and
        normalizes colors to hex before merging.

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

        Raises
        ------
        RuntimeError
            If any key in ``updates`` is not a valid annotation style key.
        """
        # Validate the incoming keys and values.
        validate_annotation_style(updates)

        # Merge into existing style.
        styles = self.preferences["annotation_style"]
        current = styles.get(annotation, {})
        styles[annotation] = {**current, **updates}
        return styles[annotation]

    # Viewer Style Methods -----------------------------------------------------

    def get_viewer_style(self, key = None):
        """Returns viewer style values.

        Parameters
        ----------
        key : str or None
            If None, returns the full viewer style dict.
            If a string, returns the single value for that key.

        Returns
        -------
        dict or scalar
            Full style dict, or a single value if key was given.
        """
        styles = self.preferences["viewer_style"]
        if key is None: return styles
        if key not in VIEWER_STYLE_KEYS:
            raise RuntimeError(f"Invalid viewer style key: {key}")
        return styles.get(key)


    def set_viewer_style(self, key, value):
        """Sets a single viewer style value.

        Parameters
        ----------
        key : str
            Must be one of the VIEWER_STYLE_KEYS.
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