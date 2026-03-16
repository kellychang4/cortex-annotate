# -*- coding: utf-8 -*-
################################################################################
# annotate/_config.py

"""Configuration system for the cortex-annotate annotation tool.

This module defines the top-level Config class that loads and validates
a project's config.yaml file. Config composes typed sub-configuration
objects for each section:

    display     : Figure generation parameters and base style overrides
                  (DisplayConfig).
    init        : Shared code environment for all other sections
                  (InitConfig).
    targets     : Annotation target definitions and metadata
                  (TargetsConfig).
    annotations : Contour/boundary/point specifications
                  (AnnotationsConfig).
    figures     : Figure-generating functions (FiguresConfig).
    viewer      : Optional 3D cortex viewer geometry and overlays
                  (ViewerConfig).
"""

# Imports ----------------------------------------------------------------------

import yaml

from ._display     import DisplayConfig
from ._init        import InitConfig
from ._targets     import TargetsConfig
from ._annotations import AnnotationsConfig
from ._figures     import FiguresConfig
from ._viewer      import ViewerConfig

# Config Object ----------------------------------------------------------------

class Config:
    """Top-level configuration parsed from a project's config.yaml.

    Loads the YAML file and delegates each section to a typed
    sub-configuration class. Unrecognized top-level keys are not
    parsed but remain accessible via :attr:`yaml`.

    Parameters
    ----------
    config_path : str, optional
        Filesystem path to the config.yaml file (default
        ``"/config/config.yaml"``).

    Attributes
    ----------
    config_path : str
        The path that was loaded.

    yaml : dict
        The raw parsed YAML mapping (read-only reference).

    display : DisplayConfig
        Figure generation parameters and base style overrides.

    init : InitConfig
        Shared code environment for all other sections.

    targets : TargetsConfig
        Annotation target definitions and metadata.

    annotations : AnnotationsConfig
        Contour, boundary, and point specifications.

    figures : FiguresConfig
        Compiled figure-generating functions.
        
    viewer : ViewerConfig
        3D cortex viewer geometry and overlays. Empty dict if the
        ``viewer`` section is omitted (2D-only mode).
    """
    
    __slots__ = (
        "config_path", "yaml", "display", "init", "targets", "annotations", 
        "figures", "viewer" 
    )
    
    def __init__(self, config_path = "/config/config.yaml"):
        # Load the configuration YAML file.
        self.config_path = config_path
        with open(config_path, "rt") as f:
            self.yaml = yaml.safe_load(f)

        # Parse the display section (optional).
        self.display = DisplayConfig(self.yaml.get("display", None))

        # Parse the init section (optional).
        self.init = InitConfig(self.yaml.get("init", None))

        # Parse the targets section.
        self.targets = TargetsConfig(
            self.yaml.get("targets", None), 
            self.init
        )

        # Parse the annotations section.
        self.annotations = AnnotationsConfig(
            self.yaml.get("annotations", None), self.init)

        # Parse the figures section.
        self.figures = FiguresConfig(
            self.yaml.get("figures", None),
            self.annotations.figure_names, 
            self.init
        )

        # Parse the viewer section (optional).
        self.viewer = ViewerConfig(
            self.yaml.get("viewer", None),
            self.annotations.figure_names, 
            self.init
        )

