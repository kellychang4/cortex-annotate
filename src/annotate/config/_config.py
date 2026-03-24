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
        "display", "init", "targets", "annotations", "figures", "viewer" 
    )
    
    def __init__(self, config_path = "/config/config.yaml"):
        # Read the configuration YAML file.
        with open(config_path, "rt") as f:
            config_yaml = yaml.safe_load(f)

        # Parse the display section (optional).
        display_yaml = config_yaml.get("display", None)
        self.display = DisplayConfig(display_yaml)

        # Parse the init section (optional).
        init_yaml = config_yaml.get("init", None)
        self.init = InitConfig(init_yaml)

        # Parse the targets section.
        targets_yaml = config_yaml.get("targets", None)
        self.targets = TargetsConfig(targets_yaml, self.init)

        # Parse the annotations section.
        annotations_yaml   = config_yaml.get("annotations", None)
        self.annotations   = AnnotationsConfig(annotations_yaml, self.init)
        annot_figure_names = self.annotations.figure_names

        # Parse the figures section.
        figures_yaml = config_yaml.get("figures", None)
        self.figures = FiguresConfig(figures_yaml, annot_figure_names, self.init)

        # Parse the viewer section (optional).
        viewer_yaml = config_yaml.get("viewer", None)
        self.viewer = ViewerConfig(viewer_yaml, annot_figure_names, self.init)