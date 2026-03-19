# -*- coding: utf-8 -*-
################################################################################
# annotate/config/__init__.py

# -*- coding: utf-8 -*-
################################################################################
# annotate/config/__init__.py

"""YAML configuration parsing for the cortex-annotate toolkit.

Modules
-------
_config
    ``Config`` — top-level configuration compositor; loads and validates
    config.yaml, composes all typed sub-configurations.

_error
    ``ConfigError`` — exception raised for invalid or missing YAML values.

_init
    ``InitConfig`` — executes the user-provided ``init`` code block and
    exposes the resulting environment to downstream sections.

_targets
    ``TargetsConfig`` — lazy-dict of annotation targets parsed from the
    ``targets`` section.

_annotations
    ``AnnotationsConfig`` — annotation specifications (contours,
    boundaries, points) from the ``annotations`` section.

_figures
    ``FiguresConfig`` — compiled figure-generating functions from the
    ``figures`` section.

_display
    ``DisplayConfig`` — read-only display settings (``figsize``, ``dpi``,
    ``layout``, styles) from the optional ``display`` section.
    
_viewer
    ``ViewerConfig`` — 3D cortex viewer geometry, coordinate systems,
    overlays, and transforms from the optional ``viewer`` section.
"""

# ----------------------------------------------------------------------

from ._config import Config

__all__ = ( "Config", )