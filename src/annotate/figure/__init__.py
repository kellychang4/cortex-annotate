# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/__init__.py

"""Figure panel composition for the cortex-annotate toolkit.

This subpackage provides the annotation editing model and the visual
rendering panels (2D canvas + optional 3D cortex viewer) used to
display and interact with annotations.

``AnnotationEditor`` is a pure data model that owns annotation state
(points, cursors, fixed heads/tails, editable indices) and exposes
push/pop/toggle operations.  It never creates widgets and never
displays anything.

``FigurePanel`` is the facade widget that composes the renderers
(``CanvasPanel``, ``CortexViewerPanel``) and a ``MessageOverlay``.
The orchestrator interacts only with ``FigurePanel``; renderers are
private implementation details.

Modules
-------
_editor
    ``AnnotationEditor`` — pure annotation manipulation model.
    Owns target, active annotation, coordinate arrays, cursor,
    editable indices, fixed-point calculation, and dependency
    recalculation.

_canvas
    ``CanvasPanel`` — 2D ipycanvas renderer.  Multi-layer canvas
    for grid images, background annotations, dependent-background
    annotations, and the active annotation overlay.

_viewer
    ``CortexViewerPanel`` — 3D k3d cortex mesh renderer.  Handles
    mesh geometry, overlays, morph interpolation, and 3D annotation
    display.  Only instantiated when ``config.viewer`` is non-empty.

_figure
    ``FigurePanel`` — facade widget.  Composes canvas + viewer,
    layout management, ``MessageOverlay``, and the public API
    for the orchestrator (redraw, resize, loading context, messages).
"""

# ------------------------------------------------------------------------------

from ._editor import AnnotationEditor
from ._figure import FigurePanel

__all__ = ( "AnnotationEditor", "FigurePanel" )