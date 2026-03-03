# -*- coding: utf-8 -*-
################################################################################
# annotate/_figure.py

"""
Implementation code for the Figure Panel.
"""

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

from ._canvas import CanvasPanel
from ._viewer import CortexViewerPanel

# The Figure Panel State ------------------------------------------------------

class FigurePanelState:
    """Panel-specific state for the figure panel.
    
    This class manages the data that is specific to the figure panel but not 
    relevant to the 2D canvas or the broader annotation tool. It consumes 
    cortex geometry and overlay data from `AnnotationState.cortex_data()` and
    converts flatmap annotations into 3D surface coordinates.
    """

    # Point type constants (used for annotation rendering)
    POINT_FIXED  = 2  # fixed head/tail point
    POINT_USER   = 1  # user-placed point
    POINT_INTERP = 0  # interpolated point (between user/fixed points)


    def __init__(self, state):
        """Initialize the figure panel state.
        
        Parameters
        ----------
        annotation_state : AnnotationState
            The shared annotation state that provides cortex data, annotation
            coordinates, annotation config, and style preferences.
        """
        # Store the state.
        self.state = state

        # Cortex viewer-specific (only) display style options.
        self.style = {
            "inflation_percent" : 100,
            "overlay"           : "curvature",
            "overlay_alpha"     : 1.0, 
            "point_size"        : 1.5, 
            "line_width"        : 0.25,
            "line_interp"       : 10,
        }

        # Current viewer data — populated by update methods, read by the panel.
        self.target_id  = None  # current target id tuple
        self.annotation = None  # current active annotation name 

        # Cortex geometry (set by update_cortex)
        self.faces       = None  # (3, N_faces) face indices
        self.coordinates = None  # (3, N_vertices) blended coordinates
        self.curvature   = None  # (N_vertices, 3) curvature RGB colors

        # Overlay data (set by update_overlay)
        self.overlay = None  # (N_vertices, 3) overlay RGB colors, or None

        # Surface annotations (set by update_surface_annotations)
        # Dict of annotation_name -> { "addresses", "coordinates", "point_types" }
        self.surface_annotations = {}


# Figure Panel Class -----------------------------------------------------------

class FigurePanel(ipw.Box):

    # Define the horizontal and vertical layouts for the figure panel. 
    _HORIZONTAL_LAYOUT = ipw.Layout(
        display     = "flex", 
        flex_flow   = "row", 
        align_items = "stretch",
        overflow    = "hidden",
        border      = "1px solid deeppink",
    )

    _VERTICAL_LAYOUT = ipw.Layout(
        display     = "flex", 
        flex_flow   = "column", 
        align_items = "stretch",
        overflow    = "hidden",
        border      = "1px solid deeppink",
    )

    def __init__(self, state, width = 512, height = 512):
        """Initialize the figure panel."""
        #TODO!!!!
        # Store the state. (self.state = figure state, not annotation state)
        # self.state = FigurePanelState(state)

        # Make the canvas panel.
        self.canvas_panel = CanvasPanel(state)

        # Make the cortex viewer panel. 
        self.viewer_panel = CortexViewerPanel(
            state, width = width, height = height
        )

        # Create the Box (HBox/VBox) figure area.
        super().__init__(
            children = [ self.canvas_panel, self.viewer_panel ],
            layout   = self._HORIZONTAL_LAYOUT
        )

