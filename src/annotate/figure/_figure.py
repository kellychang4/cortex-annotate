# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_figure.py

"""
Implementation code for the Figure Panel.
"""

# Imports ----------------------------------------------------------------------

import threading
import numpy as np
import ipywidgets as ipw
from functools import partial

from ._canvas import CanvasPanel
from ._viewer import CortexViewerPanel

# Figure Panel Class -----------------------------------------------------------

class FigurePanel(ipw.Box):
    """Container widget holding the 2D canvas and 3D cortex viewer."""

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

    def __init__(self, annot_state, width = 512, height = 512):
        """Initialize the figure panel."""
        # Store the annotation state.
        self.annot_state = annot_state

        # Create the figure panels state (NOT annotation state)
        self.figure_state = FigurePanelState(
            annot_state = self.annot_state
        )
        print("Inside FigurePanel...")

        # Make the canvas panel.
        self.canvas_panel = CanvasPanel(
            figure_state = self.figure_state,
            figure_size  = annot_state.preferences["figure_size"]
        )

        # Make the cortex viewer panel. 
        self.viewer_panel = CortexViewerPanel(
            figure_state = self.figure_state,
            width = width, 
            height = height
        )

        # Create the Box (HBox/VBox) figure area.
        super().__init__(
            children = [ self.canvas_panel, self.viewer_panel ],
            layout   = self._HORIZONTAL_LAYOUT
        )

        # Canvas-specific observers (mouse clicks and key presses)
        self.canvas_panel.observe_mouse(self.on_mouse_click)
        self.canvas_panel.observe_key(self.on_key_press)

    
    # Property Methods ------------------------------------------------------------

    @property
    def loading_context(self):
        """Expose the canvas loading context."""
        return self.canvas_panel.loading_context
    
    # Redraw Method ------------------------------------------------------------

    def redraw(
            self, clear = False, base = False, active = True, 
            background = False
        ):
        """Redraw both the canvas and viewer panels."""
        # Redraw the viewer panel.
        self.canvas_panel.redraw_canvas(
            image      = base, 
            active     = active,
            background = background
        )
        
        # Redraw the viewer panel.
        self.viewer_panel.redraw_viewer(
            clear      = clear, 
            cortex     = base, 
            active     = active,
            background = background
        )

    # Mouse Event Handler Methods ----------------------------------------------

    def on_mouse_click(self, points):
        """Handle a mouse click on the canvas."""
        # If the figure is locked, we do not allow events.
        if self.annot_state.locked: return

        # Push points (in figure coordinates) to the state. 
        fixed_deps = self.figure_state.push_point(points)
        
        # Update the viewer annotations (active + dependencies).
        self.figure_state.update_viewer_annotations(
            annotations = [ self.figure_state.active, *fixed_deps ])

        # Redraw canvas and viewer.
        self.redraw(active = True, background = len(fixed_deps) > 0)

    # Key Press Event Handler Methods ------------------------------------------

    def on_key_press(self, key, shift_down, ctrl_down, meta_down):
        """Handle a key press on the canvas."""
        # If the figure is locked, we do not allow events.
        if self.annot_state.locked: return

        # Handle the key press.
        fixed_deps = []
        key = key.lower()
        if key == "tab":
            # Toggle the cursor (active) position. 
            self.figure_state.toggle_cursor()            
        elif key == "backspace":
            # Delete current cursor (active) point.
            fixed_deps = self.figure_state.pop_point()

            # Update the viewer annotations (active + dependencies).
            self.figure_state.update_viewer_annotations(
                annotations = [ self.figure_state.active, *fixed_deps ])
        else: 
            # Unrecognized key press, can skip redrawing
            return 

        # Redraw canvas because of annotation change 
        self.redraw(active = True, background = len(fixed_deps) > 0) 

    # Canvas Resizing Method ---------------------------------------------------

    # def resize_canvas(self, new_figure_size = None):
    #     """Resize the canvas so that each grid cell has the given pixel size.

    #     Triggers a full redraw because resizing clears the canvas.
    #     """
    #     # If there is no new_figure_size give, we just use the current figure size.
    #     if new_figure_size is None:
    #         new_figure_size = self.figure_size

    #     # Update the figure size (pixels per grid cell).
    #     self.figure_size = np.array([new_figure_size, new_figure_size])

    #     # The canvas size is a product of the figure size and the grid shape.
    #     self.canvas_size = self.figure_size * np.array(self.state.grid_shape)
    #     canvas_width, canvas_height = self.canvas_size.astype(int)

    #     # Resize the multicanvas (this clears it).
    #     self.multicanvas.width         = canvas_width
    #     self.multicanvas.height        = canvas_height
    #     self.multicanvas.layout.width  = f"{canvas_width}px"
    #     self.multicanvas.layout.height = f"{canvas_height}px"

    #     # Redraw everything.
    #     self.redraw_canvas()

    # Internal Helpers ---------------------------------------------------------

    # def _increment_annotation_change(self):
    #     """Increments the annotation change traitlet after redraw triggers."""
    #     self.figure_state._annotation_change += 1        

    #TODO; these are temporary until i figure out a nicer way to do this...
    def write_message(self, message):
        """Writes a message to the figure panel."""
        self.canvas_panel.write_message(message)

    def clear_message(self):
        """Clears the message from the figure panel."""
        self.canvas_panel.clear_message()