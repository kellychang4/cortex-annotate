# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_legend.py
#
# DOCSTRING
    
# Imports ----------------------------------------------------------------------

import os.path as op
import ipywidgets as ipw
from functools import partial

from .._widgets import make_section_title, make_hline, darken_color

# The Legend Subpanel ----------------------------------------------------------

class LegendPanel(ipw.VBox):
    """The subpanel of the control panel containing the legend controls."""

    __slots__ = ( "state", "image_dir", "hemisphere_index", "image_widget" )

    def __init__(self, state):
        # Store the state
        self.state = state

        # Set up the path to the annotation legend images.
        self.image_dir = op.join(op.dirname(__file__), "annotation-legends")

        # Store the hemisphere index for later use in legend updates.
        concrete_keys = state.config.targets.concrete_keys
        self.hemisphere_index = concrete_keys.index("Hemisphere")

        # Create the image widget
        self.image_widget = ipw.Image(
            format = "png",
            layout = { "margin": "0% 3% 0% 3%", "width": "94%" }
        )

        # Initialize the VBox
        super().__init__(
            children = [ 
                make_section_title("Annotation Legend"), 
                self.image_widget 
            ],
            layout = { "margin": "0% 0% 3% 0%" }
        )

        # Update the legend with the initial image
        target_id  = state.config.targets.target_keys[0]
        annotation = state.config.annotations.names[0]
        self.update(target_id, annotation)


    def _read_image(self, image_path):
        """Reads the image data from the given path."""
        # Read the image data and return it.
        with open(image_path, "rb") as f:
            image_data = f.read()
        return image_data
    

    def update(self, target_id, annotation):
        """Updates the legend image to the given legend name."""
        hemisphere = target_id[self.hemisphere_index]
        image_path = op.join(self.image_dir, hemisphere, f"{annotation}.png")
        if not op.isfile(image_path): # if the image does not exist, use empty
            image_path = op.join(self.image_dir, "empty.png")
        self.image_widget.value = self._read_image(image_path)
