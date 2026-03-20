# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_annotate/_legend.py

"""Annotation legend image subpanel.
 
Provides the ``LegendSection`` widget, which displays a reference image
for the currently selected annotation and hemisphere.  Legend images
are PNG files stored in a sibling ``annotation-legends/`` directory,
organised by hemisphere subdirectory.
"""
 
# Imports ----------------------------------------------------------------------
 
import os.path as op
import ipywidgets as ipw
 
from ..._widgets import make_section_title

# The Legend Subpanel ----------------------------------------------------------

class LegendSection(ipw.VBox):
    """Annotation legend image display.
 
    Shows a PNG legend for the active annotation, selected by
    hemisphere.  If no matching image is found, a placeholder
    ``empty.png`` is displayed.
 
    Parameters
    ----------
    config : Config
        The configuration object. Used to read target and annotation
        configuration during initialisation.

    hemisphere_key : str, optional
        The concrete target key name used to look up the hemisphere
        value from a target tuple.  Defaults to ``"Hemisphere"``.
 
    Attributes
    ----------
    legend_dir : str
        Absolute path to the ``legends/`` directory.

    hemisphere_index : int
        Positional index of *hemisphere_key* within the concrete keys.

    image_widget : ipywidgets.Image
        The widget displaying the legend PNG.
    """

    # legend directory
    _LEGEND_DIR = op.join(op.dirname(__file__), "legend")

    __slots__ = ( "legend_dir", "hemisphere_index", "image_widget" )

    def __init__(self, config, hemisphere_key = "Hemisphere"): 
        # Set up the path to the annotation legend images.
        self.legend_dir = self._LEGEND_DIR
 
        # Resolve the hemisphere index from the concrete target keys.
        concrete_keys = config.targets.concrete_keys
        try: self.hemisphere_index = concrete_keys.index(hemisphere_key)
        except Exception:
            raise ValueError(
                f"hemisphere_key {hemisphere_key!r} not found in "
                f"concrete_keys: {concrete_keys}"
            )
 
        # Create the image widget.
        self.image_widget = ipw.Image(
            format = "png",
            layout = { "margin": "0% 3% 0% 3%", "width": "94%" }
        )
 
        # Initialize the VBox.
        super().__init__(
            children = [
                make_section_title("Annotation Legend"),
                self.image_widget,
            ],
            layout = { "margin": "0% 0% 3% 0%" }
        )
 
        # Update the legend with the initial image.
        # target_keys[0] is a tuple of default values (one per concrete
        # key) representing the first valid target combination.
        target_id  = config.targets.target_keys[0]
        annotation = config.annotations.names[0]
        self.update(target_id, annotation)
    
    # Update Method ------------------------------------------------------------

    def update(self, target_id, annotation):
        """Update the legend image to match a target and annotation.
 
        Parameters
        ----------
        target_id : tuple[str, ...]
            The current target selection tuple (one value per concrete
            key).  The hemisphere value is extracted using
            ``self.hemisphere_index``.

        annotation : str
            The annotation name; corresponds to a PNG filename in the
            hemisphere's legend subdirectory.
        """
        # Construct the path to the legend image and update the widget.
        hemisphere = target_id[self.hemisphere_index]
        image_path = op.join(self.legend_dir, hemisphere, f"{annotation}.png")

        # If the image file does not exist, use the empty placeholder.
        if not op.isfile(image_path): 
            image_path = op.join(self.legend_dir, "empty.png")
        
        # Read the image file and update the widget value.
        with open(image_path, "rb") as f:
            self.image_widget.value = f.read()
