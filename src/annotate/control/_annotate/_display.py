# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_annotate/_display.py

"""Display subpanel for cortex-annotate.

Provides the ``DisplaySection`` widget, which exposes two controls that
affect the overall tool layout:

    image_pixel_slider : pixel size of one figure tile in the annotation
                         grid (drives canvas and image sizing).

    layout_toggle      : switches between horizontal and vertical
                         arrangement of the control and figure panels.
"""

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

from ..._widgets import make_section_title

# The Display Subpanel ---------------------------------------------------------

class DisplaySection(ipw.VBox):
    """Image pixel size and layout controls.

    Parameters
    ----------
    prefs : PrefsManager
        Preferences manager, used to read initial display values.

    Attributes
    ----------
    image_pixel_slider : ipywidgets.IntSlider
        Controls the pixel size of one figure tile.

    layout_toggle : ipywidgets.ToggleButton
        ``True`` for horizontal layout, ``False`` for vertical.
    """

    # Shared widget layout.
    _WIDGET_LAYOUT = { "width": "94%", "margin": "1% 3% 1% 3%" }

    __slots__ = ( "image_pixel_slider", "layout_toggle" )

    def __init__(self, prefs):
        # Read initial values from preferences.
        initial_pixel  = prefs.get_display("image_pixel")
        initial_layout = prefs.get_display("layout")

        # Initialize figure pixel slider.
        self.image_pixel_slider = ipw.IntSlider(
            value             = initial_pixel,
            min               = 128,
            max               = 1280,
            step              = 1,
            description       = "Figure Size:",
            readout           = False,
            continuous_update = False,
            layout            = self._WIDGET_LAYOUT,
        )

        # Initialize layout toggle.
        self.layout_toggle = ipw.ToggleButton(
            value       = initial_layout == "horizontal",
            description = "Horizontal Layout",
            tooltip     = ( "Toggle between horizontal and vertical layout "
                            "of the control and figure panels." ),
            layout      = self._WIDGET_LAYOUT,
        )
        self.layout_toggle.add_class("annotate-layout-toggle")

        # Assemble children.
        super().__init__(
            children = [
                make_section_title("Display Options"),
                self.image_pixel_slider,
                self.layout_toggle,
            ],
            layout = { "margin": "0% 0% 3% 0%" },
        )

    # Observer Registration ----------------------------------------------------

    def observe_image_pixel(self, fn):
        """Register a callback for image pixel size changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(change)`` where *change* is the ipywidgets
            change object with ``change.new`` being the new pixel size.
        """
        self.image_pixel_slider.observe(fn, names = "value")


    def observe_layout(self, fn):
        """Register a callback for layout toggle changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(change)`` where ``change.new`` is ``True``
            for horizontal layout, ``False`` for vertical.
        """
        self.layout_toggle.observe(fn, names = "value")

    # Lock / Unlock ------------------------------------------------------------

    def lock(self):
        """Disable all display controls."""
        self.image_pixel_slider.disabled = True
        self.layout_toggle.disabled      = True


    def unlock(self):
        """Enable all display controls."""
        self.image_pixel_slider.disabled = False
        self.layout_toggle.disabled      = False