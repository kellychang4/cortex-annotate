# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_annotate/_display.py

"""Display subpanel for cortex-annotate.

Provides the ``DisplaySection`` widget, which exposes two controls that
affect the overall tool layout:

    figure_size_slider : pixel size of one figure tile in the annotation
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
    figure_size_slider : ipywidgets.IntSlider
        Controls the pixel size of one figure tile.

    layout_toggle : ipywidgets.ToggleButton
        ``True`` for horizontal layout, ``False`` for vertical.
    """

    # Shared widget layout.
    _WIDGET_LAYOUT = { "width": "94%", "margin": "1% 3% 1% 3%" }

    __slots__ = ( "prefs", "figure_size_slider", "viewer_size_slider", "layout_toggle" )

    def __init__(self, prefs, has_viewer):
        # Store preferences for later use.
        self.prefs = prefs

        # Initialize canvas pixel slider.
        self.figure_size_slider = ipw.IntSlider(
            value             = prefs.get_display("figure_size"),
            min               = 128,
            max               = 1280,
            step              = 1,
            description       = "Canvas Size:",
            readout           = False,
            continuous_update = False,
            layout            = self._WIDGET_LAYOUT,
        )

        # Initialize viewer pixel slider.
        self.viewer_size_slider = ipw.IntSlider(
            value             = prefs.get_display("viewer_size"),
            min               = 128,
            max               = 1280,
            step              = 1,
            description       = "Viewer Size:",
            readout           = False,
            continuous_update = False,
            layout            = self._WIDGET_LAYOUT,
        )

        # Initialize layout toggle.
        toggle_value = prefs.get_display("layout") == "horizontal"
        toggle_str = "Horizontal Layout" if toggle_value else "Vertical Layout"
        print(f"Initial layout toggle value: {toggle_value} ({toggle_str})")
        self.layout_toggle = ipw.ToggleButton(
            value       = toggle_value,
            description = toggle_str,
            tooltip     = ( "Toggle between horizontal and vertical layout "
                            "of the control and figure panels." ),
            layout      = self._WIDGET_LAYOUT,
        )
        self.layout_toggle.add_class("annotate-layout-toggle")

        # Assemble children.
        super().__init__(
            children = [
                make_section_title("Display Options"),
                self.figure_size_slider,
                self.viewer_size_slider if has_viewer else None, 
                self.layout_toggle,
            ],
            layout = { "margin": "0% 0% 3% 0%" },
        )

        # Wire internal observers.
        self.layout_toggle.observe(self._on_layout_toggle, names = "value")

    # Lock / Unlock ------------------------------------------------------------

    def lock(self):
        """Disable all display controls."""
        self.figure_size_slider.disabled = True
        self.viewer_size_slider.disabled = True
        self.layout_toggle.disabled      = True


    def unlock(self):
        """Enable all display controls."""
        self.figure_size_slider.disabled = False
        self.viewer_size_slider.disabled = False
        self.layout_toggle.disabled      = False

    # Observer Registration ----------------------------------------------------

    def observe_figure_size(self, fn):
        """Register a callback for canvas size changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(change)`` where *change* is the ipywidgets
            change object with ``change.new`` being the new pixel size.
        """
        self.figure_size_slider.observe(fn, names = "value")


    def observe_viewer_size(self, fn):
        """Register a callback for viewer size changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(change)`` where *change* is the ipywidgets
            change object with ``change.new`` being the new pixel size.
        """
        self.viewer_size_slider.observe(fn, names = "value")


    def observe_layout(self, fn):
        """Register a callback for layout toggle changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(change)`` where ``change.new`` is ``True``
            for horizontal layout, ``False`` for vertical.
        """
        self.layout_toggle.observe(fn, names = "value")

    # Internal Observers -------------------------------------------------------

    def _on_layout_toggle(self, change):
        """Update the layout setting in the preferences manager and figure panel.

        Parameters
        ----------
        change : traitlets.Bunch
            The ipywidgets change object.
        """
        if self.layout_toggle.value:
            self.layout_toggle.description = "Horizontal Layout"
        else:
            self.layout_toggle.description = "Vertical Layout"