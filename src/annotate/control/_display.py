# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_display.py

"""Display settings panel for cortex-annotate.

The DisplayPanel provides controls for general display preferences that
affect both the 2D canvas and 3D viewer: figure size, layout orientation,
and (when a viewer is configured) viewer-specific controls like morph
percentage, overlay selection, and overlay opacity.
"""

# Imports ----------------------------------------------------------------------

import os.path as op
import ipywidgets as ipw
from functools import partial

from .._widgets import make_section_title, make_hline, darken_color

# The Display Subpanel -------------------------------------------------------

class DisplayPanel(ipw.VBox):
    """Tab panel for display preferences (size, layout, viewer controls).

    Parameters
    ----------
    prefs : PrefsManager
        Preferences manager, used to read/write display and viewer style prefs.
    has_viewer : bool
        Whether the viewer is enabled. If False, viewer controls are hidden.

    Widgets
    -------
    figure_size_slider : ipw.IntSlider
        Controls the figure size in pixels (250 - 1280).
    layout_toggle : ipw.ToggleButton
        Switches between horizontal and vertical layout.
    morph_slider : ipw.IntSlider or None
        Controls morph percentage (0 - 100). Only present if has_viewer.
    overlay_dropdown : ipw.Dropdown or None
        Selects overlay type. Only present if has_viewer.
    overlay_alpha_slider : ipw.FloatSlider or None
        Controls overlay opacity (0.0 - 1.0). Only present if has_viewer.
    """

    def __init__(self):
        pass

    def observe_figure_size(self, fn):
        """Registers the argument to be called when the figure size changes.

        `control_panel.observe_figure_size(fn)` is equivalent to
        `control_panel.figure_size_slider.observe(fn, names="value")`.
        """
        self.figure_size_slider.observe(fn, names = "value")

    
    def observe_layout(self, fn):
        """Registers the argument to be called when the layout toggle button is toggled.

        The function is called with a single argument, which is the layout toggle
        button instance.
        """
        self.layout_toggle.observe(fn, names = "value")