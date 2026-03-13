# -*- coding: utf-8 -*-
################################################################################
# annotate/_widgets.py
# 
# Widget helper functions for constructing control panel UI elements.
# These are small functions that return styled ipywidgets for use in 
# ControlPanel and its subpanels.

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw
import matplotlib as mpl

# Widget Helper Functions ------------------------------------------------------

def make_section_title(title, margin = "0% 3% 0% 3%"):
    """Returns an HTML widget containing the given title formatted as a section title."""
    return ipw.HTML(f"<b style=\"margin: {margin};\">{title}:</b>")


def make_hline(class_name = "annotate-control-panel-hline"):
    """Returns an HTML widget containing a horizontal line."""
    return ipw.HTML(f"""<div class="{class_name}"></div>""")


def darken_color(color, amount = 0.10):
    """Darkens a color by the given fraction and returns an integer RGB tuple.
    
    Accepts any matplotlib-compatible color input (hex string, named color, 
    RGB tuple). Returns a tuple of integers in 0-255 range, suitable for 
    use in CSS `rgb(...)` values.
    """
    color = mpl.colors.to_rgb(color) # convert to RGB tuple if hex string
    color = [ max(0.0, x * (1 - amount)) for x in color ] # darken the color by the given amount
    return tuple([ int(x * 255) for x in color ]) # convert back to 0-255 range and return as list
