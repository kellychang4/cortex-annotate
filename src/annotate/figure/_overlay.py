# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_overlay.py

"""Base overlay widget for cortex-annotate.
 
``Overlay`` is a lightweight ``ipw.HTML`` subclass that renders a
semi-transparent ``<div>`` over the ``FigurePanel`` when shown, and
collapses to ``display: none`` when hidden.
 
Stacking is achieved via the CSS Grid overlap technique used by
``FigurePanel``: all children occupy the same grid cell, and later
DOM children naturally sit on top of earlier ones.  ``z-index`` values
are set as a secondary guarantee.
 
Subclasses (``MessageOverlay``, ``LoadingOverlay``) extend ``show()``
and ``hide()`` to add timeout and reference-counting behaviour
respectively.
"""

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

# CSS Template -----------------------------------------------------------------

_CSS_TEMPLATE = (
    "width: 100%; "
    "height: 100%; "
    "z-index: {z_index}; "
    "display: flex; "
    "align-items: center; "
    "justify-content: center; "
    "background: rgba(255, 255, 255, 0.85); "
    "font: {font_size}px HelveticaNeue, sans-serif; "
    "padding: 20px; "
    "text-align: center; "
    "pointer-events: auto;"
)

# Overlay Base Class -----------------------------------------------------------

class Overlay(ipw.HTML):
    """Semi-transparent HTML overlay base class.

    Renders a full-size ``<div>`` with ``pointer-events: auto`` to
    block mouse interaction with widgets underneath.  Hidden by default.

    Subclasses should call ``super().__init__(...)`` with the desired
    ``z_index`` and ``font_size`` to customise the inline CSS, then
    override ``show()`` / ``hide()`` to add behaviour (timers,
    reference counting, etc.).

    Parameters
    ----------
    z_index : int, optional
        CSS ``z-index`` for the overlay ``<div>``.  Higher values
        stack on top.  Defaults to ``10``.

    font_size : int, optional
        Font size in pixels for the overlay message text.
        Defaults to ``20``.

    Attributes
    ----------
    _css : str
        Compiled inline CSS string applied to the inner ``<div>``
        element.  Built once at construction from the template and
        the provided keyword arguments.
    """

    __slots__ = ( "_css", )

    def __init__(self, z_index = 10, font_size = 20):
        """Initialize the overlay (hidden by default)."""
        # Compile the inline CSS once at construction for efficiency.
        self._css = _CSS_TEMPLATE.format(
            z_index   = z_index,
            font_size = font_size,
        )

        # Initialize the HTML widget with an empty value and hide display.
        super().__init__(
            value  = "",
            layout = ipw.Layout(display = "none"),
        )

    # Show / Hide --------------------------------------------------------------

    def show(self, message):
        """Display a message over the figure area.

        Parameters
        ----------
        message : str
            The message text to display.  May contain HTML markup.
        """
        self.value = f"<div style='{self._css}'>{message}</div>"
        self.layout.display = "block"


    def hide(self):
        """Hide the overlay."""
        self.value = ""
        self.layout.display = "none"