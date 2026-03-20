# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_message.py
 
"""Transient error message overlay for cortex-annotate.
 
``MessageOverlay`` displays a semi-transparent message over the entire
``FigurePanel`` (canvas + viewer) and optionally auto-hides after a
configurable timeout.
 
Uses ``pointer-events: auto`` to block all mouse interaction with the
underlying widgets while visible.  The overlay clears itself after the
timeout, restoring interaction.
 
Stacking order is managed by the CSS Grid overlap in ``FigurePanel``.
``MessageOverlay`` is added to the DOM after ``LoadingOverlay`` so
that it renders on top, and its ``z-index`` (2) is set higher as a
secondary guarantee.
"""

# Imports ----------------------------------------------------------------------

import threading

from ._overlay import Overlay

# Message Overlay --------------------------------------------------------------

class MessageOverlay(Overlay):
    """Semi-transparent overlay for transient error messages.

    Extends ``Overlay`` with optional auto-hide via a ``threading.Timer``.
    When a timeout is provided to ``show()``, the overlay hides itself
    automatically after the specified duration.  Calling ``show()``
    again before the timer fires cancels the pending timer and starts
    a new one.

    Parameters
    ----------
    z_index : int, optional
        CSS ``z-index`` for the overlay.  Defaults to ``2``.

    font_size : int, optional
        Font size in pixels.  Defaults to ``20``.

    Attributes
    ----------
    _timer : threading.Timer or None
        Active auto-hide timer, or ``None`` if no timer is pending.
    """

    __slots__ = ( "_timer", )

    def __init__(self, z_index = 2, font_size = 20):
        """Initialize the message overlay (hidden)."""
        # Timer is initialized to None (no pending timer) and managed by show/hide.
        self._timer = None

        # Initialize the base Overlay with the specified z-index and font size.
        super().__init__(z_index = z_index, font_size = font_size)

    # Show / Hide --------------------------------------------------------------

    def show(self, message, timeout = None):
        """Display a message over the figure area.

        Parameters
        ----------
        message : str
            The message text to display.  May contain HTML markup.

        timeout : float or None, optional
            If given, auto-hide after this many seconds.  If ``None``
            (default), the message persists until ``hide()`` is called.
        """
        # Clear any existing timer.
        self._cancel_timer()
        
        # Show the message.
        super().show(message)

        # Start timer if timeout is given.
        if timeout is not None:
            self._timer = threading.Timer(timeout, self.hide)
            self._timer.start()


    def hide(self):
        """Hide the overlay and cancel any pending auto-hide timer."""
        # Clear any existing timer to prevent. 
        self._cancel_timer()

        # Remove the message.
        super().hide()

    # Timer --------------------------------------------------------------------

    def _cancel_timer(self):
        """Cancel the timer, if any."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None