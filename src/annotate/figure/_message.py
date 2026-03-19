# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_message.py
 
#DOCSTRING

# Imports ----------------------------------------------------------------------
 
import threading
import ipywidgets as ipw

# Message Overlay --------------------------------------------------------------
 
class MessageOverlay(ipw.HTML):
    """Semi-transparent HTML overlay for transient error messages.
 
    Positioned absolutely over the entire ``FigurePanel`` (canvas +
    viewer) via CSS.  Hidden by default.  Supports optional auto-hide
    after a configurable timeout.
 
    Uses ``pointer-events: auto`` to block all mouse interaction with
    the underlying widgets while an error message is displayed.  The
    overlay clears itself after the timeout, restoring interaction.
 
    This widget requires its parent container to have
    ``position: relative`` in its layout so that the absolute
    positioning anchors correctly.
 
    Attributes
    ----------
    _timer : threading.Timer or None
        Active auto-hide timer, or ``None`` if no timer is pending.
    """
 
    # pointer-events: auto — blocks clicks from reaching the
    # canvas/viewer underneath.  Error messages indicate a state where
    # user interaction should be paused (e.g. dependency-blocked
    # deletion).  The overlay clears itself after the timeout.
    _CSS = (
        "position: absolute; top: 0; left: 0; "
        "width: 100%; height: 100%; z-index: 10; "
        "display: flex; align-items: center; justify-content: center; "
        "background: rgba(255, 255, 255, 0.85); "
        "font: 20px HelveticaNeue, sans-serif; "
        "padding: 20px; text-align: center; "
        "pointer-events: auto;"
    )
 
    def __init__(self):
        """Initialize the message overlay (hidden)."""
        self._timer = None
        super().__init__(
            value = "",
            layout = ipw.Layout(display = "none"),
        )
 

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
        self._cancel_timer()
        self.value = f"<div style='{self._CSS}'>{message}</div>"
        self.layout.display = "block"
        if timeout is not None:
            self._timer = threading.Timer(timeout, self.hide)
            self._timer.start()
 

    def hide(self):
        """Hide the overlay and cancel any pending auto-hide timer."""
        self._cancel_timer()
        self.value = ""
        self.layout.display = "none"
 

    def _cancel_timer(self):
        """Cancel the pending auto-hide timer, if any."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

