# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_overlay.py
 
"""Message overlay widget for cortex-annotate.
 
``Overlay`` is a single ``ipw.HTML`` widget that covers the
entire ``FigurePanel`` area to display transient messages, persistent
messages, or loading screens.  Only one message is active at a time;
the orchestrator's ``self.locked`` flag prevents conflicting states.
 
Three usage modes, all through the same class:
 
**Timed messages** — ``show(message, timeout=3.0)`` displays a
message that auto-hides after the timeout.  Used for transient
errors like dependency-blocked deletion.
 
**Persistent messages** — ``show(message)`` with no timeout.  The
message remains until ``hide()`` is called explicitly.  Used for
errors that require a user action to resolve.
 
**Loading screens** — managed via the ``LoadingContext`` context
manager, which provides reference counting for nested loading
operations.  The overlay shows when the first context enters and
hides when the outermost context exits.
 
Visual styling is applied entirely through CSS classes targeting
the widget's ``.widget-html-content`` container.  ``show()`` sets
``self.value`` to the message text directly — no inner ``<div>``
wrapper.
 
Stacking is handled by CSS Grid in ``FigurePanel``.  The overlay
occupies the same grid cell as the figure content.  DOM order and
``z-index`` control visual stacking (overlay on top).
 
The ``annotate-overlay`` class applies ``pointer-events: auto`` to
block all mouse interaction with the underlying canvas and viewer
widgets while visible.
"""
 
# Imports ----------------------------------------------------------------------
 
import threading
import ipywidgets as ipw
 
# Message Overlay --------------------------------------------------------------
 
class Overlay(ipw.HTML):
    """Semi-transparent overlay for messages and loading screens.

    Arguments
    ---------
    css_classes : list of str, optional
        Additional CSS classes to add to the widget.  Defaults to
        ``[ "annotate-figure-item" ]``.
 
    Attributes
    ----------
    _count : int
        Reference count of active loading contexts.  The overlay is
        visible whenever ``_count > 0``.
    
    _timer : threading.Timer or None
        Active auto-hide timer, or ``None`` if no timer is pending.
    """
 
    __slots__ = ( "_timer", )
 
    def __init__(self, css_classes = []):
        """Initialize the overlay (hidden)."""
        # Initialize timer (if needed).
        self._timer = None

        # Initialize empty HTML widget.
        super().__init__(
            value  = "",
            layout = ipw.Layout(display = "none"),
        )

        # Prepare CSS classes if a string was given.
        if isinstance(css_classes, str):
            css_classes = [ css_classes ]
        
        # Add CSS classes.
        for css_class in css_classes:
            self.add_class(css_class)

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
        self.value = message
        self.layout.display = "block"

        # Start timer, if timeout is given.
        if timeout is not None:
            self._timer = threading.Timer(timeout, self.hide)
            self._timer.start()
 

    def hide(self):
        """Hide the overlay and cancel any pending auto-hide timer."""
        # Clear any existing timer.
        self._cancel_timer()

        # Clear the message.
        self.value = ""
        self.layout.display = "none"
 

    # Timer Method -------------------------------------------------------------

    def _cancel_timer(self):
        """Cancel the pending timer, if there is one."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
 

# Loading Context --------------------------------------------------------------

class LoadingContext:
    """Reference-counted context manager for loading screens.
 
    Wraps a ``Overlay`` to provide nested-safe loading
    display.  The overlay shows when the first context enters and
    hides when the outermost context exits.
 
    Parameters
    ----------
    overlay : Overlay
        The overlay widget to show/hide.
 
    message : str, optional
        The loading message to display.  Defaults to
        ``"Loading..."``.
    
    Examples
    --------
    >>> ctx = figure_panel.loading_context
    >>> with ctx:              # count 1 → shows "Loading..."
    ...     with ctx:          # count 2 → still showing
    ...         pass           # inner exit → count 1
    ...                        # outer exit → count 0 → hides
    """
 
    __slots__ = ( "_count", "_overlay", "_message" )
 
    def __init__(self, overlay, message = "Loading..."):
        # Initialize counter.
        self._count = 0

        # Store overlay and message.
        self._overlay = overlay
        self._message = message
 

    def __enter__(self):
        self._count += 1
        if self._count == 1:
            self._overlay.show(self._message)
        return self
 

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._count -= 1
        if self._count <= 0:
            self._count = 0
            self._overlay.hide()
        return False
 