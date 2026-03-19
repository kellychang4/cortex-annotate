# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_loading.py
 
#DOCSTRING

# Imports ----------------------------------------------------------------------
 
import ipywidgets as ipw

# Loading Overlay --------------------------------------------------------------
 
class LoadingOverlay(ipw.HTML):
    """Semi-transparent HTML overlay for loading screens.
 
    Positioned absolutely over the entire ``FigurePanel`` (canvas +
    viewer) via CSS, on top of ``MessageOverlay`` so that the loading 
    screens always take visual priority.
 
    Uses ``pointer-events: auto`` to block all mouse interaction with
    the underlying widgets, preventing the user from rotating the k3d
    viewer or clicking the canvas while geometry or images are being
    swapped.
 
    Managed exclusively through the ``LoadingContext`` context
    manager, which provides reference counting for nested loading
    operations.  Direct calls to ``show()`` / ``hide()`` are possible
    but discouraged — use the context manager instead.
 
    This widget requires its parent container to have
    ``position: relative`` in its layout so that the absolute
    positioning anchors correctly.
 
    Attributes
    ----------
    _count : int
        Reference count of active loading contexts.  The overlay is
        visible whenever ``_count > 0``.
    """
 
    _CSS = (
        # "position: absolute; top: 0; left: 0; "
        "width: 100%; height: 100%; z-index: 20; "
        "display: flex; align-items: center; justify-content: center; "
        "background: rgba(255, 255, 255, 0.85); "
        "font: 32px HelveticaNeue, sans-serif; "
        "padding: 20px; text-align: center; "
        "pointer-events: auto;"
    )
 
    def __init__(self):
        """Initialize the loading overlay (hidden)."""
        self._count = 0
        super().__init__(
            value  = "",
            layout = ipw.Layout(display = "none"),
        )
 

    def show(self, message = "Loading..."):
        """Display a loading message over the figure area.
 
        Parameters
        ----------
        message : str, optional
            The loading message to display.  Defaults to
            ``"Loading..."``.
        """
        self.value = f"<div style='{self._CSS}'>{message}</div>"
        self.layout.display = "block"
 

    def hide(self):
        """Hide the loading overlay."""
        self.value = ""
        self.layout.display = "none"
 

# Loading Context --------------------------------------------------------------

class LoadingContext:
    """Context manager for the loading overlay.
 
    Parameters
    ----------
    overlay : LoadingOverlay
        The loading overlay widget to show/hide.

    message : str, optional
        The loading message to display.  Defaults to ``"Loading..."``.
 
    Examples
    --------
    >>> loading_context = figure_panel.loading_context
    >>> with loading_context:          # shows "Loading..."
    ...     with loading_context:      # ref count 2, still showing
    ...         pass                   # inner exit, ref count 1
    ...                                # outer exit, ref count 0 → hides
    """
 
    __slots__ = ( "_overlay", "_message" )
 
    def __init__(self, overlay, message = "Loading..."):
        self._overlay = overlay
        self._message = message
 

    def __enter__(self):
        self._overlay._count += 1
        if self._overlay._count == 1:
            self._overlay.show(self._message)
        return self
 

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._overlay._count -= 1
        if self._overlay._count <= 0:
            self._overlay._count = 0
            self._overlay.hide()
        return False