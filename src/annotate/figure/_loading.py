# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_loading.py

"""Loading screen overlay for cortex-annotate.
 
``LoadingOverlay`` displays a semi-transparent loading message over the
entire ``FigurePanel`` (canvas + viewer) during long-running operations
such as cache generation, target switching, or initial startup.
 
Uses ``pointer-events: auto`` to block all mouse interaction with the
underlying widgets, preventing canvas clicks or k3d viewer rotation
while geometry or images are being swapped.
 
``LoadingContext`` is a reference-counted context manager that wraps a
``LoadingOverlay``.  Nested ``with`` blocks increment and decrement an
internal counter; the overlay is shown on the first entry and hidden
only when the outermost block exits.
 
Stacking order is managed by the CSS Grid overlap in ``FigurePanel``.
``LoadingOverlay`` is added to the DOM before ``MessageOverlay`` so
that error messages can appear on top of an active loading screen.
"""

# Imports ----------------------------------------------------------------------

from ._overlay import Overlay

# Loading Overlay --------------------------------------------------------------

class LoadingOverlay(Overlay):
    """Semi-transparent overlay for loading screens.

    Extends ``Overlay`` with a reference count (``_count``) used by
    ``LoadingContext`` to support nested loading operations.  Direct
    calls to ``show()`` / ``hide()`` are possible but discouraged —
    use the context manager instead.

    Parameters
    ----------
    z_index : int, optional
        CSS ``z-index`` for the overlay.  Defaults to ``1``.

    font_size : int, optional
        Font size in pixels.  Defaults to ``32`` (larger than
        ``MessageOverlay`` to distinguish loading screens visually).

    Attributes
    ----------
    _count : int
        Reference count of active loading contexts.  The overlay is
        visible whenever ``_count > 0``.
    """

    __slots__ = ( "_count", )

    def __init__(self, z_index = 1, font_size = 32):
        """Initialize the loading overlay."""
        self._count = 0 # initalize count for loading context
        super().__init__(z_index = z_index, font_size = font_size)


# Loading Context --------------------------------------------------------------

class LoadingContext:
    """Reference-counted context manager for the loading overlay.

    Each ``__enter__`` increments the reference count and shows the
    overlay (on the first entry).  Each ``__exit__`` decrements the
    count and hides the overlay only when it reaches zero.  This
    allows nested loading operations (e.g. cache generation inside a
    target switch) to share a single visible overlay.

    Parameters
    ----------
    overlay : LoadingOverlay
        The loading overlay widget to manage.

    message : str, optional
        The loading message to display.  Defaults to ``"Loading..."``.

    Examples
    --------
    >>> ctx = figure_panel.loading_context
    >>> with ctx:              # count 1 → shows "Loading..."
    ...     with ctx:          # count 2 → still showing
    ...         pass           # inner exit → count 1
    ...                        # outer exit → count 0 → hides
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