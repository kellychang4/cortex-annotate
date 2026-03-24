# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_annotate/_annotate.py

"""DOCSTRING"""


# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

from ._selection import SelectionSection
from ._display   import DisplaySection
from ._legend    import LegendSection
from ._buttons   import ButtonSection
from ._info      import InfoSection
from ..._widgets import make_hline

# The Annotate Tab -------------------------------------------------------------

class AnnotateTab(ipw.VBox):
    """Composed selection tab wrapping the core subpanels.

    Groups ``SelectionSection``, ``DisplaySection``, ``LegendSection``,
    ``ButtonSection``, and ``InfoSection`` into a single tab.  Implements
    ``lock()`` / ``unlock()`` by propagating to each sub-panel that
    supports it.

    Parameters
    ----------
    selection : SelectionSection
    display : DisplaySection
    legend : LegendSection
    buttons : ButtonSection
    info : InfoSection
    """

    __slots__ = ( "_lockable", )

    def __init__(self, config, prefs, button_color):
        # Determine if the viewer is enabled.
        has_viewer = config.viewer != {}

        # Create the required subsection widgets.
        self._selection = SelectionSection(config)
        self._display   = DisplaySection(prefs, has_viewer)
        self._legend    = LegendSection(config)
        self._button    = ButtonSection(button_color)
        self._info      = InfoSection()

        # Define which subpanels are lockable.
        self._lockable = [ 
            self._display, 
            self._button 
        ]

        # Create the annotate tab.
        super().__init__(
            children = [
                self._selection,
                make_hline(),
                self._display,
                make_hline(),
                self._legend,
                make_hline(),
                self._button,
                make_hline(),
                self._info,
            ],
        )

    # Properties ---------------------------------------------------------------

    @property
    def target(self):
        return self._selection.target
 

    @property
    def annotation(self):
        return self._selection.annotation
 

    @property
    def selection(self):
        return self._selection.selection

    # Lock / Unlock ------------------------------------------------------------

    def lock(self):
        """Disable interactive widgets in all lockable sub-panels."""
        for panel in self._lockable: panel.lock()


    def unlock(self):
        """Enable interactive widgets in all lockable sub-panels."""
        for panel in self._lockable: panel.unlock()
