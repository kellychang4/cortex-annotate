# -*- coding: utf-8 -*-
################################################################################
# annotate/control/__init__.py

"""Control subpanel composition for the cortex-annotate toolkit.

Modules
-------
_control
    ``ControlPanel`` — facade widget, tab registry, and internal
    cross-panel wiring.
    
_selection
    ``SelectionPanel`` — target and annotation dropdown management.

_legend
    ``LegendPanel`` — annotation legend image display.

_style
    ``StylePanel`` — annotation and viewer style controls.

_display
    ``DisplayPanel`` — figure size and layout controls.

_buttons
    ``ButtonPanel`` — Save and Clear All action buttons.

_info
    ``InfoPanel`` — static keyboard-shortcut instructions.

Tab Registration
----------------
Additional tabs can be added to the control panel after construction
via ``ControlPanel.register_tab(panel, title)``.  Any widget can serve
as a tab panel.  If the panel exposes ``lock()`` and ``unlock()``
methods, they will be called during tool-wide lock/unlock cycles.
"""

# ------------------------------------------------------------------------------

from ._control import ControlPanel

__all__ = ( "ControlPanel", )