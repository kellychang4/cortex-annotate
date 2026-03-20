# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_annotate/__init__.py

"""Annotate tab composition for the cortex-annotate toolkit.

Modules
-------
_selection
    ``SelectionSection`` — target and annotation dropdown management.

_legend
    ``LegendSection`` — annotation legend image display.

_style
    ``StyleSection`` — annotation and viewer style controls.

_display
    ``DisplaySection`` — figure size and layout controls.

_buttons
    ``ButtonSection`` — Save and Clear All action buttons.

_info
    ``InfoSection`` — static keyboard-shortcut instructions.
"""

# ------------------------------------------------------------------------------

from ._annotate import AnnotateTab

__all__ = ( "AnnotateTab", )