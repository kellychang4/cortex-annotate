# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_annotate/_info.py

"""Keyboard shortcut instructions section for cortex-annotate.

Provides the ``InfoSection`` widget, a static display of mouse and
keyboard interaction hints shown inside the Selection tab.
"""

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

# The Info Section -------------------------------------------------------------

class InfoSection(ipw.VBox):
    """Static HTML panel describing mouse and keyboard shortcuts."""

    __slots__ = ()

    def __init__(self):
        super().__init__(
            children = [
                ipw.HTML(
                    "<div style='line-height:1.2; margin: 2%;'>"
                    "<center><b>CLICK</b> to add a point to the circled "
                    "end of the current annotation.</center></div>"
                ),
                ipw.HTML(
                    "<div style='line-height:1.2; margin: 2%;'>"
                    "<center><b>BACKSPACE</b> to delete the circled "
                    "point.</center></div>"
                ),
                ipw.HTML(
                    "<div style='line-height:1.2; margin: 2%;'>"
                    "<center><b>TAB</b> to toggle the circled end."
                    "</center></div>"
                ),
            ],
            layout = { "margin": "3%", "width": "88%" },
        )