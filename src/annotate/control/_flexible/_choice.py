# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_flexible/_choice.py

"""DOCSTRING
"""

# Imports ----------------------------------------------------------------------

from xml.sax import handler

import ipywidgets as ipw

# The Choice Section -----------------------------------------------------------

class ChoiceSection(ipw.HBox):

    __slots__ = ( "_choice_dropdown", "_add_button" )

    def __init__(self, button_color = "#e0e0e0"):
        self._choice_dropdown = ipw.Dropdown(
            options     = [ "Face Area", "Body Area" ],
            value       = "Face Area",
            description = "Annotation:",
        )

        self._add_button = ipw.Button(
            description = "+",
            tooltip     = "Add the current annotation to the list of annotations.",
            layout      = { "width": "30px" },  
        )
        self._add_button.style.button_color = button_color

        super().__init__(
            children = [ self._choice_dropdown, self._add_button ],
            layout   = { "margin": "3%", "width": "88%" },
        )


    @property
    def choice(self):
        """Return the current annotation choice."""
        return self._choice_dropdown.value


    def observe_add(self, fn):
        """Register a handler for the add button."""
        self._add_button.on_click(fn)  