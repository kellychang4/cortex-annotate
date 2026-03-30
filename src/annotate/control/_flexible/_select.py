# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_flexible/_select.py

"""DOCSTRING
"""

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

# The Select Section -----------------------------------------------------------

class SelectSection(ipw.VBox):

    __slots__ = ( "_select_menu")

    def __init__(self, button_color = "#e0e0e0"):
        self._select_menu = ipw.Select(
            options = [], 
            value   = None
        )

        self._remove_button = ipw.Button(
            description = "-",
            tooltip     = "Remove the current annotation from the list of annotations.",
            layout      = { "width": "30px" },  
        )
        self._remove_button.style.button_color = button_color  

        super().__init__(
            children = [ self._select_menu, self._remove_button ],
            layout   = { "margin": "3%", "width": "88%" },
        )

    def get_options(self):
        """Return the current selection."""
        return self._select_menu.options
    
    def set_options(self, options):
        """Set the options for the selection menu."""
        self._select_menu.options = options

    def get_value(self):
        """Return the current selection."""
        return self._select_menu.value
    
    def set_value(self, value):
        """Set the current selection."""
        self._select_menu.value = value

    def observe_remove(self, fn):
        """Register a handler for the remove button."""
        self._remove_button.on_click(fn)

    def observe_select(self, fn):
        """Register a handler for the selection menu."""
        self._select_menu.observe(fn, names = "value")