# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_flexible/_note.py

"""DOCSTRING
"""

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

# The Note Section -------------------------------------------------------------

class NoteSection(ipw.VBox):

    __slots__ = ()

    def __init__(self):
        self._note_textarea = ipw.Textarea(
            value       = 'Hello World',
            placeholder = 'Type something',
            description = 'String:',
        )

        super().__init__(
            children = [ self._note_textarea ],
            layout   = { "margin": "3%", "width": "88%" },
        )

    def get_note(self):
        """Get the current note."""
        return self._note_textarea.value
    
    def set_note(self, note):
        """Set the current note."""
        self._note_textarea.value = note