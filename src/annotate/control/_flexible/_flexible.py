# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_annotate/_annotate.py

"""DOCSTRING"""


# Imports ----------------------------------------------------------------------

import re
import ipywidgets as ipw

from ._choice  import ChoiceSection
from ._select  import SelectSection
from ._note    import NoteSection
from ._buttons import ButtonSection

from ..._widgets import make_hline

# The Annotate Tab -------------------------------------------------------------

class FlexibleTab(ipw.VBox):
    
    __slots__ = ( )

    def __init__(self):
        # Create the required section widgets.
        self._choice  = ChoiceSection()
        self._select  = SelectSection()
        self._note    = NoteSection()
        self._buttons = ButtonSection()

        self.notes = {} # initialize

        # Define which sections are lockable.
        # self._lockable = [ 
        #     self._display, 
        #     self._button 
        # ]

        # Create the annotate tab.
        super().__init__(
            children = [
                self._choice,
                make_hline(),
                self._select, 
                make_hline(),
                self._note,
                make_hline(),
                self._buttons
            ],
            layout = { "width": "50%", "border": "1px solid #c0c0c0" },
        )

        # Wire internal observer handlers.
        self._choice.observe_add(self._on_add)
        self._select.observe_select(self._on_select)
        self._select.observe_remove(self._on_remove)


    # Properties ---------------------------------------------------------------
 
    @property
    def choice(self):
        """The ChoiceSection widget."""
        return self._choice.choice
    
    
    def _on_add(self, button):
        """Register a handler for the add button."""
        
        # get the template choice annotation
        choice = self.choice
        print("Choice:", choice)

        # get the selection menu options
        options = self._select.get_options()
        print("Options:", options)

        # find the matching options and choices
        n = [int(re.sub(".+ (\\d+)$", "\\1", x))
             for x in options if x.startswith(choice)]
        
        if len(n) == 0: n = 1
        else:
            n_missing = set(range(1, max(n) + 1)) - set(n)
            n = max(n) + 1 if len(n_missing) == 0 else min(n_missing)

        new_choice = f"{choice} {n}"
        print("New Choice:", new_choice)

        # add the new choice to the selection menu options
        options = options + (new_choice,)
        self._select.set_options(options)

        self.notes[new_choice] = "" # initialize the note for the new choice


    def _on_remove(self, button):
        """Register a handler for the remove button."""
        # get the selection menu options
        value = self._select.get_value()
        print("Value:", value)

        options = self._select.get_options()
        print("Options:", options)

        # remove the current selection from the options
        options = tuple([x for x in options if x != value])
        print("New Options:", options)
        self._select.set_options(options)

        # remove the note for the removed choice
        if value in self.notes:
            del self.notes[value]


    def _on_select(self, change):
        """Register a handler for the select menu."""
        self.notes[change.old] = self._note.get_note() # save the note for the old choice

        print("Selected:", change.new)

        # TODO: prevent saving "None" as a note.
        self._note.set_note( self.notes[change.new] ) # load the note for the new choice