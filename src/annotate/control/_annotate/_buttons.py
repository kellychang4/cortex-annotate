# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_annotate/_buttons.py

"""Save and Clear action buttons for cortex-annotate.

Provides the ``ButtonSection`` widget containing the Save, Clear Current,
and Clear All buttons. 
"""

# Imports ----------------------------------------------------------------------

import threading
import ipywidgets as ipw

from ..._util import SAVE_TIMEOUT

# The Button Subpanel ----------------------------------------------------------

class ButtonSection(ipw.HBox):
    """Horizontal row of action buttons (Save, Clear Current, Clear All).

    Parameters
    ----------
    button_color : str, optional
        CSS color for the button background, except the Clear All button.
        Defaults to ``"#e0e0e0"``.

    Attributes
    ----------
    save_button : ipywidgets.Button
        Saves annotations and preferences to disk.

    clear_current_button : ipywidgets.Button
        Clears the currently selected annotation only.

    clear_all_button : ipywidgets.Button
        Clears all annotations from the active figure.
    """

    __slots__ = ( 
        "_button_color", "_save_timer",
        "_save_button",  "_clear_current_button", "_clear_all_button", 
        "_cancel_button", "_confirm_button", "_button_box",
    )

    def __init__(self, button_color = "#e0e0e0"):
        # Store the button color
        self._button_color = button_color

        # Initialize the save timer reference
        self._save_timer   = None

        # Define the save button
        self._save_button = ipw.Button(
            description  = "Save",
            tooltip      = "Save all annotations and preferences.",
        )
        self._save_button.style.button_color = self._button_color

        # Define the clear current button
        self._clear_current_button = ipw.Button(
            description  = "Clear",
            tooltip      = "Clear the current annotation.",
        )
        self._clear_current_button.style.button_color = self._button_color

        # Define the clear all button
        self._clear_all_button = ipw.Button(
            description  = "Clear All",
            tooltip      = "Clear all annotations from the figure.",
        )
        self._clear_all_button.style.button_color = self._button_color

        # Define the cancel clear all button
        self._cancel_button = ipw.Button(
            description  = "Cancel",
            tooltip      = "Cancel the clear all action.",
            button_style = "info",
        )

        # Define the confirm clear all button
        self._confirm_button = ipw.Button(
            description  = "Confirm",
            tooltip      = "Confirm the clear all action.",
            button_style = "danger",
        )

        # Assemble the button panel
        super().__init__(
            children = [ self._save_button, self._clear_current_button, self._clear_all_button ],
            layout   = { "margin": "3% 3% 3% 3%", "width": "94%" },
        )

        # Wire the internal button events
        self._save_button.on_click(self._on_save)
        self._clear_all_button.on_click(self._on_clear_all)
        self._cancel_button.on_click(self._on_reverse_clear_all)
        self._confirm_button.on_click(self._on_reverse_clear_all)

    # Lock / Unlock ------------------------------------------------------------

    def lock(self):
        """Disable all action buttons."""
        self._save_button.disabled          = True
        self._clear_current_button.disabled = True
        self._clear_all_button.disabled     = True
        self._cancel_button.disabled        = True
        self._confirm_button.disabled       = True


    def unlock(self):
        """Enable all action buttons."""
        self._save_button.disabled          = False
        self._clear_current_button.disabled = False
        self._clear_all_button.disabled     = False
        self._cancel_button.disabled        = False
        self._confirm_button.disabled       = False

    # Expose Internal Buttons --------------------------------------------------

    def observe_save(self, fn):
        """Register an external callback for the Save button click event."""
        self._save_button.on_click(fn)

    def observe_clear_current(self, fn):
        """Register an external callback for the Clear Current button click event."""
        self._clear_current_button.on_click(fn)

    def observe_clear_all(self, fn):
        """Register an external callback for the Clear All button click event."""
        self._confirm_button.on_click(fn)

    # Internal Event Handlers --------------------------------------------------

    def _on_save(self, button):
        """DOCSTRING"""
        def _reset_save_button():
            """Reset the Save button to its default state."""
            self._save_button.description  = "Save"
            self._save_button.style.button_color = self._button_color 
            self._timer = None

        self._save_button.description  = "Saved!"
        self._save_button.style.button_color = "#4caf51"

        self._timer = None  
        self._timer = threading.Timer(SAVE_TIMEOUT, _reset_save_button)
        self._timer.start()

    def _on_clear_all(self, button):
        """DOCSTRING"""
        self.children = [ self._cancel_button, self._confirm_button ]

    def _on_reverse_clear_all(self, button):
        """DOCSTRING"""
        self.children = [ self._save_button, self._clear_current_button, self._clear_all_button ]
