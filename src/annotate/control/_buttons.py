# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_buttons.py

"""Save and Clear action buttons for cortex-annotate.

Provides the ``ButtonPanel`` widget containing the Save, Clear Current,
and Clear All buttons. 
"""

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

# The Button Subpanel ----------------------------------------------------------

class ButtonPanel(ipw.HBox):
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

    __slots__ = ( "save_button", "clear_current_button", "clear_all_button", )

    def __init__(self, button_color = "#e0e0e0"):
        # Define the save button
        self.save_button = ipw.Button(
            description  = "Save",
            tooltip      = "Save all annotations and preferences.",
        )
        self.save_button.style.button_color = button_color

        # Define the clear current button
        self.clear_current_button = ipw.Button(
            description  = "Clear",
            tooltip      = "Clear the current annotation.",
        )
        self.clear_current_button.style.button_color = button_color

        # Define the clear all button
        self.clear_all_button = ipw.Button(
            description  = "Clear All",
            tooltip      = "Clear all annotations from the figure.",
            button_style = "warning",
        )

        # Assemble the button panel
        super().__init__(
            children = [
                self.save_button,
                self.clear_current_button,
                self.clear_all_button,
            ],
            layout = { "margin": "3% 3% 3% 3%", "width": "94%" },
        )

    # Lock / Unlock ------------------------------------------------------------

    def lock(self):
        """Disable all action buttons."""
        self.save_button.disabled          = True
        self.clear_current_button.disabled = True
        self.clear_all_button.disabled     = True


    def unlock(self):
        """Enable all action buttons."""
        self.save_button.disabled          = False
        self.clear_current_button.disabled = False
        self.clear_all_button.disabled     = False
