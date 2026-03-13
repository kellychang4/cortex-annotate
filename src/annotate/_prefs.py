# -*- coding: utf-8 -*-
################################################################################
# annotate/_prefs.py
# 
# DOCSTRING

# Imports ----------------------------------------------------------------------

import yaml
import os.path as op
from ._style import DEFAULT_ANNOTATION_STYLE

# Preference Manager Class -----------------------------------------------------

class PrefsManager:
    def __init__(self, preferences_file = ".annot-prefs.yaml"):
        # Store the preferences file paths.
        self.preferences_file = preferences_file

        # Load the preferences (from file if it exists, otherwise default).
        self.preferences = self.load_preferences()

    
    def load_preferences(self):
        """Loads the preferences. 
        
        Loads existing preferences filename. Otherwise, returns a default
        preferences dictionary.
        """
        # If there is no preferences file, initailize the preferences
        if not op.isfile(self.preferences_file):
            # Initialize with default preferences dictionary.
            preferences = { 
                "display"          : {}, 
                "figure_size"      : 256, 
                "annotation_style" : {}, 
                "viewer_style"     : {},
            }

            # For each annotation, set the default style dictionary.
            # DEFAULT_ANNOTATION_STYLE << config.display.default_style
            style_dict = DEFAULT_ANNOTATION_STYLE.copy()
            style_dict = { **style_dict, **self.config.display.default_style }
            for annotation in self.config.annotations.keys():
                preferences["annotation_style"][annotation] = style_dict.copy()
            
            # Set the annotation for the active style as None.
            # DEFAULT_ANNOTATION_STYLE << config.display.default_style << config.display.active_style
            style_dict = { **style_dict, **self.config.display.active_style }
            preferences["annotation_style"][None] = style_dict.copy()

            # Return the preferences.
            return preferences
        
        # Else, there is a preference file. Read and return.
        with open(self.preferences_file, "rt") as f:
            return yaml.safe_load(f)
    

    def get_preferences(self):
        """Returns the preferences."""
        return self.preferences


    # Style Methods ------------------------------------------------------------

    def style(self, annotation, *args):
        """Returns the styledict from preferences of the given annotation.

        `state.style(annot)` returns the current styledict for the
        annotation named `annot`. This style dictionary is always fully reified
        with all style keys.

        `state.style(annot, new_styledict)` updates the current styledict
        to have the contents of `new_styledict` then returns the new value.

        `state.style(annot, key, value)` is equivalent to
        `state.style(annot, { key : value })`.
        
        The styledict contains the keys `"linewidth"`, `"linestyle"`,
        `"markersize"`, `"color"`, and `"visible"`.
        """
        # Check the annotation name is valid.
        if annotation is not None and annotation not in self.config.annotations:
            raise RuntimeError(f"Invalid annotation name: {annotation}")

        # Check the number of argumments 
        nargs = len(args)
        if nargs > 1 and nargs % 2 != 0:
            raise RuntimeError("Invalid number of arguments given to styledict.")
            
        # In all cases, we start by calculating our own styledict.
        # See if there is a dictionary in the preferences already.
        preferences = self.preferences["style"]
        if nargs == 0:
            # We're just returning the current annotation styledict.
            new_styledict = preferences.get(annotation, {})
        elif nargs == 1:
            # We're creating a new styledict based on the provided dict.
            new_styledict = self.fix_style(args[0])
        else:
            # We're creating a new styledict based on the provided key-value pairs.
            new_styledict = self.fix_style(
                { key: value for (key, value) in zip(args[0::2], args[1::2])})
            
        # Update user's preferences with the new styledict for this annotation.
        preferences[annotation] = { **preferences[annotation], **new_styledict }
        self.preferences["style"] = preferences

        # And return the updated styledict for the queried annotation.
        return preferences[annotation]

        

    def set_preferences(self, preferences):
        """Sets the preferences."""
        self.preferences = preferences  

    
    def save_preferences(self):
        """Saves the preferences to the preferences file."""
        with open(self.preferences_file, "wt") as f:
            yaml.dump(self.preferences, f)
