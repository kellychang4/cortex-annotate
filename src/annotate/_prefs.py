# -*- coding: utf-8 -*-
################################################################################
# annotate/_prefs.py
# 
# DOCSTRING

# Imports ######################################################################

import os
import re
import json
import yaml
import numpy as np
import pandas as pd
import os.path as op
import ipywidgets as ipw
import imageio.v3 as iio
from warnings import warn 
import matplotlib.pyplot as plt

# Preference Manager Class -----------------------------------------------------

class PrefsManager():
    def __init__(self):
        pass


    def load_preferences(self):
        """Loads the preferences (figure sizing, annotation colors). 
        
        Loads existing preferences filename. Otherwise, returns a default
        preferences dictionary.
        """
        # Define the preferences filename.
        preferences_yaml = op.join(self.save_path, ".annot-prefs.yaml")
        
        # If there is no preferences file, initailize the preferences
        if not op.isfile(preferences_yaml):
            # Start with default preferences dictionary.
            preferences = { "style": {}, "figure_size": 256 } 

            # For each annotation, set the default style dictionary.
            # DEFAULT_ANNOTATION_STYLE << config.display.default_style
            styledict = DEFAULT_ANNOTATION_STYLE.copy()
            styledict = { **styledict, **self.config.display.default_style }
            for annotation in self.config.annotations.keys():
                preferences["style"][annotation] = styledict.copy()
            
            # Set the annotation for the active style as None.
            # DEFAULT_ANNOTATION_STYLE << config.display.default_style << config.display.active_style
            styledict = { **styledict, **self.config.display.active_style }
            preferences["style"][None] = styledict.copy()

            # Return the preferences.
            return preferences
        
        # Else, there is a preference file. Read and return.
        with open(preferences_yaml, "rt") as f:
            return yaml.safe_load(f)
    
    
    def save_preferences(self):
        """Saves the preferences to the save directory."""
        preferences_yaml = op.join(self.save_path, ".annot-prefs.yaml")
        with open(preferences_yaml, "wt") as f:
            yaml.dump(self.preferences, f)
