# -*- coding: utf-8 -*-
################################################################################
# annotate/_state.py
# 
# DOCSTRING

# Imports ----------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import os.path as op

from .config._config import Config
from ._util   import ldict, delay


# Annotation State Class -------------------------------------------------------

class AnnotationState:
    """The manager of the state of the annotation and the annotation tool.

    The `AnnotationState` class manages the state of the annotation tool. This
    state includes the cache, the user preferences (style settings), and the
    saved annotations.
    """

    def __init__(self, config, paths):
        # Store the config and paths.
        self.config = config
        self.paths  = paths 

        # (Lazily) load the annotations.
        self.annotations = self.load_annotations()

        # And (lazily) load the preferences.
        # self.preferences = self.load_preferences()
        
        # Declare the locked state of the annotation tool. When locked, the user
        # cannot interact with the figure panel and some control panel options 
        # are disabled. This is used when there is an error with the current
        # selection that prevents the figure from being properly displayed.
        self.locked = False

    # Annotation Methods -------------------------------------------------------

    def load_target_annotation(self, target_id, annotation):
        """Loads a single annotation from the save path for a given target."""
        # Get the path for this annotation.
        tsv_file = self.paths.get_annotation_path(target_id, annotation)

        # If there is no file, we return an empty matrix of points.
        if not op.isfile(tsv_file):
            return np.zeros((0, 2), dtype = float)

        # Read in the coordinates using pandas (tab separated, no header).
        coords = pd.read_csv(tsv_file, sep = "\t", header = None).values

        # The TSV file must contain an N x 2 matrix of values!
        if len(coords.shape) != 2 or coords.shape[1] != 2:
            raise RuntimeError(
                f"File '{tsv_file}' for annotation '{annotation}' and "
                f"target '{target_id}' has invalid shape: {coords.shape}"
            )

        # Return the coordinates.
        return coords
    
    
    def load_target_annotations(self, target_id):
        """Loads (lazily) the annotations for the current tool user for a single target"""
        target_annotations = ldict() # initialize
        for annotation, annotation_info in self.config.annotations.items():
            if annotation_info.filter is None or \
                annotation_info.filter(self.config.targets[target_id]):
                target_annotations[annotation] = delay(
                    self.load_target_annotation, target_id, annotation)
        return target_annotations

    
    def load_annotations(self):
        """Loads (lazily) the annotations for the current tool user from the save path."""
        return ldict({
            target_id: delay(self.load_target_annotations, target_id)
                for target_id in self.config.targets.keys()
            })


    def save_target_annotations(self, target_id):
        """Saves the annotations for the current tool user for a single target"""
        # Get the target's annotations.
        target_annotations = self.annotations[target_id]

        for annotation_name in target_annotations.keys(): 
            # Skip anything lazy. We never want to save anything that's still
            # lazy because that means that the original file hasn't been read in
            # (and thus can't have any updates).
            if target_annotations.is_lazy(annotation_name): continue
            
            # Get this annotation's coordinates.
            coords = np.asarray(target_annotations.get(annotation_name))

            # Make sure they are the right shape.
            if len(coords.shape) != 2 or coords.shape[1] != 2:
                raise RuntimeError(
                    f"Annotation '{annotation_name}' for target "
                    f"{target_id} has invalid shape: {coords.shape}"
                )
            
            # If they're empty, no need to save them.
            tsv_file = self.paths.target_save_path(target_id, annotation_name)
            if coords.shape[0] == 0: 
                # delete the file if it exists instead.
                if op.isfile(tsv_file): os.remove(tsv_file)
                continue

            # Save them using pandas.
            df = pd.DataFrame(coords)
            df.to_csv(tsv_file, index = False, header = None, sep = "\t")
    
    
    def save_annotations(self):
        """Saves the annotations for a given target."""
        annotations = self.annotations
        for target_id in annotations.keys():
            # Skip lazy keys; these targets have not even been loaded yet.
            if not annotations.is_lazy(target_id):
                self.save_target_annotations(target_id)