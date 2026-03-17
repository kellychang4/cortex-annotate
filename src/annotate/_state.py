# -*- coding: utf-8 -*-
################################################################################
# annotate/_state.py

"""Annotation data management for cortex-annotate.
 
AnnotationState owns the in-memory annotation coordinate data for all
targets. It handles lazy loading from TSV files on disk and saving
modified annotations back. Each annotation is an N x 2 numpy array of
(x, y) canvas coordinates.
 
This module does not manage preferences, caching, or UI state — those
responsibilities belong to ``PrefsManager``, ``FigureCache``, and the
figure/control subpackages respectively.
"""

# Imports ----------------------------------------------------------------------
 
import os
import numpy as np
import pandas as pd
import os.path as op
 
from ._util import ldict, delay

# Annotation State Class -------------------------------------------------------

class AnnotationState:
    """Manages in-memory annotation data with lazy loading and persistence.
 
    Annotation coordinates are lazily loaded from TSV files on first
    access. Only annotations that have been accessed (and therefore
    potentially modified) are written back on save — unaccessed lazy
    entries are skipped.
 
    Parameters
    ----------
    config : Config
        Parsed configuration object. Used to enumerate targets and
        annotations and to evaluate annotation filters.
 
    paths : PathManager
        Path manager for resolving annotation file locations.
 
    Attributes
    ----------
    config : Config
        The configuration object.
 
    paths : PathManager
        The path manager.
 
    annotations : ldict
        Nested lazy dict: ``{target_id: {annotation_name: ndarray}}``.
        Outer and inner dicts are both :class:`~annotate._util.ldict`
        instances that reify on first access.
    """

    def __init__(self, config, paths):
        # Store the config and paths.
        self.config = config
        self.paths  = paths
 
        # (Lazily) load the annotations.
        self.annotations = self.load_annotations()

    # Annotation Methods -------------------------------------------------------

    def load_target_annotation(self, target_id, annotation):
        """Load a single annotation's coordinates from disk.
 
        Reads a headerless, tab-separated file containing an N x 2 matrix
        of (x, y) canvas coordinates. If the file does not exist,
        returns an empty (0, 2) array.
 
        Parameters
        ----------
        target_id : tuple of str
            Target identifier tuple.

        annotation : str
            Annotation name.
 
        Returns
        -------
        ndarray, shape (N, 2)
            Annotation coordinates. Empty array if no file exists.
        """
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
        """Lazily load all annotations for a single target.
 
        Creates an :class:`~annotate._util.ldict` with one entry per
        annotation that passes the target filter. Each entry is a
        :class:`~annotate._util.delay` that calls
        :meth:`load_target_annotation` on first access.
 
        Parameters
        ----------
        target_id : tuple of str
            Target identifier tuple.
 
        Returns
        -------
        ldict
            Lazy dict mapping annotation names to coordinate arrays.
        """
        target_annotations = ldict() # initialize
        for annotation, annotation_info in self.config.annotations.items():
            if annotation_info.filter is None or \
                annotation_info.filter(self.config.targets[target_id]):
                target_annotations[annotation] = delay(
                    self.load_target_annotation, target_id, annotation)
        return target_annotations

    
    def load_annotations(self):
        """Lazily load all annotations for all targets.
 
        Creates a nested :class:`~annotate._util.ldict` where the
        outer dict maps target ids to inner lazy dicts of per-annotation
        coordinate arrays.
 
        Returns
        -------
        ldict
            ``{target_id: ldict({annotation_name: ndarray})}``.
        """
        return ldict({
            target_id: delay(self.load_target_annotations, target_id)
                for target_id in self.config.targets.keys()
            })


    def save_target_annotations(self, target_id):
        """Save all modified annotations for a single target to disk.

        Only annotations that have been accessed (i.e., whose lazy
        values have been reified) are saved. Unaccessed entries are
        skipped because they cannot have been modified.

        Annotations with zero coordinates are deleted from disk rather
        than saved as empty files.

        Parameters
        ----------
        target_id : tuple of str
            Target identifier tuple.
        """
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
            tsv_file = self.paths.get_annotation_path(target_id, annotation_name)
            if coords.shape[0] == 0: 
                # delete the file if it exists instead.
                if op.isfile(tsv_file): os.remove(tsv_file)
                continue

            # Save them using pandas.
            df = pd.DataFrame(coords)
            df.to_csv(tsv_file, index = False, header = None, sep = "\t")
    
    
    def save_annotations(self):
        """Save all modified annotations across all targets.
 
        Iterates over all targets and calls
        :meth:`save_target_annotations` for each that has been
        accessed. Targets whose lazy values have not been reified
        are skipped.
        """
        annotations = self.annotations
        for target_id in annotations.keys():
            # Skip lazy keys; these targets have not even been loaded yet.
            if not annotations.is_lazy(target_id):
                self.save_target_annotations(target_id)