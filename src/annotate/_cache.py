# -*- coding: utf-8 -*-
################################################################################
# annotate/_cache.py

"""Figure and grid image caching for cortex-annotate.
 
FigureCache handles the generation and retrieval of matplotlib figure images
and their assembled annotation grids. Each image is generated once and cached
to disk as a PNG file with a companion JSON metadata sidecar. Subsequent
requests read from the cache.
 
FigureCache uses composition: it holds references to a ``Config`` (for
figure-generating functions and target metadata), a ``PathManager`` (for
file path construction), and a ``PrefsManager`` (for figure generation
parameters like ``figsize`` and ``dpi``).
"""

# Imports ----------------------------------------------------------------------

import re
import json
import numpy as np
import os.path as op
import imageio.v3 as iio
import matplotlib.pyplot as plt

# NoOpContext Class ------------------------------------------------------------

class NoOpContext:
    """A no-op context manager used as a fallback when no loading context is provided."""
    def __enter__(self): pass
    def __exit__(self, type, value, traceback): pass

# Figure Cache Class -----------------------------------------------------------

class FigureCache:
    """Generate figure and grid images.
 
    On first request for a figure or grid, generates the image using
    the configured figure function and writes it to disk. Subsequent
    requests read from the cached files.
 
    Parameters
    ----------
    config : Config
        Parsed configuration object. Provides figure-generating
        functions (``config.figures``), target metadata
        (``config.targets``), and annotation grid layouts
        (``config.annotations``).
 
    paths : PathManager
        Path manager for resolving cache file locations.
 
    prefs : PrefsManager
        Preferences manager. Provides ``figsize`` and ``dpi`` for
        figure generation via ``prefs.get_display()``.
 
    loading_context : context manager or None
        An optional context manager entered while generating images
        (e.g. to display a loading indicator). If ``None``, a
        :class:`NoOpContext` is used.
 
    Attributes
    ----------
    config : Config
        The configuration object.
 
    paths : PathManager
        The path manager.
 
    prefs : PrefsManager
        The preferences manager.
 
    loading_context : context manager
        The loading context (or :class:`NoOpContext`).
    """

    __slots__ = ( "config", "paths", "prefs", "loading_context" )

    def __init__(self, config, paths, prefs, loading_context = None):
        # Store the arguments.
        self.config = config
        self.paths  = paths 
        self.prefs  = prefs

        # Store the loading context. 
        if loading_context is None:
            loading_context = NoOpContext()
        self.loading_context = loading_context   
    
    # Figure Methods -----------------------------------------------------------
        
    def generate_figure(self, target_id, figure_name):
        """Generate a single figure image and write it to disk.
 
        Calls the configured figure function, saves the resulting
        matplotlib figure as a PNG, and writes axis limits and any
        additional metadata to a companion JSON sidecar.
 
        Parameters
        ----------
        target_id : tuple of str
            Target identifier tuple.
 
        figure_name : str
            Name of the figure to generate (must exist in
            ``config.figures``).
        """
        # Prepare the figure and meta data file paths.
        impath = self.paths.get_figure_path(target_id, figure_name)
        mdpath = re.sub(".png$", ".json", impath)
        
        # Get the configuration information. 
        figure_fn = self.config.figures[figure_name]
        target    = self.config.targets[target_id]
        
        # Get the figure parameters from preferences.
        figsize = self.prefs.get_display("figsize") 
        dpi     = self.prefs.get_display("dpi")

        # Run the function from the config that draws the figure.
        fig, ax = plt.subplots(1, 1, figsize = figsize, dpi = dpi)
        meta_data = {} # initalize, can be populated by figure function
        figure_fn(target, figure_name, fig, ax, figsize, dpi, meta_data)
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax.axis("off")

        # Save the figure out as a png file.
        plt.savefig(impath, bbox_inches = None)
        
        # We also need a companion meta-data file.
        if "xlim" not in meta_data: meta_data["xlim"] = ax.get_xlim()
        if "ylim" not in meta_data: meta_data["ylim"] = ax.get_ylim()

        # Save the meta data as a json file.
        meta_data = json.dumps(meta_data)
        with open(mdpath, "wt") as f:
            f.write(meta_data)

        # We can close the figure now as well.
        plt.close(fig)
    

    def figure(self, target_id, figure_name):
        """Return the image data and metadata for a figure.
 
        If the figure has not been generated yet, generates it first
        (inside the loading context). If *figure_name* is ``None``,
        returns a blank white RGBA image whose dimensions match the
        configured ``figsize`` and ``dpi``.
 
        Parameters
        ----------
        target_id : tuple of str
            Target identifier tuple.
 
        figure_name : str or None
            Figure name, or ``None`` for a blank placeholder image.
 
        Returns
        -------
        tuple of (ndarray, dict or None)
            ``(image_data, meta_data)`` where *image_data* is a uint8
            RGBA numpy array and *meta_data* is a dict with at least
            ``"xlim"`` and ``"ylim"`` keys (or ``None`` for blank images).
        """
        # If the figure name is None, return an empty (white) RGBA image
        # whose pixel dimensions match the configured figsize and dpi.
        if figure_name is None: 
            sz = self.prefs.get_display("figure_size")
            return ( np.ones((sz, sz, 4), dtype = np.uint8) * 255, None )
        
        # Prepare the image and meta data file paths.
        impath = self.paths.get_figure_path(target_id, figure_name)
        mdpath = re.sub(".png$", ".json", impath)   
        
        # If the files does not already exist, we generate them first.
        if not op.isfile(impath) or not op.isfile(mdpath):
            with self.loading_context:
                self.generate_figure(target_id, figure_name)
        
        # Now read the figure image data and meta data.
        image_data = iio.imread(impath)
        with open(mdpath, "rt") as f:
            meta_data = json.load(f)

        # And return the image data and meta data.
        return ( image_data, meta_data )

    # Grid Methods -------------------------------------------------------------

    def generate_grid(self, target_id, annotation):
        """Generate an assembled grid image for an annotation.
 
        Retrieves (generating if necessary) every figure referenced in
        the annotation's figure grid, validates that all figures share
        the same axis limits, concatenates them into a single grid
        image, and writes the result to disk with a JSON metadata
        sidecar.
 
        Parameters
        ----------
        target_id : tuple of str
            Target identifier tuple.

        annotation : str
            Annotation name whose figure grid to assemble.
        """
        # Prepare the image and meta data file paths.
        impath = self.paths.get_grid_path(target_id, annotation)
        mdpath = re.sub(".png$", ".json", impath)

        # Get the figure grid for this annotation.
        figure_grid = self.config.annotations.figure_grid[annotation]
        
        # Get the figure image/meta data for the entire figure grid
        figure_data = [
            [ self.figure(target_id, figure_name) for figure_name in row ]
            for row in figure_grid
        ]
        
        # Validate each figure xlim and ylim meta data all match!
        meta_data = [ md for row in figure_data for (_, md) in row ]
        meta_data0 = meta_data[0] # we use this as the reference meta data
        for md in meta_data: # for each figure in the grid
            if md is not None: # skip the empty figures
                if meta_data0["xlim"] != md["xlim"]:
                    raise RuntimeError(
                        f"Not all figures have the same `xlim` for "
                        f"annotation: {annotation}")
                if meta_data0["ylim"] != md["ylim"]:
                    raise RuntimeError(
                        f"Not all figures have the same `ylim` for "
                        f"annotation: {annotation}")
                
        # Concatenate the figures to make a single grid image.
        grid = np.concatenate([
            np.concatenate([fig for (fig, _) in row], axis = 1)
            for row in figure_data], axis = 0
        )
        
        # Save it out as a png file.
        iio.imwrite(impath, grid)

        # And save out the meta data.
        jscode = json.dumps(meta_data0)
        with open(mdpath, "wt") as f:
            f.write(jscode)


    def grid(self, target_id, annotation):
        """Return the grid image bytes and metadata for an annotation.
 
        If the grid has not been generated yet, generates it first
        (inside the loading context).
 
        Parameters
        ----------
        target_id : tuple of str
            Target identifier tuple.
            
        annotation : str
            Annotation name whose grid to retrieve.
 
        Returns
        -------
        tuple of (bytes, dict)
            ``(image_data, meta_data)`` where *image_data* is the raw
            PNG bytes and *meta_data* is a dict with ``"xlim"`` and
            ``"ylim"`` keys.
        """
        # Prepare the image and meta data file paths.
        impath = self.paths.get_grid_path(target_id, annotation)
        mdpath = re.sub(".png$", ".json", impath)

        # If the files aren't here already, we generate them first.
        if not op.isfile(impath) or not op.isfile(mdpath):
            with self.loading_context:
                self.generate_grid(target_id, annotation)
        
        # Read in image data. 
        with open(impath, "rb") as f:
            image_data = f.read()

        # Read in meta data.
        with open(mdpath, "rt") as f:
            meta_data = json.load(f)
        
        # And return them.
        return ( image_data, meta_data )