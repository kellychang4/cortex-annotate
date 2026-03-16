# -*- coding: utf-8 -*-
################################################################################
# annotate/_cache.py
#
# DOCSTRING

# Imports ######################################################################

import os
import re
import json
import numpy as np
import os.path as op
import imageio.v3 as iio
import matplotlib.pyplot as plt

# No-Op Context Class ----------------------------------------------------------

class NoOpContext:
    def __enter__(self): pass
    def __exit__(self, type, value, traceback): pass

# Figure Cache Class -----------------------------------------------------------

class FigureCache:
    def __init__(self, config, paths, prefs, loading_context = None):
        # Store the arguments.
        self.config = config
        self.paths  = paths 
        self.prefs  = prefs

        # Store the loading control if we have one.
        if loading_context is None:
            loading_context = NoOpContext()
        self.loading_context = loading_context   
    
    # Figure Methods -----------------------------------------------------------
        
    def generate_figure(self, target_id, figure_name):
        """Generates a single figure for the given target and figure name."""
        # Prepare the figure and meta data file paths.
        impath = self.paths.get_figure_path(target_id, figure_name)
        mdpath = re.sub(".png$", ".json", impath)
        
        # Get the configuration information. 
        figure_fn = self.config.figures[figure_name]
        target    = self.config.targets[target_id]
        
        # Get the preferences information. 
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
        """Returns the image and metadata for the given target and figure name.
        
        The return value is `(image_data, meta_data)` where the `image_data` is
        a numpy array of the image data, and the `meta_data` is a `dict`.
        """
        # If the figure name is None, then return an empty (white) image.
        if figure_name is None: #TODO: config? also image_size usage...
            image_size = self.config.display.image_size
            return ( np.ones(image_size + (4,), dtype = np.uint8) * 255, None)
        
        # Prepare the image and meta data file paths.
        impath = self.paths.get_figure_path(target_id, figure_name)
        mdpath = re.sub(".png$", ".json", impath)   
        
        # If the files does not already exist, we generate them first.
        if not op.isfile(impath) or not op.isfile(mdpath):
            with self.loading_context: #TODO
                self.generate_figure(target_id, figure_name)
        
        # Now read the figure image data and meta data.
        image_data = iio.imread(impath)
        with open(mdpath, "rt") as f:
            meta_data = json.load(f)

        # And return the image data and meta data.
        return ( image_data, meta_data )

    # Grid Methods -------------------------------------------------------------

    def generate_grid(self, target_id, annotation):
        """Generates a single figure grid for an annotation."""
        # Prepare the image and meta data file paths.
        impath = self.paths.get_grid_path(target_id, annotation)
        mdpath = re.sub(".png$", ".json", impath)

        # Get the annotation figure_grid for this annotation. TODO!
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
        """Returns the grid of figures for the given target and annotation.

        The return value is `(image_data, meta_data)` where the `image_data` is 
        the raw bytes of the file, and the `meta_data` is a `dict`.
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