# -*- coding: utf-8 -*-
################################################################################
# annotate/_cache.py
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
import imageio.v3 as iio
from warnings import warn 
import matplotlib.pyplot as plt


# Figure Cache Class -----------------------------------------------------------

class FigureCache:
    """The manager of the state of the annotation and the annotation tool.

    The `AnnotationState` class manages the state of the annotation tool. This
    state includes the cache, the user preferences (style settings), and the
    saved annotations.
    """

    __slots__ = (
        "config", "cache_path", "save_path", "git_path", "username",
        "annotations", "cortex_annotations", "preferences", "loading_context", 
        "locked"
    )
    
    def __init__(
            self,
            config_path     = "/config/config.yaml",
            cache_path      = "/cache",
            save_path       = "/save",
            git_path        = "/git",
            username        = None,
            loading_context = None,
        ):

        # Store the configuration and paths.
        self.config     = Config(config_path)
        self.cache_path = cache_path
        self.save_path  = save_path
        self.git_path   = git_path

        # We add the git username to the save path if needed here.
        if username is None:
            (username, git_reponame) = self.gitdata
        if not isinstance(username, str):
            raise RuntimeError("username must be a string or None")

        # Build up the save path.
        self.username = username
        if username != "": # if username, add as subdirectory of save path
            self.save_path = op.join(save_path, username)
        if not op.isdir(self.save_path):
            os.makedirs(self.save_path, mode = 0o755)

        # Use our loading control if we have one.
        if loading_context is None:
            loading_context = NoOpContext()
        self.loading_context = loading_context

        # (Lazily) load the annotations.
        self.annotations = self.load_annotations()

        # And (lazily) load the preferences.
        self.preferences = self.load_preferences()

        # Declare the locked state of the annotation tool. When locked, the user
        # cannot interact with the figure panel and some control panel options 
        # are disabled. This is used when there is an error with the current
        # selection that prevents the figure from being properly displayed.
        self.locked = False
        
    # Git Methods --------------------------------------------------------------

    @property
    def gitdata(self):
        """Reads and returns the repo username and the repo name."""
        # If we were not given a git path, we return standard nothings.
        if self.git_path is None:
            return ( "", "" )
        try:
            # For some reason, it seems that sometimes docker does not fully
            # mount the directory until we've attempted to list its contents.
            with os.popen(f"ls {self.git_path}") as f: f.read()
            # Having performed an ls, go ahead and check git's opinion about the
            # origin with git config command line calls.
            cmd  = f"cd {self.git_path}"
            cmd += f" && git config --global --add safe.directory {self.git_path}"
            cmd +=  " && git config --get remote.origin.url"
            with os.popen(cmd) as p:
                repo_url = p.read().strip()
            repo_split = repo_url.split("/")
            repo_name = repo_split.pop()
            while repo_name == "":
                repo_name = repo_split.pop()
            repo_user = repo_split.pop()
            s1 = repo_user.split("/")[-1]
            s2 = repo_user.split(":")[-1]
            repo_user = s1 if len(s1) < len(s2) else s2
            return ( repo_user, repo_name )
        except Exception as e:
            # If there was an error, we just warn and return nothings.
            warn(f"Error finding gitdata: {e}")
            return ( "", "" )

    
    # Figure/Grid Methods ------------------------------------------------------
    
    def _generate_figure(self, target_id, figure_name):
        """Generates a single figure for the given target and figure name."""
        # Get the current target.
        target = self.config.targets[target_id]
        
        # Prepare the image and meta data file paths.
        impath = self.target_figure_path(target, figure_name)
        mdpath = re.sub(".png$", ".json", impath)
        
        # Get the display settings and figure function.
        figsize, dpi = self.config.display.figsize, self.config.display.dpi
        figure_fn = self.config.figures[figure_name]

        # Run the function from the config that draws the figure.
        (fig, ax) = plt.subplots(1, 1, figsize = figsize, dpi = dpi)
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
        jscode = json.dumps(meta_data)
        with open(mdpath, "wt") as f:
            f.write(jscode)

        # We can close the figure now as well.
        plt.close(fig)
    

    def figure(self, target_id, figure_name):
        """Returns the image and metadata for the given target and figure name.
        
        The return value is `(image_data, meta_data)` where the `image_data` is
        a numpy array of the image data, and the `meta_data` is a `dict`.
        """
        if figure_name is None:
            # This is a request for an empty image.
            image_size = self.config.display.image_size
            return ( np.ones(image_size + (4,), dtype = np.uint8) * 255, None)
        
        # Prepare the image and meta data file paths.
        impath = self.target_figure_path(target_id, figure_name)
        mdpath = re.sub(".png$", ".json", impath)   
        
        # If the files does not already exist, we generate them first.
        if not op.isfile(impath) or not op.isfile(mdpath):
            with self.loading_context:
                self._generate_figure(target_id, figure_name)
        
        # Now read the figure image data and meta data.
        image_data = iio.imread(impath)
        with open(mdpath, "rt") as f:
            meta_data = json.load(f)

        # And return the image data and meta data.
        return ( image_data, meta_data )


    def _generate_grid(self, target_id, annotation):
        """Generates a single figure grid for an annotation."""
        # Prepare the image and meta data file paths.
        impath = self.target_grid_path(target_id, annotation)
        mdpath = re.sub(".png$", ".json", impath)

        # Get the annotation information for this annotation.
        annotation_info = self.config.annotations[annotation]
        
        # Get the figure image/meta data for the entire figure grid
        figure_data = [
            [ self.figure(target_id, figure_name) for figure_name in row ]
            for row in annotation_info.figure_grid
        ]
        
        # Make sure the figure xlim and ylim meta-data all match!
        meta_data = [ md for row in figure_data for (_, md) in row ]
        meta_data0 = meta_data[0] # we use this as the reference meta data
        for md in meta_data: # for each figure in the grid
            if md is not None: # skip the empty figures
                if meta_data0["xlim"] != md["xlim"]:
                    raise RuntimeError(f"Not all figures have the same `xlim` "
                                    f"for annotation: {annotation}")
                if meta_data0["ylim"] != md["ylim"]:
                    raise RuntimeError(f"Not all figures have the same `ylim` "
                                    f"for annotation: {annotation}")
                
        # Concatenate the figures to make a single grid image.
        grid = np.concatenate([
            np.concatenate([fig for (fig, _) in row], axis = 1)
            for row in figure_data], axis = 0
        )
        
        # Save it out as a png file.
        iio.imwrite(impath, grid)

        # And save out the meta-data.
        jscode = json.dumps(meta_data0)
        with open(mdpath, "wt") as f:
            f.write(jscode)


    def grid(self, target_id, annotation):
        """Returns the grid of figures for the given target and annotation.

        The return value is `(image_data, meta_data)` where the `image_data` is 
        the raw bytes of the file, and the `meta_data` is a `dict`.
        """
        # Prepare the image and meta data file paths.
        impath = self.target_grid_path(target_id, annotation)
        mdpath = re.sub(".png$", ".json", impath)

        # If the files aren't here already, we generate them first.
        if not op.isfile(impath) or not op.isfile(mdpath):
            with self.loading_context:
                self._generate_grid(target_id, annotation)
        
        # Read in image data. 
        with open(impath, "rb") as f:
            image_data = f.read()

        # Read in meta data.
        with open(mdpath, "rt") as f:
            meta_data = json.load(f)
        
        # And return them.
        return ( image_data, meta_data )

    # Annotation Methods -------------------------------------------------------

    def load_target_annotation(self, target_id, annotation):
        """Loads a single annotation from the save path for a given target."""
        # Get the path for this annotation.
        tsv_file = self.target_save_path(target_id, annotation)

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
            tsv_file = self.target_save_path(target_id, annotation_name)
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
        """Saves the preferences to the save directory."""
        preferences_yaml = op.join(self.save_path, ".annot-prefs.yaml")
        with open(preferences_yaml, "wt") as f:
            yaml.dump(self.preferences, f)