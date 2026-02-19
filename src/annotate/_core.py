# -*- coding: utf-8 -*-
################################################################################
# annotate/_core.py

"""Core implementation code for the annotation tool's interface.

This file primarily contains code for managing the widget and window state of
the control panel; the canvas and figure code is largely handled by the
FigurePanel widget in the _figure.py file.
"""


# Imports ######################################################################

import os
import re
import json
import yaml
import numpy as np
import pandas as pd
import os.path as op
import matplotlib as mpl
import ipywidgets as ipw
import imageio.v3 as iio
from warnings import warn
import matplotlib.pyplot as plt

from ._util    import (ldict, delay)
from ._config  import Config
from ._control import ControlPanel
from ._figure  import FigurePanel


# The State Manager ############################################################

class NoOpContext:
    def __enter__(self): pass
    def __exit__(self, type, value, traceback): pass


class AnnotationState:
    """The manager of the state of the annotation and the annotation tool.

    The `AnnotationState` class manages the state of the annotation tool. This
    state includes the cache, the user preferences (style settings), and the
    saved annotations.
    """
    
    DEFAULT_STYLE = {
        "color"      : "black",
        "linestyle"  : "solid",
        "linewidth"  : 1,
        "markersize" : 1,
        "visible"    : True
    }

    STYLE_KEYS = tuple(DEFAULT_STYLE.keys())

    #TODO: what are save_hooks....also remove builtin_annotations for now.
    __slots__ = (
        "config", "cache_path", "save_path", "git_path", "username",
        "annotations", "builtin_annotations", "preferences",
        "loading_context", "save_hooks"
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
        self.save_hooks = None

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
        # self.builtin_annotations = self.load_builtin_annotations()

        # # And (lazily) load the preferences.
        self.preferences = self.load_preferences()
        
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
    
    # Pathing Methods ----------------------------------------------------------

    def _target_path(self, target):
        """Returns the relative path for a target."""
        if isinstance(target, tuple):
            path = target
        else:
            path = [target[k] for k in self.config.targets.concrete_keys]
        return op.join(*path)
    
    
    def target_figure_path(self, target, figure = None, ensure = True):
        """Returns the cache path for a target's figures."""
        path = self._target_path(target)
        path = op.join(self.cache_path, "figures", path)
        if ensure and not op.isdir(path):
            os.makedirs(path, mode = 0o755)
        if figure is not None:
            path = op.join(path, f"{figure}.png")
        return path
    
    
    def target_grid_path(self, target, annotation = None, ensure = True):
        """Returns the cache path for a target's grids."""
        path = self._target_path(target)
        path = op.join(self.cache_path, "grids", path)
        if ensure and not op.isdir(path):
            os.makedirs(path, mode = 0o755)
        if annotation is not None:
            path = op.join(path, f"{annotation}.png")
        return path
    
    
    def target_save_path(self, target, annotation = None, ensure = True):
        """Returns the save path for a target's annotation data."""
        path = self._target_path(target)
        path = op.join(self.save_path, path)
        if ensure and not op.isdir(path):
            os.makedirs(path, mode = 0o755)
        if annotation is not None:
            path = op.join(path, f"{annotation}.tsv")
        return path
    
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

        The return value is `(image_data, grid_shape, meta_data)` where the
        `image_data` is the raw bytes of the file, `grid_shape` is a tuple of
        the `(row_count, column_count)` of the grid, and the `meta_data` is a
        `dict`.
        """
        # Prepare the image and meta data file paths.
        impath = self.target_grid_path(target_id, annotation)
        mdpath = re.sub(".png$", ".json", impath)

        # Get the annotation information for this annotation.
        annotation_info = self.config.annotations[annotation]
        grid_shape = np.shape(annotation_info.figure_grid) 

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
        return ( image_data, grid_shape, meta_data )

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
        for annotation in self.config.annotations.keys():
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
            
            # If they're empty, no need to save them; delete the file if it
            # exists instead.
            tsv_file = self.target_save_path(target_id, annotation_name)
            if len(coords) == 0 and op.isfile(tsv_file):
                os.remove(tsv_file)
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
    
    # Preferences Methods ------------------------------------------------------

    def load_preferences(self):
        """Loads the preferences from the save directory and returns them.

        If no preferences file is found, an empty dictionary is returned.
        """
        preferences_yaml = op.join(self.save_path, ".annot-prefs.yaml")
        if not op.isfile(preferences_yaml):
            return { "style": {}, "image_scale" : 0.5 }
        with open(preferences_yaml, "rt") as f:
            return yaml.safe_load(f)
    
    
    def save_preferences(self):
        """Saves the preferences to the save directory."""
        preferences_yaml = op.join(self.save_path, ".annot-prefs.yaml")
        with open(preferences_yaml, "wt") as f:
            yaml.dump(self.preferences, f)
    
    # Image Scale Methods ------------------------------------------------------

    def image_scale(self, new_image_scale = None):
        """Returns the image size from the user's preferences.

        `state.image_size()` returns the current image size.

        `state.image_scale(new_image_scale)` updates the current image scale.
        """
        if new_image_scale is None:
            # Just return the current image scale, or the default if it is not set.
            return self.preferences.get("image_scale", 1.0)
        else:
            # Update the image scale in the preferences, and return the new value.
            self.preferences["image_scale"] = new_image_scale
            return new_image_scale

    # Style Methods ------------------------------------------------------------

    @classmethod
    def fix_style(cls, style_dict):
        """Ensures that the given dictionary is valid as a style dictionary."""
        # Check that all the keys are valid style keys.
        for key in style_dict.keys():
            if key not in AnnotationState.STYLE_KEYS:
                raise RuntimeError(f"Invalid style key: {key}")
            
        # Check that the linewidth is a valid number.
        if "linewidth" in style_dict:
            linewidth = style_dict["linewidth"]
            if linewidth < 0 or linewidth > 20:
                raise RuntimeError(f"Invalid linewidth: {linewidth}")
        
        # Check that the linestyle is valid.
        if "linestyle" in style_dict:
            linestyle = style_dict["linestyle"]
            if linestyle not in ("solid", "dashed", "dot-dashed", "dotted"):
                raise RuntimeError(f"Invalid linestyle: {linestyle}")
            
        # Check that the color is valid.
        if "color" in style_dict:
            color = style_dict["color"]
            try: color = mpl.colors.to_hex(color)
            except Exception as e: 
                raise RuntimeError(f"Invalid color: {color}") from e
            style_dict["color"] = color # store as hex, if valid

        # Check that the markersize is a valid number.
        if "markersize" in style_dict:
            markersize = style_dict["markersize"]
            if markersize < 0 or markersize > 20:
                raise RuntimeError(f"Invalid markersize: {markersize}")
        
        # Check that the visible is a boolean.
        if "visible" in style_dict:
            visible = style_dict["visible"]
            if not isinstance(visible, bool):
                raise RuntimeError(f"Invalid visible: {visible}")
        
        # Return the style dictionary, if valid.
        return style_dict
    
    # TODO: remove builtinannotations for now, we are not currently loading them.
    def style(self, annotation, *args):
        """Returns the style dict of the given annotation.

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
        styles = self.preferences["style"]
        if nargs == 0:
            # We're just returning the current reified dict.
            new_styledict = styles.get(annotation, {})
        elif nargs == 1:
            # We're updating the dict to have the provided dictionary values.
            new_styledict = self.fix_style(args[0])
        else:
            # We're updating the dict to have the provided key/value pairs.
            new_styledict = self.fix_style(
                { key: value for (key, value) in zip(args[0::2], args[1::2])})

        # Now that we have determined the update, we just need to merge with
        # default options in order to reify the styledict.
        update_prefs = AnnotationState.DEFAULT_STYLE.copy()
        if annotation is None: # reference to active style
            update_prefs.update(self.config.display.active_style)
        else: # else, update a background annotation
            update_prefs.update(self.config.display.background_style)
        update_prefs.update(new_styledict)

        # Finally, update user's preferences.
        styles.setdefault(annotation, {})
        styles[annotation].update(update_prefs)
        self.preferences["style"] = styles

        # And return the updated styledict for the queried annotation.
        return update_prefs
    
    
    # TODO: this probably should move to figure...
    def apply_style(self, annotation, canvas, style = None):
        """Applies the style associated with an annotation name to a canvas.

        `state.apply_style(name, canvas)` applies the annotation preferences
        associated with the given annotation name to the given canvas.

        Note that the annotation name `None` refers to the foreground style.

        If the requested annotation style is not visible, this function still
        applies the style but returns `False`. Otherwise, it returns `True`.

        If the optional argument `style` is given, then that style dictionary is
        used in place of the style dictionary associated with `annotation`.
        """
        # Get the appropriate style first.
        if style is None:
            style = self.style(annotation)

        # And walk through the key/values applying them.
        lw = style["linewidth"]
        ls = style["linestyle"]
        c  = style["color"]
        v  = style["visible"]
        canvas.line_width = lw
        if ls == "solid":
            canvas.set_line_dash([])
        elif ls == "dashed":
            canvas.set_line_dash([lw*3, lw*3])
        elif ls == "dot-dashed":
            canvas.set_line_dash([lw*1, lw*2, lw*4, lw*2])
        elif ls == "dotted":
            canvas.set_line_dash([lw, lw])
        else:
            raise RuntimeError(
                f"Invalid linestyle for annotation '{annotation}': {ls}")
        c = mpl.colors.to_hex(c)
        canvas.stroke_style = c
        canvas.fill_style = c
        return v

    # Canvas Drawing Methods ---------------------------------------------------

    def draw_path(
            self, annotation, points, canvas, path = True, closed = False,
            style = None, cursor = None, fixed_head = False, fixed_tail = False
        ):
        """Draws the given path on the given canvas using the named style.

        `state.draw_path(name, path, canvas)` applies the style for the named
        annotation then draws the given `path` on the given `canvas`. Note that
        the `path` coordinate must be in canvas pixel coordinates, not figure
        coordinates.

        If the optional argument `path` is `False`, then only the points are
        drawn.

        If the optional argument `style` is given, then the given style dict
        is used instead of the stling for the `ann_name` annotation.
        """
        self.apply_style(annotation, canvas, style = None)
        # First, draw stroke the path.
        if path and len(points) > 1:
            canvas.begin_path()
            (x0,y0) = points[0]
            canvas.move_to(x0, y0)
            for (x,y) in points[1:]:
                canvas.line_to(x, y)
            if closed:
                canvas.line_to(x0, y0)
            canvas.stroke()
        # Next, draw the points.
        sty = self.style(annotation)
        ms = sty["markersize"]
        if ms <= 0 and cursor is None: return
        if fixed_head and len(points) > 0:
            (x,y) = points[0]
            canvas.fill_rect(x-ms, y-ms, ms*2, ms*2)
            rest = points[1:]
        else:
            rest = points
        if fixed_tail and len(rest) > 0:
            (x,y) = points[-1]
            canvas.fill_rect(x-ms, y-ms, ms*2, ms*2)
            rest = points[:-1]
        else:
            rest = points
        if ms > 0:
            for (x,y) in rest:
                canvas.fill_circle(x, y, ms)
        if cursor is not None:
            ms = (ms + 1) * 4/3
            canvas.set_line_dash([])
            canvas.line_width = sty["linewidth"] * 3/4
            if len(rest) == 1:
                # We plot the circle if the cursor is head otherwise we
                # don"t plot the circle.
                if cursor == "head":
                    (x,y) = rest[0]
                else:
                    ms = 0
            elif len(rest) > 1:
                if cursor == "head":
                    (x,y) = rest[0]
                else:
                    (x,y) = rest[-1]
            else:
                ms = 0
            if ms > 0:
                canvas.stroke_circle(x, y, ms)
        # That's all!


    def _calc_fixed_ends(self, annotation, target_id, error = False):
        """Given an annotation name and a dict of annotations for a target, 
        calculates the fixed head/tail and returns them as a tuple.
        """
        target      = self.config.targets[target_id]
        targ_annots = self.annotations[target_id]
        annot_data  = self.config.annotations[annotation]

        fs = []
        for fixed in (annot_data.fixed_head, annot_data.fixed_tail):
            if fixed is None:
                fs.append(None)
                continue
            missing = []
            found = {}
            for r in fixed["requires"]:
                xy = targ_annots.get(r, ())
                if len(xy) == 0:
                    missing.append(r)
                else:
                    found[r] = xy
            if len(missing) == 0:
                try:
                    f = fixed["calculate"](target, found)
                    fs.append(f)
                except Exception as e:
                    if error:
                        error = f"Error generating fixed points:\n  {e}"
                        raise ValueError(error) from None
                    else:
                        fs.append(None)
                        continue
            elif error:
                annlist = ", ".join(missing)
                raise ValueError(
                    f"The following annotations are required:\n  {annlist}")
            else:
                fs.append(None)
        return fs


    # TODO: we are not current loading the builtin annotations, skip for now.
    # def _fix_targets(self, target_id):
    #     targ = self.config.targets[target_id]
    #     return {k: annot.with_target(targ)
    #             for (k,annot) in self.config.builtin_annotations.items()}
    
    
    # TODO: we are not current loading the builtin annotations, skip for now.
    # def load_builtin_annotations(self):
    #     "Preps the builtin annotations for the tool."
    #     # We really just need to prep individual BuiltinAnnotation objects.
    #     return ldict({tid: delay(self._fix_targets, tid)
                    #   for tid in self.config.targets.keys()})
    

# The Annotation Tool ##########################################################

class AnnotationTool(ipw.HBox):
    """The core annotation tool for the `cortex-annotate` project.

    The `AnnotationTool` type handles the annotation of the cortical surface
    images for the `cortex-annotate` project.
    """

    def __init__(
            self,
            config_path = "/config/config.yaml",
            cache_path  = "/cache",
            save_path   = "/save",
            git_path    = "/git",
            username    = None,
            control_panel_background_color = "#f0f0f0",
            save_button_color = "#e0e0e0",
            # allow_fixed_edit  = True
        ):        
        """Initializes the annotation tool."""
        
        # Store the state.
        self.state = AnnotationState(
            config_path = config_path,
            cache_path  = cache_path,
            save_path   = save_path,
            git_path    = git_path,
            username    = username
        )
        # self.allow_fixed_edit = allow_fixed_edit

        # Make the control panel.
        self.control_panel = ControlPanel(
            state             = self.state,
            background_color  = control_panel_background_color,
            save_button_color = save_button_color,
        )
        
        # Make the figure panel.
        self.figure_panel = FigurePanel(self.state)
        
        # Pass the loading context over to the state.
        self.state.loading_context = self.figure_panel.loading_context

        # Go ahead and initialize the HBox component.
        super().__init__([self.control_panel, self.figure_panel])

        # Give the figure the initial image to plot.
        with self.state.loading_context:
            self.refresh_figure()

        # And a listener for the selection change.
        self.control_panel.observe_selection(self.on_selection_change)

        # Add a listener for the image scale change.
        self.control_panel.observe_image_scale(self.on_image_scale_change)

        # And a listener for the style change.
        self.control_panel.observe_style(self.on_style_change)

        # And a listener for the save button.
        self.control_panel.observe_save(self.on_save)

    # Figure Refresh Methods ---------------------------------------------------

    def refresh_figure(self):
        # Get the target and annotation.
        target_id     = self.control_panel.target
        annotation    = self.control_panel.annotation
        target_annots = self.state.annotations[target_id]

        # Draw the grid image.
        (image_data, grid_shape, meta_data) = self.state.grid(target_id, annotation)
        self.figure_panel.redraw_canvas(
            image      = ipw.Image(value = image_data, format = "png"), 
            grid_shape = grid_shape, 
            xlim       = meta_data["xlim"], 
            ylim       = meta_data["ylim"]
        )

        # builtin annotations update if we had them


        # First of all, if there is any nonempty annotation that requires the
        # current annotation, we need to print an error about it.
        # deps = []
        # for (annot_name, annot_data) in self.state.config.annotations.items():
        #     # If the annotation is empty, it doesn't matter if it a dependant.
        #     xy = target_annots.get(annot_name)
        #     if xy is None or len(xy) == 0:
        #         continue
        #     for fixed in (annot_data.fixed_head, annot_data.fixed_tail):
        #         if fixed is not None and annotation in fixed["requires"]:
        #             deps.append(annot_name)
        #             break
        # if not self.allow_fixed_edit and len(deps) > 0:
        #     fs = None
        #     annlist = ", ".join(deps)
        #     error = (
        #         f"The following annotations are dependant on the annotation"
        #         f" {annotation}: {annlist}. Please select an annotation that does"
        #         f" not depend on other existing annotations.")
        # else:
        #     # Figure out the fixed heads and tails
        #     try:
        #         fs = self._calc_fixed_ends(annotation, target_id, error = True)
        #         error = None
        #     except ValueError as e:
        #         fs = None
        #         error = e.args[0]
        # (fh, ft) = (None, None) if fs is None else fs
        # self.figure_panel.change_annotations(
        #     target_annots,
        #     builtin_annots = {},
        #     # self.state.builtin_annotations[target_id],
        #     redraw = False,
        #     annotation_types = self.state.config.annotations.types,
        #     allow = (fs is not None),
        #     fixed_heads = {annotation: fh},
        #     fixed_tails = {annotation: ft},
        #     target = target_id
        # )

        # Update the foreground style.
        # self.figure_panel.change_foreground(annotation, redraw = False)

        # If the annotation requires something that is missing, or if a fixed
        # head or tail can't yet be calculated, we need to put an appropriate
        # message up.
        # if error is not None:
        #     self.figure_panel.write_message(error)
        # else:
        #     self.figure_panel.clear_message()

    # def _calc_fixed_ends(self, annotation, target_id = None, error = False):
    #     target_id = self.control_panel.target if target_id is None else target_id
    #     return self.state._calc_fixed_ends(annotation, target_id, error = error)

    # def _prep_image(self, target_id, annotation):
    #     # Get the grid image and meta data for this target and annotation.
    #     (image_data, grid_shape, meta_data) = self.state.grid(target_id, annotation)

    #     # And return them.
    #     return (image_data, grid_shape, meta_data)

    # Event Handler Methods ----------------------------------------------------

    def on_selection_change(self, key, change):
        """This method runs when the control panel's selection changes."""
        if change.name != "value": return
        # First, things first: save the annotations.
        self.state.save_annotations()
        
        # Clear the save hooks if there are any.
        self.state.save_hooks = None

        # Update the control panel legend. 
        self.control_panel.legend_panel.update_legend(
            target_id  = self.control_panel.target,
            annotation = self.control_panel.annotation, 
        )

        # The selection has changed; we need to redraw the image and update the
        # annotations.
        self.refresh_figure()


    def on_image_scale_change(self, change):
        """This method runs when the control panel's image scale slider changes."""
        if change.name != "value": return
        # Update the state.
        self.state.image_scale(change.new)

        # Resize the figure panel. 
        self.figure_panel.resize_canvas()


    def on_style_change(self, annotation, key, change):
        """This method runs when the control panel's style elements change."""
        if change.name != "value": return
        # Update the state.
        self.state.style(annotation, key, change.new)
        
        # Then redraw the annotation.
        self.figure_panel.redraw_canvas(redraw_image = False)


    def on_save(self, button):
        """This method runs when the control panel's save button is clicked."""
        self.state.save_annotations()
        self.state.save_preferences()