# -*- coding: utf-8 -*-
################################################################################
# annotate/_view.py

"""
Implementation code for the Cortex viewer.

# This file primarily contains code for managing the widget and window state of
# the control panel; the canvas and figure code is largely handled by the
# FigurePanel widget in the _figure.py file.
"""


# Imports ######################################################################

import os
import json
# from functools import partial
# from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as ipw
import imageio.v3 as iio
import yaml, json

# from ._util    import (ldict, delay)
# from ._config  import Config
# from ._control import ControlPanel
# from ._figure  import FigurePanel


# The State Manager ############################################################

# class NoOpContext:
#     def __enter__(self): pass
#     def __exit__(self, type, value, traceback): pass
#TODO: what is this ^^^
class CortexViewerState:
    """
    The manager of the state of the cortex viewer.

    The `CortexViewerState` class manages the state of the cortex viewer.
    """
    def __init__(
            self,
            config_path       = '/config/config.yaml',
            cache_path        = '/cache',
            loading_context   = None,
            reviewing_context = None
        ):

        self.config     = Config(config_path)
        self.cache_path = cache_path
        self.save_hooks = None
        
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path, mode=0o755)
            
        # Use our loading control if we have one.
        if loading_context is None:
            loading_context = NoOpContext()
        self.loading_context = loading_context

        if reviewing_context is None:
            reviewing_context = NoOpContext()
        self.reviewing_context = reviewing_context

        # (Lazily) load the annotations.
        self.annotations = self.load_annotations()
        self.builtin_annotations = self.load_builtin_annotations()

        # And (lazily) load the preferences.
        self.preferences = self.load_preferences()

    # @property
    #TODO: what is this ^^^
    def target_path(self, target):
        """Returns the relative path for a target."""
        if isinstance(target, tuple):
            path = target
        else:
            path = [target[k] for k in self.config.targets.concrete_keys]
        return os.path.join(*path)
    
    def target_figure_path(self, target, figure=None, ensure=True):
        """Returns the cache path for a target's figures."""
        path = self.target_path(target)
        path = os.path.join(self.cache_path, 'figures', path)
        if ensure and not os.path.isdir(path):
            os.makedirs(path, mode=0o755)
        if figure is not None:
            path = os.path.join(path, f"{figure}.png")
        return path
    
    def target_grid_path(self, target, annotation=None, ensure=True):
        """Returns the cache path for a target's grids."""
        path = os.path.join(self.cache_path, 'grids', self.target_path(target))
        if ensure and not os.path.isdir(path):
            os.makedirs(path, mode=0o755)
        if annotation is not None:
            path = os.path.join(path, f"{annotation}.png")
        return path
    
    def target_save_path(self, target, annotation=None, ensure=True):
        """Returns the save path for a target's annotation data."""
        path = os.path.join(self.save_path, self.target_path(target))
        if ensure and not os.path.isdir(path):
            os.makedirs(path, mode=0o755)
        if annotation is not None:
            path = os.path.join(path, f"{annotation}.tsv")
        return path
    
    def generate_figure(self, target_id, figure_name):
        """Generates a single figure for the given target and figure name."""
        target = self.config.targets[target_id]
        # Make a figure and axes for the plots.
        figsize = self.config.display.figsize
        dpi = self.config.display.dpi
        (fig,ax) = plt.subplots(1,1, figsize=figsize, dpi=dpi)
        # Run the function from the config that draws the figure.
        fn = self.config.figures[figure_name]
        meta_data = {}
        fn(target, figure_name, fig, ax, figsize, dpi, meta_data)
        # Tidy things up for image plotting.
        ax.axis('off')
        fig.subplots_adjust(0,0,1,1,0,0)
        path = self.target_figure_path(target, figure_name)
        plt.savefig(path, bbox_inches=None)
        # We also need a companion meta-data file.
        if 'xlim' not in meta_data: meta_data['xlim'] = ax.get_xlim()
        if 'ylim' not in meta_data: meta_data['ylim'] = ax.get_ylim()
        jscode = json.dumps(meta_data)
        path = os.path.join(self.target_figure_path(target), 
                            f"{figure_name}.json")
        with open(path, "wt") as f:
            f.write(jscode)
        # We can close the figure now as well.
        plt.close(fig)

    def figure(self, target_id, figure_name):
        """
        Returns the image and metadata for the given target and figure name.
        
        The return value is `(image_data, meta_data)` where the `image_data` is
        a numpy array of the image data, and the `meta_data` is a `dict`.
        """
        
        if figure_name is None:
            # This is a request for an empty image.
            return (np.zeros(self.config.display.imsize + (4,), dtype=np.uint8),
                    {'xlim':(0,1), 'ylim':(0,1)})
        impath = self.target_figure_path(target_id, figure_name)
        mdpath = os.path.join(self.target_figure_path(target_id),
                              f"{figure_name}.json")
        
        # If the files aren't here already, we generate them first.
        if not os.path.isfile(impath) or not os.path.isfile(mdpath):
            with self.loading_context:
                self.generate_figure(target_id, figure_name)
        
        # Now read them both in.
        image_data = iio.imread(impath)
        with open(mdpath, "rt") as f:
            meta_data = json.load(f)

        # And return them.
        return (image_data, meta_data)
    
    def imagesize(self, new_imagesize=None):
        """
        Returns the image size from the user's preferences.

        `state.imagesize()` returns the current image size.

        `state.imagesize(new_imagesize)` updates the current image size.
        """
        if new_imagesize is None:
            return self.preferences.get('imagesize', 256)
        else:
            self.preferences['imagesize'] = new_imagesize
            return new_imagesize
        
  
    def save_annotations(self):
        "Saves the annotations for a given target."
        annots = self.annotations
        for tid in annots.keys():
            # Skip lazy keys; these targets have not even been loaded yet.
            if not annots.is_lazy(tid):
                self.save_target_annotations(tid)

    #TODO: what are hooks?
    def run_save_hooks(self):
        "Runs any save hooks that were registered by the review."
        hooks = self.save_hooks
        self.save_hooks = None
        if hooks is not None:
            for (filename, (tid, fn)) in hooks.items():
                filename = os.path.join(self.target_save_path(tid), filename)
                fn(filename)


# The Cortex Viewer ############################################################

class CortexViewer(ipw.HBox):
    """
    The cortex viewer tool for the `cortex-annotate` project.

    The `CortexViewer` type handles the 3d cortex rendering of the cortical mesh
    that assists the flatmap viewer.
    """
    def __init__(
            self,
            config_path = '/config/config.yaml',
            cache_path  = '/cache',
            save_path   = '/save',
            control_panel_background_color = "#f0f0f0",
            save_button_color="#e0e0e0"
        ):

        self.cache_path = cache_path
        
        # NOTE: this is the state configuration, my interpretation is it prepares
        # the initial values....
        # TODO: does it also control the updating JS?
        self.state = CortexViewerState(
            config_path = config_path,
            cache_path  = cache_path,
            save_path   = save_path,
        )
        
        # Make the control panel.
        imagesize = self.state.imagesize()
        self.control_panel = ControlPanel(
            self.state,
            background_color  = control_panel_background_color,
            save_button_color = save_button_color,
            imagesize         = imagesize
        )
        
        # Make the figure panel.
        self.figure_panel = FigurePanel(
            self.state,
            imagesize = imagesize
        )
        
        # Pass the loading context over to the state.
        self.state.loading_context = self.figure_panel.loading_context
        self.state.reviewing_context = self.figure_panel.reviewing_context
        
        # Go ahead and initialize the HBox component.
        super().__init__((self.control_panel, self.figure_panel))
        # Now, we want to display ourselves while we load, so do that.
        from IPython.display import display
        display(self)

        # Give the figure the initial image to plot.
        with self.state.loading_context:
            self.refresh_figure()
        
        # Add a listener for the image size change.
        self.control_panel.observe_imagesize(self.on_imagesize_change)

        # And a listener for the selection change.
        self.control_panel.observe_selection(self.on_selection_change)

        # And a listener for the style change.
        self.control_panel.observe_style(self.on_style_change)

        # And a listener for the review, save, and edit buttons.
        self.control_panel.observe_save(self.on_save)
        if self.state.config.review.function is not None:
            self.control_panel.observe_review(self.on_review)
            self.control_panel.observe_edit(self.on_edit)
        # TODO: Finally initialize the outer HBox component.

    def on_imagesize_change(self, change):
        "This method runs when the control panel's image size slider changes."
        if change.name != 'value': return
        self.state.imagesize(change.new)
        # Resize the figure panel.
        self.figure_panel.resize_canvas(change.new)

    def refresh_figure(self):
        targ = self.control_panel.target
        annot = self.control_panel.annotation
        targ_annots = self.state.annotations[targ]
        # First of all, if there is any nonempty annotation that requires the
        # current annotation, we need to print an error about it.
        deps = []
        for (annot_name, annot_data) in self.state.config.annotations.items():
            # If the annotation is empty, it doesn't matter if it a dependant.
            xy = targ_annots.get(annot_name)
            if xy is None or len(xy) == 0:
                continue
            for fixed in (annot_data.fixed_head, annot_data.fixed_tail):
                if fixed is not None and annot in fixed['requires']:
                    deps.append(annot_name)
                    break
        if len(deps) > 0:
            fs = None
            annlist = ", ".join(deps)
            error = (
                f"The following annotations are dependant on the annotation"
                f" {annot}: {annlist}. Please select an annotation that does"
                f" not depend on other existing annotations.")
            fh = None
            ft = None
        else:
            # Figure out the fixed heads and tails
            annot_data = self.state.config.annotations[annot]
            (fs, reqs) = ([], [])
            for fixed in [annot_data.fixed_head, annot_data.fixed_tail]:
                if fixed is not None:
                    reqs += fixed['requires']
                fs.append(fixed)
            missing = []
            found = {}
            for r in reqs:
                xy = targ_annots.get(r, ())
                if len(xy) == 0:
                    missing.append(r)
                else:
                    found[r] = xy
            if len(missing) == 0:
                target = self.state.config.targets[targ]
                try:
                    fs = [(None if f is None else f['calculate'](target, found))
                          for f in fs]
                    error = None
                except Exception as e:
                    error = f"Error generating fixed points:\n  {e}"
                    fs = None
            else:
                fs = None
                annlist = ", ".join(missing)
                error = f"The following annotations are required:\n  {annlist}"
            (fh,ft) = (None,None) if fs is None else fs

        self.figure_panel.change_annotations(
            targ_annots,
            self.state.builtin_annotations[targ],
            redraw=False,
            annotation_types=self.state.config.annotation_types,
            allow=(fs is not None),
            fixed_heads={annot: fh},
            fixed_tails={annot: ft}
        )
        
        self.figure_panel.change_foreground(annot, redraw=False)
        # Draw the grid image.
        (imdata, grid_shape, meta) = self.state.grid(targ, annot)
        im = ipw.Image(value=imdata, format='png')
        meta = {k:meta[k] for k in ('xlim','ylim') if k in meta}
        self.figure_panel.redraw_canvas(image=im, grid_shape=grid_shape, **meta)
        # If the annotation requires something that is missing, or if a fixed
        # head or tail can't yet be calculated, we need to put an appropriate
        # message up.
        if error is not None:
            self.figure_panel.write_message(error)
        else:
            self.figure_panel.clear_message()

    def on_selection_change(self, key, change):
        "This method runs when the control panel's selection changes."
        if change.name != 'value': return
        # First, things first: save the annotations.
        self.state.save_annotations()
        # Clear the save hooks if there are any.
        self.state.save_hooks = None
        # The selection has changed; we need to redraw the image and update the
        # annotations.
        self.refresh_figure()

    def on_style_change(self, annotation, key, change):
        "This method runs when the control panel's style elements change."
        # Update the state.
        if change.name != 'value': return
        self.state.style(annotation, key, change.new)
        # Then redraw the annotation.
        self.figure_panel.redraw_canvas(redraw_image=False)

