# -*- coding: utf-8 -*-
################################################################################
# annotate/_core.py

"""Core orchestrator for the cortex-annotate annotation tool.

``AnnotationTool`` is the top-level widget that wires together the
configuration, persistence, editing, rendering, and control layers.

Construction Chain
------------------
``PathManager`` → ``Config`` → ``AnnotationState`` → ``PrefsManager``
→ ``AnnotationEditor`` → ``FigurePanel`` → ``FigureCache`` →
``ControlPanel``.  Each object receives only the dependencies it needs;
there are no circular references.

Event Routing
-------------
All user-facing events originate from either ``ControlPanel`` (dropdown
changes, button clicks, style changes) or ``FigurePanel`` (mouse
clicks, key presses).  ``FigurePanel`` handles its own input events
internally.  ``ControlPanel`` events are routed to private handler
methods on ``AnnotationTool`` via the facade's ``observe_*`` methods.

The orchestrator never reaches past ``ControlPanel`` into sub-panels,
and never reaches past ``FigurePanel`` into renderers.

Lock / Unlock
-------------
The tool-wide ``locked`` flag lives on ``AnnotationTool``.  When the
tool is locked (during target switches, error states, etc.),
``ControlPanel.lock()`` disables interactive widgets (except selection
dropdowns, so users can navigate away from errors) and
``FigurePanel.lock()`` suppresses mouse/key events.
"""

# Imports ----------------------------------------------------------------------

import numpy as np
import ipywidgets as ipw
from functools import partial

from .config  import Config
from ._paths  import PathManager
from ._prefs  import PrefsManager
from ._annots import AnnotationState
from ._editor import AnnotationEditor
from ._cache  import FigureCache
from .control import ControlPanel
from .figure  import FigurePanel

# The Annotation Tool ----------------------------------------------------------

class AnnotationTool(ipw.HBox):
    """Top-level orchestrator widget for cortex-annotate.

    Composes and wires the control panel (left) and figure panel
    (right) into a horizontal layout.  All cross-component
    coordination — target switching, annotation saving, style
    propagation, legend updates — is handled here.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file.

    cache_path : str, optional
        Directory for cached figure and grid images.

    save_path : str, optional
        Directory for saved annotation TSV files.

    git_path : str, optional
        Path to the git repository root (for username detection).

    username : str or None, optional
        Override for the git-detected username.

    background_color : str, optional
        CSS background color for the control panel.

    button_color : str, optional
        CSS background color for the Save button.

    Attributes
    ----------
    config : Config
        Parsed project configuration (read-only after construction).

    paths : PathManager
        File path construction for cache, save, and git directories.

    state : AnnotationState
        In-memory annotation data with lazy loading and persistence.

    prefs : PrefsManager
        User preferences (styles, display settings).

    editor : AnnotationEditor
        Shared annotation editing model (pure data, no widgets).

    figure_panel : FigurePanel
        Figure rendering facade (canvas + viewer + overlays).

    cache : FigureCache
        Figure and grid image cache (generate-if-missing).

    control_panel : ControlPanel
        Control UI facade (selection, legend, style, display, buttons).

    locked : bool
        Tool-wide lock state.  When ``True``, the control panel's
        interactive widgets (except selection) are disabled and the
        figure panel ignores user input.
    """

    __slots__ = (
        "config", "paths", "prefs", "editor", "cache",
        "control_panel", "figure_panel", "has_viewer", "locked",
    )

    def __init__(
            self,
            config_path      = "/config/config.yaml",
            cache_path       = "/cache",
            save_path        = "/save",
            git_path         = "/git",
            username         = None,
            background_color = "#f0f0f0",
            button_color     = "#e0e0e0",
        ):
        # Read configuration file and construct path manager from arguments.
        self.config = Config(config_path)
        self.paths  = PathManager(cache_path, save_path, git_path, username)
        self.prefs  = PrefsManager(self.config, self.paths)

        self.has_viewer = self.config.viewer != {}

        # Prepare annotations and preferences manager (tool persistence).
        self.annotations = AnnotationState(self.config, self.paths)
        
        print("Inside AnnotationTool __init__")
        print("Configuration:")
        print(f" -> display.figsize: {self.config.display.figsize}")
        print(f" -> display.dpi: {self.config.display.dpi}")

        print("Preferences:")
        print(f" -> display.canvas_size: {self.prefs.get_display('canvas_size')}")

        # Prepare the annotation editor functionality.
        self.editor = AnnotationEditor(self.config)

        # Prepare the control panel UI.
        self.control_panel = ControlPanel(
            self.config, self.prefs,
            background_color = background_color,
            button_color     = button_color,
        )

        # Prepare the figure panel UI.
        self.figure_panel = FigurePanel(self.prefs, self.editor, has_viewer=self.has_viewer)

        # Prepare figure caching functionality.
        self.cache = FigureCache(
            self.config, self.paths, self.prefs,
            self.figure_panel.loading_context
        )

        # Declare the locked state of the annotation tool. When locked, the user
        # cannot interact with the figure panel and some control panel options 
        # are disabled. This is used when there is an error with the current
        # selection that prevents the figure from being properly displayed.
        self.locked = False

        # Initialize the Annotation Tool (= control panel + figure panel).
        super().__init__(
            children = [ self.control_panel, self.figure_panel ],
            layout   = { "border": "2px solid black" }
        )

        # Wire the control panel observers to their handlers.
        self.control_panel.observe_selection(self._on_selection_change) #TODO
        self.control_panel.observe_annotation_style(
            self._on_annotation_style_change) # TODO
        self.control_panel.observe_viewer_style(self._on_viewer_style_change)
        self.control_panel.observe_canvas_size(self._on_canvas_size_change)
        self.control_panel.observe_viewer_size(self._on_viewer_size_change)
        self.control_panel.observe_layout(self._on_layout_change)
        self.control_panel.observe_save(self._on_save)
        # self.control_panel.observe_clear_current(self._on_clear_current)
        # self.control_panel.observe_clear_all(self._on_clear_all)

        # Refresh figure with initial values.
        self.refresh_figure()

    # Lock / Unlock ------------------------------------------------------------

    def lock(self):
        """Lock the tool, disabling user interaction.

        Sets the tool-wide ``locked`` flag and propagates to both control
        and figure panels.
        """
        self.locked = True
        self.control_panel.lock()
        self.figure_panel.lock()


    def unlock(self):
        """Unlock the tool, re-enabling user interaction.

        Clears the tool-wide ``locked`` flag and propagates to both control
        and figure panels.
        """
        self.locked = False
        self.control_panel.unlock()
        self.figure_panel.unlock()

    # Messages -----------------------------------------------------------------

    def write_message(self, message, timeout = None):
        """Write a message to the figure panel.

        Parameters
        ----------
        message : str
            The message text to display.  May contain HTML markup.

        timeout : float or None, optional
            If a number of seconds, clears the message after that time.
            If ``None``, leaves the message until the next update.
            Defaults to ``None``.
        """
        self.figure_panel.write_message(message, timeout)

    def clear_message(self):
        """Clear any message from the figure panel."""
        self.figure_panel.clear_message()

    # Core Orchestration -------------------------------------------------------

    def refresh_figure(self):
        """Load and display the figure for the current selection.

        This is the central orchestration method, called on every
        target or annotation change.  
        
        The actions performed are:
        1. Reads the current target and annotation from the control
           panel.
        2. Loads the target's annotation data from ``AnnotationState``
           (triggers lazy loading on first access).
        3. Validates that all fixed-point dependencies are satisfiable.
        4. On validation failure: locks the tool and displays an error
           message.
        5. On success: updates the ``AnnotationEditor``, loads cached
           grid/cortex data into the renderers, and redraws.
        """
        # Read current selection from the control panel.
        target_id  = self.control_panel.target
        if target_id is None: return # no target selected yet, wait for valid selection

        annotation = self.control_panel.annotation
        if annotation is None: return # no annotation selected yet, wait for valid selection

        # Load the target's annotation coordinates.
        target_annots = self.annotations[target_id]

        # Validate that all fixed-point dependencies can be resolved.
        error = self._validate_fixed_dependencies(
            target_id, annotation, target_annots)

        # If there is an error, lock the tool and display the error.
        if error is not None:
            # Lock the tool and write the message
            self.lock()
            self.figure_panel.write_message(error)
            return

        # There was no error, unlock and clear any lingering error messages.
        self.unlock()
        self.figure_panel.clear_message()

        # Update the annotation editor. 
        target_changed = self.editor.update(
            target_id, annotation, target_annots)

        # If the target did not changed, no work to do.
        if target_changed is None: return

        # If the target changed, load new canvas and viewer data.
        self._load_canvas(target_id, annotation)
        if self.has_viewer: self._load_viewer(target_id)

        # Update viewer annotations from editor state (needed on both
        # target and annotation changes, since the active annotation
        # determines which annotations are "dependent" vs "background").
        if self.has_viewer: self.figure_panel.update_viewer()

        # Full redraw.
        self.figure_panel.redraw(
            base       = True,
            active     = True,
            dependent  = True,
            background = True,
        )


    def _validate_fixed_dependencies(self, target_id, annotation, target_annots):
        """Check that fixed-point dependencies can be resolved.

        For each annotation that the active annotation depends on
        (via ``fixed_heads`` or ``fixed_tails``), verifies that:

        1. The dependency annotation has enough data points.
        2. The fixed-point calculation function succeeds.

        Parameters
        ----------
        annotation : str
            The annotation to validate.

        target_id : tuple of str
            The current target identifier.

        target_annots : dict
            Annotation coordinate arrays for *target_id*.

        Returns
        -------
        str or None
            An error message if validation fails, or ``None`` if all
            dependencies are satisfiable.
        """
        # Get the annotation configuration for fixed point dependencies.
        annot_cfg = self.config.annotations

        # Check each annotation that this annotation's fixed points require.
        for fp in annot_cfg.fixed_points[annotation]:
            # Get the fixed point type (head or tail) for this dependency.
            fp_type = (
                "fixed_head"
                if fp in annot_cfg.fixed_heads[annotation]
                else "fixed_tail"
            )

            # For non point type annotations, check that the dependency has
            # enough data points to be considered "live."
            if annot_cfg.type[fp] != "point":
                n_required = len(annot_cfg.fixed_points[fp])
                fp_points  = target_annots.get(fp)
                if fp_points is None or fp_points.shape[0] <= n_required:
                    return (
                        f"Annotation '{annotation}' requires fixed point "
                        f"'{fp}' which is not yet available for target: "
                        f"{target_id}."
                    )

            # Verify that the fixed point can be calculated without error.
            try:
                self.editor.calc_fixed_point(
                    annotation, target_annots, fp_type)
            except Exception as e:
                return (
                    f"Annotation '{annotation}' requires fixed point "
                    f"'{fp}' which cannot be calculated for target: "
                    f"{target_id} — {e}"
                )

        # Verified all dependencies, return None (= no error)
        return None


    def _load_canvas(self, target_id, annotation):
        """Load canvas image and metadata from cache into the canvas.

        Parameters
        ----------
        target_id : tuple of str
            The target whose grid to load.

        annotation : str
            The annotation whose figure grid layout to use.
        """
        # Get the image data and figure limits from the cache.
        image_data, meta_data = self.cache.grid(target_id, annotation)

        # Get the figure grid layout from the annotation config.
        annot_cfg  = self.config.annotations
        grid       = annot_cfg.figure_grid[annotation]
        grid_shape = annot_cfg.grid_shape[annotation]

        # Extract axis limits from the metadata.
        xlim = tuple(meta_data["xlim"]) if meta_data else None
        ylim = tuple(meta_data["ylim"]) if meta_data else None

        # Set the canvas rendering variables.
        self.figure_panel.set_canvas(
            image = ipw.Image(value = image_data, format = "png"), 
            grid = grid, 
            grid_shape = grid_shape, 
            xlim = xlim, 
            ylim = ylim
        )


    def _load_viewer(self, target_id):
        """Load cortex geometry and overlays into the viewer.

        No-op if the viewer is not configured.

        Parameters
        ----------
        target_id : tuple of str
            The target whose cortex data to load.
        """
        # Get the current target data from the config.
        target_id = self.control_panel.target
        target = self.config.targets[target_id]

        # Extract viewer data from the viewer config.
        viewer_cfg       = self.config.viewer
        faces            = viewer_cfg["faces"](target)

        morph_between = viewer_cfg["morph_between"]
        if morph_between == "_default": 
            #TODO: test!
            coordinates = viewer_cfg["coordinates"]["_default"](target)
        else:
            coordinates = [
                viewer_cfg["coordinates"][x](target) 
                for x in morph_between
            ]
        
        overlays = {
            key: fn(target, key)
            for key, fn in viewer_cfg["overlays"].items()
        }
        canvas_to_viewer = partial(viewer_cfg["canvas_to_viewer"], target)
   
        print()
        self.figure_panel.set_viewer(
            faces, coordinates, overlays, canvas_to_viewer)

    # Control Panel Event Handlers ---------------------------------------------

    def _on_selection_change(self, key, change):
        """Handle a target and annotation selection change.

        Saves current annotations, updates the legend, and refreshes
        the figure for the new target.

        Parameters
        ----------
        key : str
            The concrete key of the dropdown that changed.

        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Save annotations for the previous selection before switching.
        self.annotations.save()

        # Update the legend for the new selection.
        self.control_panel.update_legend(
            self.control_panel.target,
            self.control_panel.annotation,
        )

        # Refresh figure for the new selection.
        self.refresh_figure()


    def _on_annotation_style_change(self, annotation, key, change):
        """Handle an annotation style change from the style panel.

        `fn(annotation, key, change)`` is the signature of the style change handlers passed to

        Parameters
        ----------
        annotation : str or None
            The annotation whose style changed, or ``None`` for the
            active annotation style.

        key : str
            The style key that changed (e.g. ``"color"``, ``"linewidth"``).

        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Update the annotation's style in the preferences manager.
        self.prefs.set_annotation_style(annotation, key, change.new)

        # Determine which annotation layers need to be redrawn
        fixed_deps = self.config.annotations.fixed_dependence[self.control_panel.annotation] 
        if annotation is None:
            active, dependent, background = True, False, False
        elif annotation in fixed_deps:
            active, dependent, background = False, True, False
        else:
            active, dependent, background = False, False, True

        # Redraw the current annotation's layer.
        self.figure_panel.redraw(
            active     = active,
            dependent  = dependent,
            background = background,
        )


    def _on_viewer_style_change(self, key, change):
        """Handle a viewer style change from the style panel.

        Persists the new viewer style value and redraws the viewer.
        No-op if the viewer is not configured.

        Parameters
        ----------
        key : str
            The viewer style key that changed (e.g. ``"morph_percent"``,
            ``"overlay"``).

        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Update the viewer's style in the preferences manager.
        self.prefs.set_viewer_style(key, change.new)

        if key == "morph_percent":
            base, active, dependent, background = True, True, True, True
        elif key in ( "overlay" , "overlay_alpha" ):
            base, active, dependent, background = True, False, False, False
        elif key in ( "point_size", "line_width", "line_interp" ):
            base, active, dependent, background = False, True, True, True

        # Redraw the viewer (cortex + all annotation layers).
        #TODO: only redraw the viewer panel! need a new argument for that option.
        self.figure_panel.redraw(
            base       = base,
            active     = active,
            dependent  = dependent,
            background = background,
        )


    def _on_canvas_size_change(self, change):
        """Handle a canvas size change from the display panel.

        Persists the new size, resizes the canvas, and triggers a
        full redraw.

        Parameters
        ----------
        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Update the display setting in the preferences manager.
        self.prefs.set_display("canvas_size", change.new)

        # Resize the canvas. 
        self.figure_panel.resize_canvas()


    def _on_viewer_size_change(self, change):
        """Handle a viewer size change from the display panel.

        Persists the new size, resizes the viewer, and triggers a
        full redraw.

        Parameters
        ----------
        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # If there is no viewer, do nothing!
        if not self.has_viewer: return

        # Update the display setting in the preferences manager.
        self.prefs.set_display("viewer_size", change.new)

        # Resize the viewer.
        self.figure_panel.resize_viewer()


    def _on_layout_change(self, change):
        """Handle a layout toggle change from the display panel.

        Persists the new layout and updates the figure panel
        arrangement.

        Parameters
        ----------
        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Determine the new layout string.
        new_layout = "horizontal" if change.new else "vertical"

        # Update the display layout in the preferences manager.
        self.prefs.set_display("layout", new_layout)

        # Update the figure panel layout.
        self.figure_panel.switch_layout()


    def _on_save(self, button):
        """Handle the Save button click.

        Saves both annotations and preferences to disk.

        Parameters
        ----------
        button : ipywidgets.Button
            The clicked button (unused, required by ``on_click``
            signature).
        """
        # Save annotations and preferences to disk.
        self.annotations.save()
        self.prefs.save()

        # Write a temporary "Save" message to the figure panel.
        self.write_message("Saved annotations.", timeout = 2.0)


    def _on_clear_current(self, button):
        """Handle the Clear Current button click.

        Clears the active annotation's points for the current target
        and redraws.

        Parameters
        ----------
        button : ipywidgets.Button
            The clicked button.
        """
        pass
        # Get the current target and annotation.
        # target_id  = self.control_panel.target
        # annotation = self.control_panel.annotation

        # # Reset the annotation to an empty coordinate array.
        #TODO: There needs to be a validity check here to prevent fixed errors.
        # self.annotations.annotations[target_id][annotation] = (
        #     np.zeros((0, 2), dtype = float))

        # # Refresh the figure to reflect the cleared annotation.
        # self.refresh_figure()


    def _on_clear_all(self, button):
        """Handle the "Clear All" button click.

        Clears every annotation's points for the current target and
        redraws.

        Parameters
        ----------
        button : ipywidgets.Button
            The clicked button.
        """
        pass
        # # Clear the annotations for the current target.
        # target_id = self.control_panel.target # current target id
        # for annotation in self.annotations[target_id].keys():
        #     self.annotations[target_id][annotation] = (
        #         self.figure_panel.empty_point_matrix())

        # # Refresh the figure to show the cleared annotations.
        # self.refresh_figure()
