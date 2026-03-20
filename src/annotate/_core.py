# -*- coding: utf-8 -*-
################################################################################
# annotate/_core.py

"""Orchestrator for the cortex-annotate annotation tool.

``AnnotationTool`` is the top-level widget that wires together the
configuration, persistence, editing, rendering, and control layers.
It owns no domain logic itself — its sole job is construction,
event routing, and coordination.

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

from .config   import Config
from ._paths   import PathManager
from ._prefs   import PrefsManager
from ._state   import AnnotationState
from ._cache   import FigureCache
from .control  import ControlPanel
from .figure   import AnnotationEditor, FigurePanel

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
        "config", "paths", "state", "prefs", "editor",
        "figure_panel", "cache", "control_panel", "locked",
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
        # -- 1. Foundation -----------------------------------------------------
        self.config = Config(config_path)
        self.paths  = PathManager(cache_path, save_path, git_path, username)

        # -- 2. State + Persistence --------------------------------------------
        self.state = AnnotationState(self.config, self.paths)
        self.prefs = PrefsManager(self.config, self.paths)

        # -- 3. Figure model (pure data — no widgets) --------------------------
        self.editor = AnnotationEditor(self.config.annotations)

        # -- 4. Figure UI (facade builds renderers internally) -----------------
        self.figure_panel = FigurePanel(self.config, self.prefs, self.editor)

        # -- 5. Cache (needs loading_context from figure_panel) ----------------
        self.cache = FigureCache(
            self.config, self.paths, self.prefs,
            self.figure_panel.loading_context,
        )

        # -- 6. Control UI -----------------------------------------------------
        self.control_panel = ControlPanel(
            self.config, self.prefs,
            background_color = background_color,
            button_color     = button_color,
        )

        # -- 7. Tool-wide state ------------------------------------------------
        self.locked = False

        # -- 8. Initialize the HBox widget -------------------------------------
        super().__init__(
            children = [ self.control_panel, self.figure_panel ],
            layout   = ipw.Layout(border = "1px solid black"),
        )

        # -- 9. Wire control panel events --------------------------------------
        self.control_panel.observe_target(self._on_target_change)
        self.control_panel.observe_annotation(self._on_annotation_change)
        self.control_panel.observe_annotation_style(
            self._on_annotation_style_change)
        self.control_panel.observe_viewer_style(self._on_viewer_style_change)
        self.control_panel.observe_image_pixel(self._on_image_pixel_change)
        self.control_panel.observe_layout(self._on_layout_change)
        self.control_panel.observe_save(self._on_save)
        self.control_panel.observe_clear_current(self._on_clear_current)
        self.control_panel.observe_clear_all(self._on_clear_all)

        # -- 10. Initial load --------------------------------------------------
        self._refresh_figure()

    # Lock / Unlock ------------------------------------------------------------

    def _lock_tool(self):
        """Lock the tool, disabling user interaction.

        Sets the tool-wide ``locked`` flag and propagates to both
        facades.  Selection dropdowns remain enabled so the user can
        navigate away from an error state.
        """
        self.locked = True
        self.control_panel.lock()
        self.figure_panel.lock()


    def _unlock_tool(self):
        """Unlock the tool, re-enabling user interaction.

        Clears the tool-wide ``locked`` flag and propagates to both
        facades.
        """
        self.locked = False
        self.control_panel.unlock()
        self.figure_panel.unlock()

    # Core Orchestration -------------------------------------------------------

    def _refresh_figure(self):
        """Load and display the figure for the current selection.

        This is the central orchestration method, called on every
        target or annotation change.  It:

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
        annotation = self.control_panel.annotation

        # Load the target's annotation coordinates (lazy — first access
        # triggers disk read).
        target_annots = self.state.annotations[target_id]

        # Validate that all fixed-point dependencies can be resolved.
        error = self._validate_dependencies(
            annotation, target_id, target_annots)

        if error is not None:
            # Lock the tool and display the error.  The user can still use
            # the selection dropdowns to navigate to a valid state.
            self._lock_tool()
            self.figure_panel.write_message(error)
            return

        # Dependencies OK — clear any lingering error and unlock.
        self._unlock_tool()
        self.figure_panel.clear_message()

        # Update the editor.  Returns True if the target changed (need to
        # reload base data), False if only the annotation changed, or None
        # if nothing changed.
        target_changed = self.editor.update(
            target_id, annotation, target_annots)

        # If nothing changed, no work to do.
        if target_changed is None: return

        # If the target changed, load new grid and cortex data.
        if target_changed:
            self._load_grid_context(target_id, annotation)
            self._load_cortex_data(target_id)

        # Update viewer annotations from editor state (needed on both
        # target and annotation changes, since the active annotation
        # determines which annotations are "dependent" vs "background").
        if self.figure_panel.viewer_panel is not None:
            self.figure_panel.update_viewer_annotations()

        # Full redraw.
        self.figure_panel.redraw(
            base       = target_changed,
            active     = True,
            dependent  = True,
            background = True,
        )


    def _validate_dependencies(self, annotation, target_id, target_annots):
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
        annot_cfg = self.config.annotations

        # Check each annotation that this annotation's fixed points require.
        for fp in annot_cfg.fixed_points[annotation]:
            # Get the fixed-point type (head or tail) for this dependency.
            fp_type = (
                "fixed_head"
                if fp in annot_cfg.fixed_heads[annotation]
                else "fixed_tail"
            )

            # For non-point annotations, check that the dependency has
            # enough data points to be considered "live."
            atype = annot_cfg.type[fp]
            if atype != "point":
                n_required = len(annot_cfg.fixed_points[fp])
                fp_points  = target_annots.get(fp)
                if fp_points is None or fp_points.shape[0] <= n_required:
                    return (
                        f"Annotation '{annotation}' requires fixed point "
                        f"'{fp}' which is not yet available for target: "
                        f"{target_id}."
                    )

            # Verify the fixed-point calculation succeeds.
            try:
                self.editor.calc_fixed_point(
                    annotation, target_annots, fp_type)
            except Exception as exc:
                return (
                    f"Annotation '{annotation}' requires fixed point "
                    f"'{fp}' which cannot be calculated for target: "
                    f"{target_id} — {exc}"
                )

        return None


    def _load_grid_context(self, target_id, annotation):
        """Load grid image and metadata from cache into the canvas.

        Parameters
        ----------
        target_id : tuple of str
            The target whose grid to load.

        annotation : str
            The annotation whose figure grid layout to use.
        """
        # Retrieve the assembled grid image and axis-limit metadata.
        image_data, meta_data = self.cache.grid(target_id, annotation)

        # Get the figure grid layout from the annotation config.
        annot_cfg  = self.config.annotations
        grid       = annot_cfg.figure_grid[annotation]
        grid_shape = annot_cfg.grid_shape[annotation]

        # Extract axis limits from the metadata.
        xlim = tuple(meta_data["xlim"]) if meta_data else None
        ylim = tuple(meta_data["ylim"]) if meta_data else None

        # Set the canvas rendering context via the figure facade.
        # NOTE: FigurePanel.set_grid_context() is a thin pass-through
        # to CanvasPanel.  It needs to be added to _figure.py.
        self.figure_panel.set_grid_context(
            image_data, grid, grid_shape, xlim, ylim)


    def _load_cortex_data(self, target_id):
        """Load cortex geometry and overlays into the viewer.

        No-op if the viewer is not configured.

        Parameters
        ----------
        target_id : tuple of str
            The target whose cortex data to load.
        """
        # Skip if no viewer.
        if self.figure_panel.viewer_panel is None:
            return

        # Resolve the target object from config.
        target = self.config.targets[target_id]

        # Extract cortex data from the viewer config and target.
        # NOTE: The exact extraction logic depends on the ViewerConfig
        # and target structure.  This mirrors the old ViewerState
        # initialization pattern.  FigurePanel.set_cortex_data() is a
        # thin pass-through to CortexViewerPanel that needs to be added.
        viewer_cfg = self.config.viewer

        faces            = viewer_cfg.faces(target)
        coordinates      = viewer_cfg.coordinates(target)
        overlays         = viewer_cfg.overlays(target)
        canvas_to_viewer = viewer_cfg.canvas_to_viewer(target)

        self.figure_panel.set_cortex_data(
            faces, coordinates, overlays, canvas_to_viewer)

    # Control Panel Event Handlers ---------------------------------------------

    def _on_target_change(self, key, change):
        """Handle a target-selection change.

        Saves current annotations, updates the legend, and refreshes
        the figure for the new target.

        Parameters
        ----------
        key : str
            The concrete key of the dropdown that changed.

        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Save annotations for the previous target before switching.
        self.state.save_annotations()

        # Update the legend for the new selection.
        self.control_panel.update_legend(
            self.control_panel.target,
            self.control_panel.annotation,
        )

        # Load and display the new target.
        self._refresh_figure()


    def _on_annotation_change(self, change):
        """Handle an annotation-selection change.

        Updates the legend and refreshes the figure for the new
        annotation.

        Parameters
        ----------
        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Update the legend for the new annotation.
        self.control_panel.update_legend(
            self.control_panel.target,
            self.control_panel.annotation,
        )

        # Refresh the figure.
        self._refresh_figure()


    def _on_annotation_style_change(self, annotation, key, change):
        """Handle an annotation-style change from the style panel.

        Persists the new style value and redraws annotation layers.

        Parameters
        ----------
        annotation : str or None
            The annotation whose style changed, or ``None`` for the
            active-annotation style.

        key : str
            The style key that changed (e.g. ``"color"``,
            ``"linewidth"``).

        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Persist the style change.
        self.prefs.set_annotation_style(annotation, {key: change.new})

        # Redraw all annotation layers (active + dependent + background).
        self.figure_panel.redraw(
            active     = True,
            dependent  = True,
            background = True,
        )


    def _on_viewer_style_change(self, key, change):
        """Handle a viewer-style change from the style panel.

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
        # Persist the style change.
        self.prefs.set_viewer_style(key, change.new)

        # Redraw the viewer (cortex + all annotation layers).
        self.figure_panel.redraw(
            base       = True,
            active     = True,
            dependent  = True,
            background = True,
        )


    def _on_image_pixel_change(self, change):
        """Handle a figure pixel-size change from the display panel.

        Persists the new size, resizes the canvas, and triggers a
        full redraw.

        Parameters
        ----------
        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Persist the display preference.
        self.prefs.set_display("image_pixel", change.new)

        # Resize the figure panel (canvas + viewer in future).
        # NOTE: FigurePanel.resize() needs to be added to _figure.py.
        self.figure_panel.resize(change.new)

        # Reload the grid image at the new size and redraw.
        target_id  = self.control_panel.target
        annotation = self.control_panel.annotation
        self._load_grid_context(target_id, annotation)

        self.figure_panel.redraw(
            base       = True,
            active     = True,
            dependent  = True,
            background = True,
        )


    def _on_layout_change(self, change):
        """Handle a layout-toggle change from the display panel.

        Persists the new layout and updates the figure panel
        arrangement.

        Parameters
        ----------
        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Persist the display preference.
        layout = change.new
        self.prefs.set_display("layout", layout)

        # Update the figure panel layout.
        self.figure_panel.set_layout(layout)


    def _on_save(self, button):
        """Handle the Save button click.

        Saves both annotations and preferences to disk.

        Parameters
        ----------
        button : ipywidgets.Button
            The clicked button (unused, required by ``on_click``
            signature).
        """
        self.state.save_annotations()
        self.prefs.save()
        self.figure_panel.write_message("Saved.", timeout = 2.0)


    def _on_clear_current(self, button):
        """Handle the Clear Current button click.

        Clears the active annotation's points for the current target
        and redraws.

        Parameters
        ----------
        button : ipywidgets.Button
            The clicked button.
        """
        target_id  = self.control_panel.target
        annotation = self.control_panel.annotation

        # Reset the annotation to an empty coordinate array.
        self.state.annotations[target_id][annotation] = (
            np.zeros((0, 2), dtype = float))

        # Refresh the figure to reflect the cleared annotation.
        self._refresh_figure()


    def _on_clear_all(self, button):
        """Handle the Clear All button click.

        Clears every annotation's points for the current target and
        redraws.

        Parameters
        ----------
        button : ipywidgets.Button
            The clicked button.
        """
        target_id = self.control_panel.target

        # Reset every annotation for this target.
        for name in self.state.annotations[target_id]:
            self.state.annotations[target_id][name] = (
                np.zeros((0, 2), dtype = float))

        # Refresh the figure to reflect the cleared annotations.
        self._refresh_figure()