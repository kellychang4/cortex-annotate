# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_figure.py

"""Figure panel facade for cortex-annotate.
 
``FigurePanel`` is the top-level widget that composes the 2D canvas
panel, optional 3D cortex viewer panel, and two HTML overlays.
It is the sole public interface through which the orchestrator
interacts with the figure rendering layer.
 
The orchestrator never imports or touches renderers directly.  All
redraw, resize, layout, lock, loading, and message operations go
through ``FigurePanel``.  User input events (mouse clicks, key
presses) are handled privately and translated into
``AnnotationEditor`` operations and redraws.
 
Overlay stacking is achieved via CSS Grid: the outer container uses a
single-cell grid and all children (the figure box, loading overlay,
message overlay) occupy the same cell.  Later DOM children render on
top of earlier ones, so the figure box sits at the bottom, the
loading overlay in the middle, and the message overlay on top.
``z-index`` values on the overlay ``<div>`` elements provide a
secondary stacking guarantee.
"""

# Imports ----------------------------------------------------------------------
 
import ipywidgets as ipw
 
from ._canvas  import CanvasPanel
from ._viewer  import ViewerPanel
from ._overlay import Overlay, LoadingContext

# Figure Panel Class -----------------------------------------------------------

class FigurePanel(ipw.Box):
    """Figure panel widget composing the 2D canvas, 3D viewer, and overlays.
 
    User input events from the canvas (mouse clicks, key presses)
    are handled internally by ``FigurePanel``, translated into
    ``AnnotationEditor`` mutations, and followed by the appropriate
    redraws.  The orchestrator does not wire input events.
 
    Two HTML overlays cover the entire figure area:
 
    * ``LoadingOverlay`` — loading screens managed via the
      ``loading_context`` context manager. Blocks interaction.
 
    * ``MessageOverlay`` — transient error messages with optional
      auto-hide timeout.  Blocks interaction.  Stacked above the
      loading overlay so error messages remain visible during loading.
 
    Parameters
    ----------
    config : Config
        Parsed tool configuration.  Used to determine whether to
        instantiate the 3D cortex viewer (``config.viewer != {}``).
 
    prefs : PrefsManager
        User preferences.  Used to read the initial canvas tile size
        via ``prefs.get_display("canvas_size")``.
 
    editor : AnnotationEditor
        The shared annotation editing model.  ``FigurePanel`` reads
        annotation state from the editor and calls its mutation
        methods in response to user input.
 
    Attributes
    ----------
    editor : AnnotationEditor
        Reference to the annotation editing model.
 
    canvas_panel : CanvasPanel
        The 2D ipycanvas renderer.
 
    viewer_panel : ViewerPanel or None
        The 3D k3d renderer, or ``None`` if the viewer section was
        omitted from config.yaml.
 
    locked : bool
        When ``True``, user input events (mouse clicks, key presses)
        are silently ignored.  Managed by ``lock()`` / ``unlock()``.
 
    loading_context : LoadingContext
        Reference-counted context manager for showing/hiding the
        loading overlay.  Used by ``FigureCache`` and the
        orchestrator.
    """

    # Define the horizontal and vertical layouts for the canvas + viewer panel.
    _HORIZONTAL_LAYOUT = ipw.Layout(
        display     = "flex", 
        flex_flow   = "row", 
        align_items = "stretch",
        overflow    = "hidden",
        border      = "1px solid deeppink",
    )

    _VERTICAL_LAYOUT = ipw.Layout(
        display     = "flex", 
        flex_flow   = "column", 
        align_items = "stretch",
        overflow    = "hidden",
        border      = "1px solid deeppink",
    )

    __slots__ = (
        "prefs", "editor", "locked", "_canvas_panel", "_viewer_panel",
        "_figure", "has_viewer", "_loading", "loading_context", "_message" 
    )

    def __init__(self, prefs, editor, has_viewer):
        """Initialize the figure panel and its child renderers.
 
        Parameters
        ----------
        config : Config
            Parsed tool configuration.
 
        prefs : PrefsManager
            User preferences manager.
 
        editor : AnnotationEditor
            The shared annotation editing model.
        """
        # Store the editor reference for internal use.
        self.prefs  = prefs
        self.editor = editor
        self.has_viewer = has_viewer

        # Build the 2D canvas renderer.
        self._canvas_panel = CanvasPanel(editor, prefs)

        # Build the 3D cortex viewer renderer, if specified.
        figure_children = [ self._canvas_panel ]
        if self.has_viewer:
            self._viewer_panel = ViewerPanel(editor, prefs)
            figure_children.append(self._viewer_panel)
        else:
            self._viewer_panel = None

        # Combine the canvas and viewer panels into a single panel for layout.
        self._figure = ipw.Box(
            children = figure_children,
            layout   = self._HORIZONTAL_LAYOUT,
        )
        self._figure.add_class("annotate-figure-item")

        # Create the loading overlay and its context manager.
        self._loading = Overlay(css_classes = "annotate-figure-item")
        self.loading_context = LoadingContext(self._loading)

        # Create the message overlay.
        self._message = Overlay(css_classes = "annotate-figure-item")

        # Assemble children: canvas + (optional) viewer, loading overlay, and 
        # message overlay.
        children = [ 
            self._make_html_header(), 
            self._figure, 
            self._loading,
            self._message,
        ]

        # Create the figure panel (CSS Grid container). 
        super().__init__(children = children)
        super().add_class("annotate-figure-container")

        # Wire canvas internal handlers.
        self._canvas_panel.observe_mouse(self._on_mouse_click)
        self._canvas_panel.observe_key(self._on_key_press)

    # CSS Helper ---------------------------------------------------------------
    
    @staticmethod
    def _make_html_header():
        """Return an ``ipw.HTML`` widget containing scoped CSS.
 
        The CSS defines the single-cell grid overlap used for overlay
        stacking and sets the crosshair cursor on canvas elements.
        """
        return ipw.HTML("""
            <style>
                .jupyter-widget-Collapse-open {
                    background-color: white;
                    width: 300px;
                }
                .annotate-figure-container {
                    display: grid;
                    grid-template-columns: [column-start] 1fr [column-end];
                    grid-template-rows: [row-start] 1fr [row-end];
                }
                .annotate-figure-item {
                    grid-column: column-start / column-end;
                    grid-row: row-start / row-end;
                    margin: 0px;
                    padding: 0px;
                }
                .annotate-figure-item > .widget-html-content {
                    width: 100%;
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: rgba(255, 255, 255, 0.85);
                    padding: 20px;
                    text-align: center;
                    pointer-events: auto;
                    font-family: HelveticaNeue, sans-serif;
                    font-size: 32px;
                    z-index: 10;
                }
                canvas {
                    cursor: crosshair !important;
                }
            </style>
        """)
 
    # Lock / Unlock ------------------------------------------------------------

    def lock(self):
        """Disable user input events on the figure panel.
 
        While locked, mouse clicks and key presses are silently
        ignored.
        """
        self.locked = True
 
 
    def unlock(self):
        """Unlock user input events on the figure panel.
 
        Enables mouse clicks and key presses after a ``lock()``.
        """
        self.locked = False
    
    # Message Methods ----------------------------------------------------------
 
    def write_message(self, message, timeout = 3.0):
        """Display a transient message over the entire figure area.
 
        Parameters
        ----------
        message : str
            The message text to display.
 
        timeout : float or None, optional
            Auto-hide after this many seconds.  Defaults to ``3.0``.
            Pass ``None`` to persist until ``clear_message()`` is
            called.
        """
        self._message.show(message, timeout = timeout)
 
 
    def clear_message(self):
        """Hide any currently displayed message."""
        self._message.hide()
 
    # Layout -------------------------------------------------------------------
 
    def switch_layout(self):
        """Switch between horizontal and vertical arrangement.
 
        Changes the inner figure box layout (canvas + viewer).  The
        outer grid container is unaffected.
 
        Parameters
        ----------
        layout : {"horizontal", "vertical"}
            The layout direction for the canvas and viewer.
        """
        layout = self.prefs.get_display("layout")
        if layout == "horizontal":
            self._figure.layout = self._HORIZONTAL_LAYOUT
        elif layout == "vertical":
            self._figure.layout = self._VERTICAL_LAYOUT
        else:
            raise ValueError(
                f"Invalid layout direction: {layout!r}. "
                f"Expected 'horizontal' or 'vertical'."
            )
 
    # Redraw -------------------------------------------------------------------
 
    def redraw(
            self,
            base       = False,
            active     = True,
            dependent  = False,
            background = False,
        ):
        """Redraw the canvas and viewer panels.
 
        Each flag controls whether the corresponding layer is
        redrawn.  Flags that are ``False`` leave that layer
        untouched.
 
        Parameters
        ----------
        base : bool, optional
            Redraw the base layer (grid image on canvas, cortex mesh
            on viewer).  Defaults to ``False``.
 
        active : bool, optional
            Redraw the active annotation layer.  Defaults to ``True``.
 
        dependent : bool, optional
            Redraw the dependent annotation layer (annotations whose
            fixed points derive from the active annotation).
            Defaults to ``False``.
 
        background : bool, optional
            Redraw all background annotation layers.
            Defaults to ``False``.
        """
        # Redraw the 2D canvas.
        self._canvas_panel.redraw_canvas(
            image      = base,
            active     = active,
            dependent  = dependent,
            background = background,
        )
 
        # Redraw the 3D viewer, if present.
        if self.has_viewer is not None:
            # TODO: might need to bring back clear for target refreshing.
            self._viewer_panel.redraw_viewer(
                cortex     = base,
                active     = active,
                dependent  = dependent,
                background = background,
            )

    # Internal Handlers --------------------------------------------------------
 
    def _on_mouse_click(self, points):
        """Handle a mouse click on the canvas.
 
        Translates the click into figure coordinates (already done by
        ``CanvasPanel.observe_mouse``), pushes the point to the
        editor, and redraws affected layers.
 
        Parameters
        ----------
        points : ndarray, shape (1, 2)
            Click position in figure coordinates.
        """
        # If the figure is locked, we do not allow events.
        if self.locked: return
 
        # Push the point and get back any dependent annotations that changed.
        fixed_deps = self.editor.push_point(points)
 
        # Redraw active annotation (redraw dependent layer if deps changed).
        self.redraw(active = True, dependent = len(fixed_deps) > 0)
 
 
    def _on_key_press(self, key, shift_down, ctrl_down, meta_down):
        """Handle a key press on the canvas.
 
        Dispatches to the appropriate ``AnnotationEditor`` method
        based on the key pressed.
 
        Parameters
        ----------
        key : str
            The key identifier (e.g. ``"Tab"``, ``"Backspace"``,
            ``"ArrowLeft"``).
 
        shift_down : bool
            Whether the Shift key was held.
 
        ctrl_down : bool
            Whether the Ctrl key was held.
 
        meta_down : bool
            Whether the Meta (Cmd) key was held.
        """
        # If the figure is locked, we do not allow events.
        if self.locked: return
 
        # Handle the key press.
        if key == "Tab":
            # Cycle cursor to the next editable point.
            self.editor.toggle_cursor()
 
            # Toggling does not edit points, so no fixed deps.
            fixed_deps = []
 
        elif key == "Backspace":
            # Delete the point at the cursor.
            fixed_deps, error_msg = self.editor.pop_point()
 
            # If error, show the error message and exit without redrawing.
            if error_msg is not None:
                self.write_message(error_msg, timeout = 3.0)
                return
 
        elif key == "ArrowLeft":
            # Switch insertion direction to "before" cursor.
            self.editor.insert = "before"
            return
 
        elif key == "ArrowRight":
            # Switch insertion direction to "after" cursor.
            self.editor.insert = "after"
            return
 
        else:
            # Unhandled key, do nothing.
            return
 
        # Redraw active annotation (redraw dependent layer if deps changed).
        self.redraw(active = True, dependent = len(fixed_deps) > 0)



    def resize_canvas(self):
        """Handle a change in the canvas tile size display preference.

        Parameters
        ----------
        new_size : int
            The new canvas tile size.
            The ipywidgets change object.
        """
        self._canvas_panel.resize()


    def set_canvas(self, image, grid, grid_shape, xlim, ylim):
        """Set the canvas image and metadata for the given target and annotation.

        Parameters
        ----------
        image : ndarray
            The image data to display.

        grid : ndarray
            The grid data for the figure.

        grid_shape : tuple
            The shape of the grid.

        xlim : tuple
            The x-axis limits for the figure.

        ylim : tuple
            The y-axis limits for the figure.
        """
        self._canvas_panel.set_canvas(
            image = image, grid = grid, grid_shape = grid_shape, xlim = xlim, ylim = ylim
        )

    def set_viewer(self, faces, coordinates, overlays, canvas_to_viewer):
        """Set the cortex geometry and overlay data for the viewer.

        Parameters
        ----------
        faces : ndarray
            The cortex mesh faces.

        coordinates : ndarray
            The cortex mesh vertex coordinates.

        overlays : dict
            A mapping from overlay names to overlay data arrays.

        canvas_to_viewer : dict
            A mapping from canvas coordinate keys to viewer coordinate keys.
        """
        self._viewer_panel.set_viewer(
            faces, coordinates, overlays, canvas_to_viewer
        )


    def update_viewer(self, annotations = None):
        """Update a specific overlay in the viewer."""
        
        self._viewer_panel.update(annotations)


    def resize_viewer(self):
        """Handle a change in the viewer tile size display preference.

        Parameters
        ----------
        new_size : int
            The new viewer tile size.
            The ipywidgets change object.
        """
        self._viewer_panel.resize()