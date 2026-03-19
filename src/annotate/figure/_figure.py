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
 
Two overlay widgets sit above the renderers via CSS absolute
positioning:
 
``MessageOverlay`` displays transient error messages (e.g.
dependency-blocked deletion).  Supports auto-hide via timeout.
Blocks user interaction while visible.
 
``LoadingOverlay`` displays a loading screen during long-running
operations (cache generation, target switch, startup).  Managed via
a reference-counted context manager so nested loading operations
(e.g. cache generation inside a target switch) show a single overlay
that only clears when the outermost context exits.
"""

# Imports ----------------------------------------------------------------------
 
import ipywidgets as ipw

from ._canvas  import CanvasPanel
from ._viewer  import CortexViewerPanel 
from ._loading import LoadingOverlay, LoadingContext
from ._message import MessageOverlay

# Figure Panel Class -----------------------------------------------------------

class FigurePanel(ipw.Box):
    """Figure panel widget composing the 2D canvas, 3D viewer, and overlays.
 
    User input events from the canvas (mouse clicks, key presses)
    are handled internally by ``FigurePanel``, translated into
    ``AnnotationEditor`` mutations, and followed by the appropriate
    redraws.  The orchestrator does not wire input events.
 
    Two HTML overlays cover the entire figure area:
 
    * ``MessageOverlay`` — transient error messages with optional
      auto-hide timeout.  Blocks interaction.

    * ``LoadingOverlay`` — loading screens managed via the
      ``loading_context`` context manager. Blocks interaction.
 
    Parameters
    ----------
    config : Config
        Parsed tool configuration.  Used to determine whether to
        instantiate the 3D cortex viewer (``config.viewer != {}``).

    prefs : PrefsManager
        User preferences.  Used to read the initial canvas tile size
        via ``prefs.get_display("image_pixel")``.
     
     editor : AnnotationEditor
        The shared annotation editing model.  ``FigurePanel`` reads
        annotation state from the editor and calls its mutation
        methods in response to user input.

    Attributes
    ----------
    editor : AnnotationEditor
        Reference to the annotation editing model.

    canvas_panel : CanvasPanel
        The 2D ipycanvas renderer (private implementation detail).

    viewer_panel : CortexViewerPanel or None
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

    # Define the horizontal and vertical layouts for the figure panel.
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

    def __init__(self, config, prefs, editor):
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
        self.editor = editor

        # Define the figure locked/unlocked state.
        self.locked = False

        # Build the 2D canvas renderer.
        self.canvas_panel = CanvasPanel(
            editor, figure_size = prefs.get_display("image_pixel"),
        )

        # Build the 3D cortex viewer renderer, if specified.
        if config.viewer != {}:
            self.viewer_panel = CortexViewerPanel(editor)
        else:
            self.viewer_panel = None

        # Build the message overlay.
        self._message = MessageOverlay()

        # Build the loading overlay and its context manager.
        self._loading = LoadingOverlay()
        self.loading_context = LoadingContext(self._loading)
 
        # Assemble children: canvas, (optional) viewer, overlays.
        children = [ self._make_html_header(), self.canvas_panel ]
        if self.viewer_panel is not None:
            children.append(self.viewer_panel)
        children.append(self._message)
        children.append(self._loading)

        # Create the Box (horizontal = default) figure panel.
        super().__init__(children, layout = self._HORIZONTAL_LAYOUT)

        # Wire canvas input events to private handlers.
        self.canvas_panel.observe_mouse(self._on_mouse_click)
        self.canvas_panel.observe_key(self._on_key_press)

    # Static Helper Methods ----------------------------------------------------

    def _make_html_header(self):
        return ipw.HTML(f"""
            <style>
                .jupyter-widget-Collapse-open {{
                    background-color: white;
                    width: 300px;
                }}
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
 
    def set_layout(self, layout):
        """Switch between horizontal and vertical arrangement.
 
        Parameters
        ----------
        layout : {"horizontal", "vertical"}
            The layout direction for the canvas and viewer.
        """
        if layout == "horizontal":
            self.layout = self._HORIZONTAL_LAYOUT
        elif layout == "vertical":
            self.layout = self._VERTICAL_LAYOUT
        else:
            raise ValueError(
                f"Invalid layout direction: {layout!r}. "
                f"Expected 'horizontal' or 'vertical'."
            )

    # Redraw -------------------------------------------------------------------
 
    def redraw(self, base = False, active = True, background = False):
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

        background : bool, optional
            Redraw the background annotation layers.
            Defaults to ``False``.
        """
        # Redraw the 2D canvas.
        self.canvas_panel.redraw_canvas(
            image      = base, 
            active     = active,
            background = background
        )
 
        # Redraw the 3D viewer, if present.
        if self.viewer_panel is not None:
            #TODO: bring back clear if needed
            self.viewer_panel.redraw_viewer(
                cortex     = base, 
                active     = active,
                background = background
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
 
        # Redraw active annotation (redraw background if deps changed).
        self.redraw(active = True, background = len(fixed_deps) > 0)


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

            # toggling does not edit points, so no fixed deps
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
 
        # Redraw active annotation (redraw background if deps changed).
        self.redraw(active = True, background = len(fixed_deps) > 0)