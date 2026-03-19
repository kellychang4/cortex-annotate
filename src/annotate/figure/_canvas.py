# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_canvas.py

"""2D canvas renderer for cortex-annotate.
 
``CanvasPanel`` is a pure rendering widget.  It reads annotation state
from a shared ``AnnotationEditor`` and annotation styles from a
``PrefsManager``, then redraws its ipycanvas layers when asked.
 
User input events (mouse clicks, key presses) are translated into
figure-coordinate callbacks and forwarded to ``FigurePanel``, which
in turn mutates the ``AnnotationEditor``.  The canvas never mutates
annotation state directly.
 
Canvas-specific rendering data (grid image, grid layout, axis limits)
is stored as instance attributes and set by the orchestrator (or
``FigurePanel``) before drawing begins.
 
Canvas Layers
-------------
Layer 0 : grid image — the background figure tile(s).
Layer 1 : background annotations — all non-active annotations.
Layer 2 : active annotation — the annotation being edited, with
          cursor indicator.
 
Loading screens and error messages are handled by overlays at the
``FigurePanel`` level, not by canvas layers.
"""

# Imports ----------------------------------------------------------------------
 
import numpy as np
import ipycanvas as ipc
import ipywidgets as ipw
import matplotlib as mpl
 
# The Canvas Panel -------------------------------------------------------------

class CanvasPanel(ipw.HBox):
    """2D ipycanvas renderer for figure images and annotation overlays.
 
    Manages a 3-layer ``MultiCanvas``:
 
    * Layer 0 — grid image (PNG tile).
    * Layer 1 — background annotations (non-active, all styles).
    * Layer 2 — active annotation (with cursor indicator).
 
    The canvas reads annotation coordinates, cursor position, and
    fixed-head/tail data from the ``AnnotationEditor``.  Annotation
    styles are read from the ``PrefsManager`` at draw time.
 
    Canvas-specific rendering data (grid image, grid layout, figure
    axis limits) are stored as instance attributes and must be set
    before the first ``redraw_canvas`` call.
 
    Parameters
    ----------
    editor : AnnotationEditor
        The shared annotation editing model (read-only access).

    prefs : PrefsManager
        User preferences for annotation styles.

    figure_size : int, optional
        Pixel size of one grid cell (both width and height).
        Defaults to ``256``.
 
    Attributes
    ----------
    editor : AnnotationEditor
        Reference to the annotation editing model.
    prefs : PrefsManager
        Reference to the user preferences manager.
    figure_size : ndarray, shape (2,)
        Pixel dimensions ``[width, height]`` of one grid cell.
    canvas_size : ndarray, shape (2,)
        Total pixel dimensions of the multicanvas.
    image : ipywidgets.Image or None
        The current grid image widget.  Set by the caller before
        drawing.
    grid : list of list or None
        The figure grid layout (2D list where ``None`` marks empty
        cells).  Set by the caller before drawing.
    grid_shape : tuple of (int, int) or None
        ``(n_rows, n_cols)`` of the grid.  Set by the caller before
        drawing.
    xlim : tuple of (float, float) or None
        X-axis figure limits for coordinate conversion.
    ylim : tuple of (float, float) or None
        Y-axis figure limits for coordinate conversion.
    """
 
    def __init__(self, editor, prefs, figure_size = 256):
        """Initialize the canvas panel.
 
        Parameters
        ----------
        editor : AnnotationEditor
            The shared annotation editing model.

        prefs : PrefsManager
            User preferences for annotation styles.

        figure_size : int, optional
            Pixel size of one grid cell.  Defaults to ``256``.
        """
        # Store references.
        self.editor    = editor
        self.prefs     = prefs
        self.annot_cfg = editor.annot_cfg
 
        # Store the figure size (pixels per grid cell).
        self.figure_size = np.array([figure_size, figure_size])
 
        # Rendering variables.
        self.image      = None
        self.grid       = None
        self.grid_shape = None
        self.xlim       = None
        self.ylim       = None
 
        # Get first grid shape from the first annotation for initial sizing.
        grid_shape0 = self.annot_cfg.grid_shape[self.annot_cfg.names[0]]
 
        # Calculate the canvas size (pixels) from figure size and grid shape.
        self.canvas_size = self.figure_size * grid_shape0
 
        # Build the multicanvas (3 layers: image, background, active).
        canvas_width, canvas_height = self.canvas_size
        self.multicanvas = ipc.MultiCanvas(
            3, width = canvas_width, height = canvas_height)
 
        # We always seem to need to explicitly set the layout size in pixels.
        self.multicanvas.layout.width  = f"{canvas_width}px"
        self.multicanvas.layout.height = f"{canvas_height}px"
 
        # Name the canvas layers.
        self.image_canvas      = self.multicanvas[0] # annotation image layer
        self.background_canvas = self.multicanvas[1] # background annotations layer
        self.active_canvas     = self.multicanvas[2] # active annotation layer
 
        # Initialize the HBox with the crosshair CSS and the multicanvas.
        super().__init__(
            children = [ self._make_html_header(), self.multicanvas ],
            layout   = { "border": "1px solid blue" }
        )

    # Static Helper ------------------------------------------------------------

    @staticmethod
    def _make_html_header():
        """Return an HTML widget that sets the canvas cursor to crosshair."""
        return ipw.HTML("""
            <style>
                canvas {
                    cursor: crosshair !important;
                }
            </style>
        """)

    # Image Layer --------------------------------------------------------------
 
    def redraw_image(self):
        """Clear and redraw the annotation image on the image canvas.
 
        Reads the image from ``self.image``.  No-op if ``self.image``
        is ``None``.
        """
        # If there is no image to draw, skip.
        if self.image is None: return

        # Draw the image on the canvas.
        with ipc.hold_canvas():
            self.image_canvas.clear()
            self.image_canvas.draw_image(
                self.image, 0, 0,
                self.image_canvas.width,
                self.image_canvas.height,
            )

    # Coordinate Conversion ----------------------------------------------------

    def canvas_to_figure(self, points):
        """Convert canvas pixel coordinates to figure coordinates.
 
        Applies grid-cell modular wrapping, y-axis inversion, and
        scaling by the figure axis limits.
 
        Parameters
        ----------
        points : ndarray, shape (N, 2) or (2,)
            Canvas pixel coordinates.  A 1D input is promoted to
            ``(1, 2)`` and the result squeezed back.
 
        Returns
        -------
        ndarray, shape (N, 2) or (2,)
            Corresponding figure coordinates.
        """
        # Coerce points into an `N x 2` matrix if necessary.
        points = np.asarray(points)
        if len(points.shape) == 1:
            return self.canvas_to_figure([points])[0]

        # Apply grid mod to wrap points into a single cell.
        (figure_width, figure_height) = self.figure_size
        points = points % [figure_width, figure_height]

        # Resolve figure limits (default to pixel extents).
        xlim = (0, figure_width)  if self.xlim is None else self.xlim
        ylim = (0, figure_height) if self.ylim is None else self.ylim

        # Invert y-axis (canvas origin is top-left, figure origin is bottom-left).
        points[:, 1] = figure_height - points[:, 1]

        # Now, make the conversion.
        points *= [(xlim[1] - xlim[0]) / figure_width,
                   (ylim[1] - ylim[0]) / figure_height]
        points += [xlim[0], ylim[0]]

        # Return the converted points.
        return points


    def figure_to_canvas(self, points):
        """Convert figure coordinates to canvas pixel coordinates.
 
        Returns one copy of the points per grid cell, if not ``None``.
 
        Parameters
        ----------
        points : ndarray, shape (N, 2) or (2,)
            Figure coordinates.  A 1D input is promoted to ``(1, 2)``
            and the result squeezed back.
 
        Returns
        -------
        list of ndarray, shape (N, 2)
            One point matrix per non-``None`` cell in the grid.
        """
        # Coerce points into an `N x 2` matrix if necessary.
        points = np.asarray(points)
        if len(points.shape) == 1:
            return self.figure_to_canvas([points])[0]

        # Resolve figure limits.
        (figure_width, figure_height) = self.figure_size
        xlim = (0, figure_width)  if self.xlim is None else self.xlim
        ylim = (0, figure_height) if self.ylim is None else self.ylim

        # Scale to pixel coordinates.
        points  = points - [xlim[0], ylim[0]]
        points *= [figure_width  / (xlim[1] - xlim[0]),
                   figure_height / (ylim[1] - ylim[0])]

        # Invert the y axis.
        points[:, 1] = figure_height - points[:, 1]

        # And build up the point matrices for each (not None) grid element.
        (n_rows, n_cols) = self.grid_shape
        return [
            points + [ii * figure_width, jj * figure_height]
            for ii in np.arange(n_cols)
            for jj in np.arange(n_rows)
            if self.grid[jj][ii] is not None
        ]

    # Annotation Drawing Methods -----------------------------------------------

    def redraw_annotations(self, active = True, background = True):
        """Clear and redraw annotation overlays on the active or background 
        annotation canvases.
 
        Iterates over all annotations in the editor, determines
        whether each is active or background, fetches its style from
        ``self.prefs``, converts coordinates, and draws.
 
        Parameters
        ----------
        active : bool, optional
            Whether to redraw the active annotation layer.
            Defaults to ``True``.

        background : bool, optional
            Whether to redraw the background annotation layer.
            Defaults to ``True``.
        """
        # Clear the specified canvas layers.
        if active:     self.active_canvas.clear()
        if background: self.background_canvas.clear()

        # Step through all annotations and draw them.
        for annotation, points in self.editor.annotations.items():
            # If there are no points, we can skip.
            if points is None or len(points) == 0: continue

            # Determine if this is the active or a background annotation.
            if self.editor.active == annotation:
                # Skip active annotation if active is False.
                if not active: continue
                canvas   = self.active_canvas
                styletag = None
                cursor   = self.editor.cursor
            else:
                # Skip background annotations if background is False.
                if not background: continue
                canvas   = self.background_canvas
                styletag = annotation
                cursor   = None

            # Determine if the annotation has a fixed-head or fixed-tail.
            fixed_head = self.state.fixed_heads.get(annotation, None) is not None
            fixed_tail = self.state.fixed_tails.get(annotation, None) is not None

            # Get style from preferences.
            style = self.prefs.get_annotation_style(styletag)

            # Skip, if the annotation is not visible.
            if not style["visible"]: continue

            # Determine if the path is closed (only boundaries are closed).
            atype  = self.annot_cfg.type[annotation]
            closed = atype == "boundary"

            # Convert figure points to canvas coordinates (repeated per panel).
            grid_points = self.figure_to_canvas(points)

            # Draw on each grid panel.
            for pts in grid_points:
                self.draw_points(
                    canvas     = canvas,
                    points     = pts,
                    style      = style,
                    cursor     = cursor,
                    closed     = closed,
                    fixed_head = fixed_head,
                    fixed_tail = fixed_tail, 
                    insert     = "after"
                )


    def _apply_linestyle(self, canvas, style):
        """Apply line width and dash pattern to a canvas context.
 
        Parameters
        ----------
        canvas : ipycanvas.Canvas
            The canvas layer to configure.

        style : dict
            Annotation style dict with ``linewidth`` and ``linestyle``
            keys.
        """
        # Get the linewidth and linestyle from the style dictionary.
        linewidth = style["linewidth"]
        linestyle = style["linestyle"]
 
         # Apply the linewidth and linestyle to the canvas.
        canvas.line_width = linewidth if linewidth is not None else 1
        if linestyle == "solid":
            canvas.set_line_dash([])
        elif linestyle == "dashed":
            canvas.set_line_dash([linewidth * 3, linewidth * 3])
        elif linestyle == "dot-dashed":
            canvas.set_line_dash([linewidth * 1, linewidth * 2,
                                  linewidth * 4, linewidth * 2])
        elif linestyle == "dotted":
            canvas.set_line_dash([linewidth, linewidth])
        else:
            raise ValueError(f"Invalid linestyle: {linestyle}")


    def draw_points(
        self, canvas, points, style, cursor = None, closed = False,
        fixed_head = False, fixed_tail = False, insert = "after"
    ):
        """Draw an annotation on the given canvas layer.
 
        Renders line segments between consecutive points, circular
        markers for user-placed points, square markers for fixed
        head/tail points, and a cursor ring around the active point.
 
        Parameters
        ----------
        canvas : ipycanvas.Canvas
            The canvas layer to draw on.

        points : ndarray, shape (N, 2)
            Annotation points in canvas pixel coordinates.

        style : dict
            Annotation style dict with ``color``, ``linewidth``,
            ``linestyle``, and ``markersize`` keys.

        cursor : int or None, optional
            Index of the cursor point.  If given, a ring is drawn
            around that point.

        closed : bool, optional
            Whether to close the path (connect last point to first).

        fixed_head : bool, optional
            Whether the first point is a fixed head (drawn as square).

        fixed_tail : bool, optional
            Whether the last point is a fixed tail (drawn as square).

        insert: str, optional
            Whether to insert the new points "before" or "after" the
            existing points.  Defaults to "after".
        """
        # Convert the color to an uint8 RGB array. 
        rgb_color = np.array(mpl.colors.to_rgb(style["color"]))
        rgb_color = np.array(rgb_color * 255, dtype = np.uint8)

        # Apply linewidth and linestyle.
        self._apply_linestyle(canvas, style)

        # We only draw line segments if there are at least two points. If the
        # annotation has fixed points, we only draw when there are more points 
        # than fixed points.
        n_fixed = int(fixed_head) + int(fixed_tail)
        if points.shape[0] > np.max([1, n_fixed]):
            # if the path is closed, we need to add the first point to the end 
            # of the point matrix to make sure the path is closed when we draw it.
            if closed: points = np.vstack([points, points[0:1, :]])

            # create segement coordinates pairs [(x1, y1), (x2, y2), ...] 
            segments = np.stack([points[:-1, :], points[1:, :]], axis = 1)

            # draw the line segments for this path
            canvas.stroke_styled_line_segments(
                points = segments,
                color  = [rgb_color],
            )

        # Separate fixed points from user points for marker styling.
        user_points  = points.copy()
        fixed_points = np.empty((0, 2))

        if fixed_head:
            fixed_points = np.vstack([fixed_points, points[0, :]])
            user_points  = user_points[1:, :]

        if fixed_tail:
            fixed_points = np.vstack([fixed_points, points[-1, :]])
            user_points  = user_points[:-1, :]

        # If fixed points, draw fixed points as squares.
        if fixed_points.shape[0] > 0:
            canvas.fill_styled_rects(
                x      = fixed_points[:, 0] - style["markersize"],
                y      = fixed_points[:, 1] - style["markersize"],
                width  = style["markersize"] * 2,
                height = style["markersize"] * 2,
                color  = [rgb_color],
            )

        # If user points, we can draw a circle for each point.
        if user_points.shape[0] > 0:
            canvas.fill_styled_circles(
                x      = user_points[:, 0],
                y      = user_points[:, 1],
                radius = style["markersize"],
                color  = [rgb_color],
            )

        # Draw the cursor indicator (larger circle around the active point).
        if cursor is not None:
            active_point = points[cursor, :]
            canvas.stroke_styled_circles(
                x      = [active_point[0]],
                y      = [active_point[1]],
                radius = (style["markersize"] + 1) * 4 / 3,
                color  = [rgb_color],
            )

        # TODO: add some arrow marker to indicate the direction of the next insert

    # Redraw Multicanvas Method ------------------------------------------------

    def redraw_canvas(self, image = False, active = True, background = False):
        """Redraw the canvas.
 
        This is the primary entry point called by ``FigurePanel``.
        Each flag controls whether the corresponding layer is redrawn.
 
        Parameters
        ----------
        image : bool, optional
            Redraw the grid image on the image canvas.  Defaults to ``False``.

        active : bool, optional
            Redraw the active annotation on the active annotation canvas.
            Defaults to ``True``.

        background : bool, optional
            Redraw the background annotations on the background annotation canvas.
            Defaults to ``False``.
        """
        # If there is no image to draw, skip
        if self.image is None: return

        # Redraw layers.
        with ipc.hold_canvas():
            # Redraw the image layer, if there is an image to draw.
            if image: self.redraw_image()

            # Redraw the annotation layers, if there are annotations to draw.
            if active or background:
                self.redraw_annotations(active = active, background = background)

    # Observer Methods ---------------------------------------------------------

    def observe_mouse(self, fn):
        """Register a callback for mouse clicks on the canvas.
 
        The callback receives figure coordinates (not canvas pixel
        coordinates).  Conversion is handled internally.
 
        Parameters
        ----------
        fn : callable
            ``fn(points)`` where *points* is ``ndarray, shape (1, 2)``
            in figure coordinates.
        """
        # Define a function to convert canvas pixel to figure coordinates. 
        # Then call the callback with the converted points.
        def _convert_xy_to_points(x, y):
            # Convert canvas pixel coordinates to figure coordinates.
            canvas_point = np.array([[x, y]]) # must be (N, 2) matrix
            figure_point = self.canvas_to_figure(canvas_point)
            fn(figure_point) # form: fn(points)
        
        # Expose the mouse event listener (with conversion!)
        self.multicanvas.on_mouse_down(_convert_xy_to_points)


    def observe_key(self, fn):
        """Register a callback for key presses on the canvas.
 
        Parameters
        ----------
        fn : callable
            ``fn(key, shift_down, ctrl_down, meta_down)`` where *key*
            is the key identifier string.
        """
        self.multicanvas.on_key_down(fn)