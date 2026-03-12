# -*- coding: utf-8 -*-
################################################################################
# annotate/_canvas.py

"""
Implementation code for the 2D Canvas Panel.

The CanvasPanel is a pure rendering widget. It reads all annotation state from
a shared `FigurePanelState` object and redraws its ipycanvas layers when 
notified of changes. User input events (mouse clicks, key presses) are 
translated into figure-coordinate mutations and forwarded to FigurePanelState.
"""


# Imports ######################################################################

import numpy as np
import ipycanvas as ipc
import ipywidgets as ipw
import matplotlib as mpl
from traitlets import Int
from collections import defaultdict

from ._util import wrap as wordwrap


# The Canvas Panel #############################################################

class CanvasPanel(ipw.HBox):
    """The 2D canvas that displays figure images and annotation overlays.

    The CanvasPanel manages a multi-layer ipycanvas for rendering:
        Layer 0: grid image
        Layer 1: background annotations (non-active)
        Layer 2: active annotation (with cursor)
        Layer 3: loading screen
        Layer 4: messages (errors, warnings)
    """

    class LoadingContext:
        """A context manager for the loading screen on the figure panel canvas."""
        __slots__ = ("canvas", "message")

        _count = defaultdict(lambda: 0)

        def __init__(self, canvas, message = "Loading..."):
            self.canvas  = canvas
            self.message = message

        def __enter__(self):
            count = CanvasPanel.LoadingContext._count
            idc   = id(self.canvas)
            c = count[idc]
            if c == 0:
                CanvasPanel._draw_loading(self.canvas, self.message)
            count[idc] = c + 1

        def __exit__(self, type, value, traceback):
            count = CanvasPanel.LoadingContext._count
            idc = id(self.canvas)
            c = count[idc]
            c -= 1
            count[idc] = c
            if c == 0:
                self.canvas.clear()
                del count[idc]

    # A traitlet that increments whenever the annotations change.
    _annotation_change = Int(default_value = 0)

    def __init__(self, figure_state, figure_size = 256):
        """Initialize the canvas panel."""
        # Store figure state.
        self.state     = figure_state 
        self.annot_cfg = figure_state.annot_cfg

        # Store the figure size (in pixels, cell in grid). 
        self.figure_size = np.array([figure_size, figure_size])

        # Get first grid shape from first annotation in state
        grid_shape0 = self.annot_cfg.grid_shape[self.annot_cfg.names[0]]

        # Calculate the canvas size (in pixels) from figure size and grid shape.
        self.canvas_size = self.figure_size * grid_shape0

        # Make a multicanvas.
        canvas_width, canvas_height = self.canvas_size
        self.multicanvas = ipc.MultiCanvas(
            5, width = canvas_width, height = canvas_height)
        
        # We always seem to need to explicitly set the layout size in pixels.
        self.multicanvas.layout.width  = f"{canvas_width}px"
        self.multicanvas.layout.height = f"{canvas_height}px"

        # Separate out the canvas layers.
        self.image_canvas      = self.multicanvas[0]  # grid image layer
        self.background_canvas = self.multicanvas[1]  # background annotation layer
        self.active_canvas     = self.multicanvas[2]  # active annotation layer
        self.loading_canvas    = self.multicanvas[3]  # loading screen layer
        self.message_canvas    = self.multicanvas[4]  # message layer

        # Draw the loading screen and save it as the loading context.
        self._draw_loading(self.loading_canvas)
        self.loading_canvas.save()
        self.loading_context = CanvasPanel.LoadingContext(self.loading_canvas)

        # Initialize the HBox.
        super().__init__(
            children = [ self._make_html_header(), self.multicanvas ],
            layout   = { "border": "1px solid blue" }
        )


    @classmethod
    def _make_html_header(cls):
        return ipw.HTML(f"""
            <style>
                canvas {{
                    cursor: crosshair !important;
                }}
            </style>
        """)

    # Image Canvas Methods -----------------------------------------------------

    def redraw_image(self):
        """Clear the image canvas and redraw the grid image."""
        with ipc.hold_canvas():
            self.image_canvas.clear()
            self.image_canvas.draw_image(
                self.state.canvas.image, 0, 0,
                self.image_canvas.width,
                self.image_canvas.height
            )


    # Canvas to Figure Coordinate Conversion -----------------------------------

    def canvas_to_figure(self, points):
        """Convert an (N, 2) matrix of canvas pixel coordinates to figure coordinates."""
        # Check the shape of the input and convert it into an `N x 2` matrix if necessary.
        points = np.asarray(points)
        if len(points.shape) == 1:
            return self.canvas_to_figure([points])[0]

        # Apply grid mod to wrap points into a single cell.
        (figure_width, figure_height) = self.figure_size
        points = points % [figure_width, figure_height]

        # Get figure limits.
        xlim, ylim = self.state.canvas.xlim, self.state.canvas.ylim
        xlim = (0, figure_width)  if xlim is None else xlim
        ylim = (0, figure_height) if ylim is None else ylim

        # We need to invert the y axis.
        points[:, 1] = figure_height - points[:, 1]

        # Now, make the conversion.
        points *= [(xlim[1] - xlim[0]) / figure_width,
                   (ylim[1] - ylim[0]) / figure_height]
        points += [xlim[0], ylim[0]]

        # Return the converted points.
        return points


    def figure_to_canvas(self, points):
        """Convert an (N, 2) matrix of figure coordinates to canvas pixel coordinates.

        Returns a list of (N, 2) matrices, one per non-None cell in the grid.
        """
        # Check the shape of the input and convert it into an `N x 2` matrix if necessary.
        points = np.asarray(points)
        if len(points.shape) == 1: 
            return self.figure_to_canvas([points])[0]

        # Get the figure limits.
        (figure_width, figure_height) = self.figure_size
        xlim, ylim = self.state.canvas.xlim, self.state.canvas.ylim
        xlim = (0, figure_width)  if xlim is None else xlim
        ylim = (0, figure_height) if ylim is None else ylim

        # Scale to pixel coordinates.
        points  = points - [xlim[0], ylim[0]]
        points *= [figure_width  / (xlim[1] - xlim[0]),
                figure_height / (ylim[1] - ylim[0])]

        # Invert the y axis.
        points[:, 1] = figure_height - points[:, 1]

        # And build up the point matrices for each (not None) grid element.
        (n_rows, n_cols) = self.state.canvas.grid_shape
        return [
            points + [ii * figure_width, jj * figure_height]
            for ii in np.arange(n_cols)
            for jj in np.arange(n_rows)
            if self.state.canvas.grid[jj][ii] is not None
        ]


    # Annotation Canvas Methods ------------------------------------------------

    def redraw_annotations(self, active = True, background = True):
        """Clear and redraw annotation overlays."""
        # Clear the appropriate canvas layers.
        if active:     self.active_canvas.clear()
        if background: self.background_canvas.clear()

        # Step through all annotations and draw them.
        for (annotation, points) in self.state.annotations.items():
            # If there are no points, we can skip.
            if points is None or len(points) == 0:
                continue

            # Determine if this is the active or a background annotation.
            if self.state.active == annotation:
                # Skip active annotation if active is False.
                if not active: continue
                canvas   = self.active_canvas
                styletag = None
                cursor   = self.state.cursor
            else:
                # Skip background annotations if background is False.
                if not background: continue
                canvas   = self.background_canvas
                styletag = annotation
                cursor   = None

            # Determine if the head or tail of this annotation is fixed.
            fixed_head = self.state.fixed_heads.get(annotation, None) is not None
            fixed_tail = self.state.fixed_tails.get(annotation, None) is not None

            # Get the style for this annotation.
            style = self.state.canvas.style(styletag)

            # Skip, if the annotation is not visible.
            if not style["visible"]: continue

            # Check if the path is closed (only boundaries are closed).
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
                    fixed_tail = fixed_tail
                )


    def _apply_linestyle(self, canvas, style):
        """Apply the line width and line style to the given canvas."""
        # Get the line width and line style from the style dict, with defaults.
        linewidth, linestyle = style["linewidth"], style["linestyle"]

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
        fixed_head = False, fixed_tail = False
    ):
        """Draw an annotation path on the given canvas.

        Points must be in canvas pixel coordinates. The ``style`` dict
        provides color, linewidth, linestyle, and markersize.
        """
        # Convert the color to an RGB array.
        rgb_color = np.array(mpl.colors.to_rgb(style["color"]))
        rgb_color = np.array(rgb_color * 255, dtype = np.uint8)

        # Apply linewidth and linestyle.
        self._apply_linestyle(canvas, style)

        # We only draw line segments if there are at least two points. If the
        # annotation has fixed points, we only draw when there are more points 
        # than fixed points.
        if points.shape[0] > np.max([1, np.sum([fixed_head, fixed_tail])]):
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

        # Separate fixed points from user points for different marker styles.
        user_points  = points.copy()
        fixed_points = self.state.empty_point_matrix()

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

    # Loading Canvas Methods ---------------------------------------------------

    @staticmethod
    def _prep_canvas_message(canvas, message, wrap = True, fontsize = 32):
        """Prepare a message for drawing on the given canvas."""
        # Prepare the message by word wrapping, if necessary.
        if wrap is True or wrap is Ellipsis:
            wrap = int(canvas.width * 13 / 15 / fontsize * 2)
        message = wordwrap(message, wrap = wrap)

        # Calculate the x0, y0, and max_width for the canvas message.
        x0        = canvas.width // 15
        y0        = canvas.height // 15
        max_width = canvas.width - (canvas.width // 15 * 2)

        # Return the prepared message and the x0, y0, and max_width for drawing it.
        return message, x0, y0, max_width


    @staticmethod
    def _draw_text_canvas(canvas, message, wrap = True, fontsize = 32):
        """Draw a message on the given canvas."""
        # Prepare the message by word wrapping, if necessary.
        message, x0, y0, max_width = CanvasPanel._prep_canvas_message(
            canvas, message, wrap = wrap, fontsize = fontsize)

        with ipc.hold_canvas():
            # Clear the canvas.
            canvas.clear()

            # Draw a white background with some transparency.
            canvas.fill_style   = "white"
            canvas.global_alpha = 0.85
            canvas.fill_rect(0, 0, canvas.width, canvas.height)

            # Draw the message in black.
            canvas.fill_style    = "black"
            canvas.global_alpha  = 1
            canvas.font          = f"{fontsize}px HelveticaNeue"
            canvas.text_align    = "left"
            canvas.text_baseline = "top"

            # Draw the message on the canvas, line by line.
            for (i, line) in enumerate(message.split("\n")):
                canvas.fill_text(
                    text = line, x = x0, y = y0 + fontsize * i,
                    max_width = max_width
                )


    @classmethod
    def _draw_loading(cls, canvas, message = "Loading...", wrap = True, fontsize = 32):
        """Clear the canvas and draw the loading screen."""
        cls._draw_text_canvas(
            canvas   = canvas,
            message  = message,
            wrap     = wrap,
            fontsize = fontsize
        )

    # Message Canvas Methods ---------------------------------------------------

    def write_message(self, message, wrap = True, fontsize = 32):
        """Write a message on the message canvas."""
        self._draw_text_canvas(
            canvas   = self.message_canvas,
            message  = message,
            wrap     = wrap,
            fontsize = fontsize
        )


    def clear_message(self):
        """Clear the current message canvas."""
        self.message_canvas.clear()
        """Increments the annotation change traitlet after redraw triggers."""
        self._annotation_change += 1