# -*- coding: utf-8 -*-
################################################################################
# annotate/_figure.py

"""
Core implementation code for the cortex-annotate tool's figure panel.
"""


# Imports ######################################################################

import numpy as np
import ipycanvas as ipc
import ipywidgets as ipw
import matplotlib as mpl
from traitlets import Int
from collections import defaultdict

from ._util import wrap as wordwrap

# The Figure Panel #############################################################

class FigurePanel(ipw.HBox):
    """The canvas that manages the display of figures and annotations.

    The `FigurePanel` is an subclass of `ipycanvas.MultiCanvas` that is designed
    to manage the display of images and annotations for the `AnnotationTool` in
    `_core.py`.
    """

    class LoadingContext:
        """A context manager for the loading screen on the figure panel canvas."""
        __slots__ = ( "canvas", "message" )

        _count = defaultdict(lambda: 0)

        def __init__(self, canvas, message = "Loading..."):
            self.canvas  = canvas
            self.message = message


        def __enter__(self):
            count = FigurePanel.LoadingContext._count
            idc   = id(self.canvas)
            c = count[idc]
            if c == 0:
                FigurePanel._draw_loading(self.canvas, self.message)
            count[idc] = c + 1 


        def __exit__(self, type, value, traceback):
            count = FigurePanel.LoadingContext._count
            idc = id(self.canvas)
            c = count[idc]
            c -= 1
            count[idc] = c
            if c == 0:
                self.canvas.clear()
                del count[idc]


    # A traitlet that increments whenever the annotations change.
    _annotation_change = Int(default_value = 0)


    def __init__(self, state):
        # Store the state.
        self.state = state
        
        # Store the annotation configuation information
        self.annot_cfg = state.config.annotations

        # Store the figure size (in pixels, cell in grid).  
        self.figure_size = np.array([256, 256]) # default value

        # Get first grid shape from first annotation in state
        annot0 = list(self.annot_cfg.types.keys())[0]
        self.grid_shape = self.annot_cfg.grid_shape[annot0]

        # Calculate the canvas size (in pixels) from the figure size and grid shape.
        self.canvas_size = self.figure_size * self.grid_shape

        # Make a multicanvas.
        canvas_width, canvas_height = self.canvas_size
        self.multicanvas = ipc.MultiCanvas(
            5, width = canvas_width, height = canvas_height)

        # We always seem to need to explicitly set the layout size in pixels.
        self.multicanvas.layout.width  = f"{canvas_width}px"
        self.multicanvas.layout.height = f"{canvas_height}px"

        # Separate out the canvas layers.
        self.image_canvas       = self.multicanvas[0] # grid image layer
        self.background_canvas  = self.multicanvas[1] # background annotation layer 
        self.active_canvas      = self.multicanvas[2] # active annotation layer
        self.loading_canvas     = self.multicanvas[3] # loading screen layer
        self.message_canvas     = self.multicanvas[4] # message layer (for errors, etc.)

        # Draw the loading screen on the loading canvas and save it as the loading context.
        self._draw_loading(self.loading_canvas)
        self.loading_canvas.save()
        self.loading_context = FigurePanel.LoadingContext(self.loading_canvas)

        # Set up our event observers for mouse clicks (to add points).
        self.multicanvas.on_mouse_down(self.on_mouse_click)

        # Set up our event observers for key presses (tab, delete).
        # self.multicanvas.on_key_down(self.on_key_press)

        # Initialize the image variables.
        self.image = None
        self.xlim  = None
        self.ylim  = None

        # Initialize the annotation variables.
        self.target      = None
        self.active      = None
        self.annotations = {}
        self.cursor      = 0

        # Initialize our parent class.
        super().__init__([ self._make_html_header(), self.multicanvas ])


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
        """Clears the image canvas and redraws the image."""
        with ipc.hold_canvas():
            # Clear the image canvas.
            self.image_canvas.clear()
            
            # Redraw the image
            self.image_canvas.draw_image(
                self.image, 0, 0, 
                self.image_canvas.width, 
                self.image_canvas.height
            )

    # Annotation Canvas Methods ------------------------------------------------

    def redraw_annotations(self, active = True, background = True):
        """Clears the annotation canvas and redraws all annotations."""
        # First, we clear the annotation canvases, depending on updates.
        if active: self.active_canvas.clear()
        if background: self.background_canvas.clear()

        # We step through all (visible) annotations and draw them.
        for (annotation, points) in self.annotations.items():
            # If there are no points, we can skip.
            if points is None or len(points) == 0: continue

            # The active annotaion is drawn on the active canvas, and all other
            # annotations are drawn on the background canvas. 
            if self.active == annotation:
                canvas   = self.active_canvas
                styletag = None 
            else:
                canvas   = self.background_canvas
                styletag = annotation
 
            # Get the style for this annotation.
            style = self.state.style(styletag) 

            # If this annotation isn't visible, we can skip it also.
            if not style["visible"]: continue
            
            # See if the boundary is closed and connected.
            atype = self.annot_cfg.types[annotation]
            # if atype == "point":
            #     (closed, joined) = (False, False)
            # elif atype == "contour":
            #     (closed, joined) = (False, True)
            # elif atype == "boundary":
            #     (closed, joined) = (True, True)
            # else:
            #     raise ValueError(f"Invalid annotation type: {atype}")
            
            # Okay, it needs to be drawn, so convert the figure points
            # into image coordinates.
            grid_points = self.figure_to_canvas(points)

            # For all the point-matrices here, we need to draw them.
            for points in grid_points:
                self.draw_points(
                    canvas, points, style, 
                    cursor = 0,
                    closed = False, 
                    fixed_head = False, 
                    fixed_tail = False, 
                )
    

    
    def _apply_linestyle(self, canvas, style):
        """Applies the given line width and line style to the given canvas."""
        # Get the line width and line style from the style dict, with defaults.
        linewidth = style.get("linewidth", 1)
        linestyle = style.get("linestyle", "solid")
        
        # Apply the line width and line style to the canvas.
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

        # if fixed:
        #     ms = style["markersize"]
        #     canvas.fill_rect(x-ms, y-ms, ms*2, ms*2)

        # Convert the color from the style into an RGB.
        rgb_color = mpl.colors.to_rgb(style["color"])
        rgb_color = np.array(rgb_color * 255, dtype = np.uint8)
        self._apply_linestyle(canvas, style)

        # if there is more than one point, we can draw a line segment
        if points.shape[0] > 1: 
            # if the path is closed, we need to add the first point to the end 
            # of the point matrix to make sure the path is closed when we draw it.
            if closed: points = np.vstack([points, points[0:1, :]])

            # create segement coordinates pairs [(x1, y1), (x2, y2), ...] 
            segments = np.stack([points[:-1,:], points[1:,:]], axis = 1)

            # draw the line segments for this path
            canvas.stroke_styled_line_segments(
                points = segments,
                color  = [rgb_color],  
            )
        
        # if there is at least one point, we can draw a circle for each point
        canvas.fill_styled_circles(
            x = points[:,0], 
            y = points[:,1], 
            radius = style["markersize"],
            color  = [rgb_color],
        )
        
        if cursor is not None: 
            active_point = points[cursor, :]
            canvas.stroke_styled_circles(
                x = [active_point[0],], 
                y = [active_point[1],], 
                radius = (style["markersize"] + 1) * 4/3, 
                color  = [rgb_color],
            )

    # Loading Canvas Methods ---------------------------------------------------

    @staticmethod
    def _prep_canvas_message(canvas, message, wrap = True, fontsize = 32):
        """Prepares a message for drawing on the given canvas."""
        # Prepare the message by word wrapping, if necessary.
        if wrap is True or wrap is Ellipsis:
            wrap = int(canvas.width * 13/15 / fontsize * 2)
        message = wordwrap(message, wrap = wrap)
    
        # Calculate the x0, y0, and max_width for the message.
        x0 = canvas.width // 15
        y0 = canvas.height // 15
        max_width = canvas.width - (canvas.width // 15 * 2)
        
        # Return the prepared message and the x0, y0, and max_width for drawing it.
        return message, x0, y0, max_width
    

    @staticmethod
    def _draw_text_canvas(canvas, message, wrap = True, fontsize = 32):
        """Draws a message on the given canvas."""
        # Prepare the message by word wrapping, if necessary.
        message, x0, y0, max_width = FigurePanel._prep_canvas_message(
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

            # Draw the message on the canvas.
            for (i, line) in enumerate(message.split("\n")):
                canvas.fill_text(
                    text = line, x = x0, y = y0 + fontsize * i, 
                    max_width = max_width
                )


    @classmethod
    def _draw_loading(cls, canvas, message = "Loading...", wrap = True, fontsize = 32):
        """Clears the canvas and draws the loading screen."""
        cls._draw_text_canvas(
            canvas   = canvas, 
            message  = message, 
            wrap     = wrap, 
            fontsize = fontsize
        )

    # Message Canvas Methods ---------------------------------------------------

    def write_message(self, message, wrap = True, fontsize = 32):
        """Writes a message on the message canvas."""
        self._draw_text_canvas(
            canvas   = self.message_canvas, 
            message  = message, 
            wrap     = wrap, 
            fontsize = fontsize
        )
  

    def clear_message(self):
        """Clears the current message canvas."""
        self.message_canvas.clear()

    # Update State Method ------------------------------------------------------

    def update_state(self, target_id, annotation, target_annotations):
        """Updates the state to reflect the given target and annotation."""
        # If the target is different from current state.
        if self.target != target_id or self.active != annotation:
            # Update the target and active annotation.
            self.target = target_id
            self.active = annotation

            # Get the data for the given target and annotation.
            (image_data, grid_shape, meta_data) = self.state.grid(
                self.target, self.active)
            
            # Update the image, grid shape, and meta data.
            self.image      = ipw.Image(value = image_data, format = "png")
            self.grid_shape = grid_shape
            self.xlim       = meta_data["xlim"]
            self.ylim       = meta_data["ylim"]

            # Update the annotations for the given target and annotation.
            self.annotations = target_annotations

    # Canvas Resizing Method ---------------------------------------------------

    def resize_canvas(self, new_figure_size = None):
        """Resizes the figure canvas so that images appear at the given scale.

        `figure_panel.resize_canvas(new_figure_size)` results in the canvas being
        resized to match the new figure size. Note that this does not resize the
        canvas to have a width of `new_figure_size` but rather resizes it so that each
        image in the grid has a width of `new_figure_size`.

        The `resize_canvas` method triggers a redraw because the resizing of the
        canvas clears it.
        """
        # If there is no new_figure_size give, we just use the current figure size.
        if new_figure_size is None:
            new_figure_size = self.state.figure_size()

        # Set the new figure size.
        self.figure_size = np.array([new_figure_size, new_figure_size])

        # The canvas size is a product of the figure size and the grid shape.
        self.canvas_size = self.figure_size * np.array(self.grid_shape)
        canvas_width, canvas_height = self.canvas_size.astype(int)

        # First resize the canvas (this clears it).
        self.multicanvas.width  = canvas_width
        self.multicanvas.height = canvas_height

        # Then we also resize the layout component.
        self.multicanvas.layout.width  = f"{canvas_width}px"
        self.multicanvas.layout.height = f"{canvas_height}px"

        # Finally, because the canvas was cleared upon resize, we redraw it.
        self.redraw_canvas()
    
    # Redraw Mulicanvas Method -------------------------------------------------

    def redraw_canvas(
            self, image = None, grid_shape = None, xlim = None, ylim = None,
            redraw_image = True, redraw_annotations = True
        ):
        """Redraws the entire canvas.

        `figure_panel.redraw_canvas()` redraws the canvas as-is.

        `figure_panel.redraw_canvas(new_image)` redraws the canvas with the
        new image; this requires that the grid has not changed.

        `figre_panel.redraw_canvas(new_image, new_grid_shape)` redraws the
        canvas with the given new image and new grid shape.

        The optional arguments `redraw_image` and `redraw_annotations` both
        default to `True`. They can be set to `False` to skip the redrawing of
        one or the other layer of the canvas.
        """
        # If no image give, redraw the current image.
        if image is None:
            image = self.image
        else: # If an image is given, we update the current image.
            self.image = image

        # Update the xlim and ylim if given.
        if xlim is not None:
            self.xlim = xlim
        if ylim is not None:
            self.ylim = ylim

        # If no grid shape is given, redraw with the current grid shape.
        if grid_shape is None:
            grid_shape = self.grid_shape
        elif grid_shape != self.grid_shape:
            # If grid shape is given and different from current grid shape, we
            # update and resize the canvas, which will trigger another redraw, 
            # so we return here to avoid doing a redudant redraw. 
            self.grid_shape = grid_shape
            self.resize_canvas()
            return

        # Redraw the loading canvas (assuming one was given).
        if redraw_image or redraw_annotations:
            self.loading_canvas.restore()
        
        # Redraw the image and annotations, if necessary.
        with ipc.hold_canvas():
            if redraw_image: 
                self.redraw_image()
            if redraw_annotations: 
                self.redraw_annotations()
    

    # Canvas to Figure Coordinate Conversion Method ----------------------------

    def canvas_to_figure(self, points):
        """Converts the `N x 2` matrix of canvas points into figure coordinates."""
        # Check the shape of the input and convert it into an `N x 2` matrix if necessary.
        points = np.asarray(points)
        if len(points.shape) == 1:
            return self.canvas_to_figure([points])[0]
        
        # First off, we want to apply the grid mod to make sure that any points 
        # that are outside the figure limits get wrapped around to the location.
        (figure_width, figure_height) = self.figure_size
        points = points % [ figure_width, figure_height ]
        
        # Get the figure limits.
        xlim = (0, figure_width) if self.xlim is None else self.xlim
        ylim = (0, figure_height) if self.ylim is None else self.ylim
        
        # We need to invert the y axis.
        points[:,1] = figure_height - points[:,1]

        # Now, make the conversion.
        points *= [(xlim[1] - xlim[0]) / figure_width,
                   (ylim[1] - ylim[0]) / figure_height]
        points += [xlim[0], ylim[0]]

        # Return the converted points.
        return points

        
    def figure_to_canvas(self, points):
        """Converts the `N x 2` matrix of figure points into canvas coordinates."""
        # Check the shape of the input and convert it into an `N x 2` matrix if necessary.
        points = np.asarray(points)
        if len(points.shape) == 1:
            return self.figure_to_canvas([points])[0]
        # Get the figure limits.
        (figure_width, figure_height) = self.figure_size
        xlim = (0, figure_width) if self.xlim is None else self.xlim
        ylim = (0, figure_height) if self.ylim is None else self.ylim

        # First, make the basic conversion.
        points  = points - [xlim[0], ylim[0]]
        points *= [figure_width / (xlim[1] - xlim[0]),
                   figure_height / (ylim[1] - ylim[0])]
        
        # Then invert the y-axis
        points[:,1] = figure_height - points[:,1]

        # And build up the point matrices for each grid element.
        (rows, cols) = self.grid_shape
        return [
            points + [ii * figure_width, jj * figure_height]
            for ii in np.arange(cols)
            for jj in np.arange(rows)
        ]

    # Mouse Event Handler Methods ----------------------------------------------

    @staticmethod
    def _to_point_matrix(x, y = None):
        x = np.asarray(x) if y is None else np.array([[x, y]])
        if x.shape == (2,):
            x = x[None,:]
        elif x.shape != (1,2):
            raise ValueError(f"Bad point shape: {x.shape}")
        return x


    @staticmethod
    def _empty_point_matrix():
        return np.zeros((0, 2), dtype = float)


    def _recalculate_deps(self, annotation):
        """Recalculates the dependent annotations for the given annotation."""
        # Get the dependent annotations for this annotation.
        dependent_annotations = [
            key for fixed in (self.annot_cfg.fixed_head, self.annot_cfg.fixed_tail)
            for key, value in fixed.items() if annotation == value
        ]

        # If there are no dependent annotations, we can skip.
        if len(dependent_annotations) == 0: return False
    
        # Otherwise, we need to recalculate the dependent annotations.
        print("Recalculating dependent annotations for:", annotation)
        for annot in dependent_annotations:
            print("  -> Dependent annotation:", annot)
            current_points = self.annotations.get(annot)

            # if current_points is None or len(current_points) == 0:
                # fixed = self.annot_cfg.fixed_head.get(annot) or self.annot_cfg.fixed_tail.get(annot)


        # Return True to indicate that the background needs to be redrawn.
        return True
    
    
    def _increment_annotation_change(self):
        """Increments the annotation change traitlet after redraw triggers."""
        self._annotation_change += 1


    def push_point(self, x, y = None, redraw = True):
        """Push the given point onto the path at the cursor end.

        The point may be given as `x, y` or as a vector or 1 x 2 matrix. The
        point is added to the head or the tail depending on the cursor.
        """
        # We can only push points if there is an active annotation.
        if self.active is None: return None

        # First convert the input into a point matrix, must be N x 2.
        new_point = FigurePanel._to_point_matrix(x, y)

        # Get the current points for this annotation. If None, initialize empty.
        points = self.annotations.get(self.active)
        if points is None: points = self._empty_point_matrix()
        print("Current points:", points)
        print(points.shape)

        # Get the annotation type for this annotation.
        atype = self.annot_cfg.types[self.active]
        print(self.active)
        print("Annotation type:", atype)
        
        # Depending on the annotation type, we add the newest point to the
        # annotation in different ways.
        if atype == "point":
            # For a point annotation, replace the current point with the new point.
            # There will never be a fixed head or tail for a point annotation.
            points = new_point 
        elif atype == "contour":
            # For a contour annotation, we add the point to after the current
            # indexed position.
            pass
        elif atype == "boundary":
            # For a boundary annotation, we add the point to the head or tail
            # depending on the cursor position, but we also need to make sure that
            # the boundary is closed, so we need to add the point to both ends.
            pass
 
        # Update the annotation with the new points.
        self.annotations[self.active] = points

        # Update dependent annotations, if necessary.
        redraw_background = self._recalculate_deps(self.active)

        # Redraw the annotations.
        if redraw:
            self.redraw_annotations(background = redraw_background)
            self._increment_annotation_change()


    def _push_impoint(self, x, y = None, redraw = True):
        """Push the given canvas point onto the selected annotation.

        The point may be given as `x, y` or as a vector or 1 x 2 matrix. Canvas
        points are always converted into figure points before being appended to
        the annotation. The point is added to the head or the tail depending on
        the cursor.
        """
        # First convert the input into a point matrix, must be N x 2.
        x = FigurePanel._to_point_matrix(x, y)

        # Convert to a figure point.
        x = self.canvas_to_figure(x)

        # And then push it onto the annotation.
        return self.push_point(x, redraw = redraw)
    

    def on_mouse_click(self, x, y):
        """This method is called when the mouse is clicked on the canvas."""        
        # Add to the current contour.
        self._push_impoint(x, y)
    
    # Key Press Event Handler Methods ------------------------------------------

    def toggle_cursor(self):
        """Toggles the cursor position between head/tail."""
        orig = self.cursor_position
        if orig == "tail":
            self.cursor_position = "head"
        else:
            self.cursor_position = "tail"
        self.redraw_annotations(background = False)
        return self.cursor_position

    
    def pop_point(self, redraw = True):
        if self.active is None:
            # We got a backspace while not accepting edits; ignore it.
            return None
        # Get the current points.
        points = self.annotations.get(self.active)
        if points is None or len(points) == 0:
            # No points to pop!
            return None
        deps = self.annotation_deps[self.active]
        hasdeps = len(deps) > 0
        if len(points) == 1 and hasdeps:
            # Can't pop because something depends on this point!
            return None
        fh = self.fixed_head(self.active)
        ft = self.fixed_tail(self.active)
        if fh is None:
            fh = np.zeros((0,2), dtype=float)
            fhq = False
        else:
            fh = points[[0]]
            points = points[1:]
            fhq = True
        if ft is None:
            ft = np.zeros((0,2), dtype=float)
            ftq = False
        else:
            ft = points[[-1]]
            points = points[:-1]
            ftq = True
        if len(points) < 2:
            if len(points) == 0:
                import warnings
                warnings.warn(
                    "Current annotation contains only fixed points. This could"
                    " indicate a corrupted save file. Discarding this"
                    " annotation.")
            self.annotations[self.active] = None
        else:
            if self.cursor_position == "head":
                points = points[1:]
            else:
                points = points[:-1]
            points = np.vstack([fh, points, ft])
            self.annotations[self.active] = points
        # Update dependant annotations.
        for dep in deps:
            self._recalc_ends(dep)
        # Redraw the annotations.
        if redraw:
            self.redraw_annotations(background = hasdeps)
            self._increment_annotation_change()


    def on_key_press(self, key, shift_down, ctrl_down, meta_down):
        """This method a key is pressed."""
        # Handle the key press.
        key = key.lower()
        if key == "tab":
            # Toggle the cursor (active) position.
            self.toggle_cursor()
        elif key == "backspace":
            # Delete from head/tail, wherever the cursor is.
            self.pop_point()
        else:
            pass
