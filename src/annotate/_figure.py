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

        # Store the figure size (in pixels, cell in grid) from the config.         
        self.base_size   = np.array(state.config.display.image_size)
        self.figure_size = self.base_size # use as initial
        self.canvas_size = self.base_size # use as initial

        # Make a multicanvas.
        canvas_width, canvas_height = self.canvas_size
        self.multicanvas = ipc.MultiCanvas(5, width = canvas_width, height = canvas_height)

        # We always seem to need to explicitly set the layout size in pixels.
        self.multicanvas.layout.width  = f"{canvas_width}px"
        self.multicanvas.layout.height = f"{canvas_height}px"

        # Separate out the canvas layers.
        self.image_canvas       = self.multicanvas[0] # grid image layer
        self.builtin_canvas     = self.multicanvas[1] # builtin annotations layer
        self.annotations_canvas = self.multicanvas[2] # annotations layer
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

        # We start out with nothing drawn initially.
        self.image      = None
        self.grid_shape = (1, 1)
        self.foreground = None
        self.xlim = None
        self.ylim = None
        self.annotations = {}
        # self.builtin_annotations = {}
        self.cursor_position = "tail"
        self.fixed_heads = None
        self.fixed_tails = None
        self.annotation_types = {}
        self.ignore_input = False

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

    # Canvas Resizing Method ---------------------------------------------------

    def resize_canvas(self, new_scale = None):
        """Resizes the figure canvas so that images appear at the given scale.

        `figure_panel.resize_canvas(new_scale)` results in the canvas being
        resized to match the new image scale. Note that this does not resize the
        canvas to have a width of `new_scale` but rather resizes it so that each
        image in the grid has a width of `new_scale`.

        The `resize_canvas` method triggers a redraw because the resizing of the
        canvas clears it.
        """
        # If there is no new_scale give, we just use the current image scale.
        #TODO: might want to remove the new_scale argument...
        if new_scale is None:
            new_scale = self.state.image_scale()

        # Calculate the new figure size.
        self.figure_size = np.array(self.base_size * new_scale).astype(int)

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
    
    # Redraw Mulicanvas Method ---------------------------------------------------

    def redraw_canvas(
            self, 
            image = None, 
            grid_shape = None, 
            xlim = None, 
            ylim = None,
            redraw_image = True, 
            redraw_annotations = True
        ):
        """Redraws the entire canvas.

        `figure_panel.redraw_canvas()` redraws the canvas as-is.

        `figure_panel.redraw_canvas(new_image)` redraws the canvas with the new
        image; this requires that the grid has not changed.

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


    # -----------------------------------------
    
    # TODO: redo background aka builtin annotations, set to false for now
    def redraw_annotations(self, foreground = True, background = False):
        """Clears the draw canvas and redraws all annotations."""
        if background: self.builtin_canvas.clear()
        if foreground: self.annotations_canvas.clear()
        # We step through all (visible) annotations and draw them.
        for (annotation, points) in self.annotations.items():
            # If annotation is the foreground, we use None as the style tag.
            # We also draw on the foreground canvas instead of the background.
            if annotation == self.foreground:
                if not foreground: continue
                styletag = None
                canvas = self.annotations_canvas
                cursor = self.cursor_position
            else:
                if not background: continue
                styletag = annotation
                canvas = self.builtin_canvas
                cursor = None
            # If there are no points, we can skip.
            if points is None or len(points) == 0: continue
            # If this annotation isn't visible, we can skip it also.
            style = self.state.style(styletag)
            if not style["visible"]: continue
            # Grab the fixed head and tail statuses.
            fh = self.fixed_head(annotation) is not None
            ft = self.fixed_tail(annotation) is not None
            # See if the boundary is closed and connected.
            atype = self.annotation_type(annotation)
            if atype in ("point", "points"):
                (closed, joined) = (False, False)
            elif atype in ("path", "contour", "paths", "contours"):
                (closed, joined) = (False, True)
            elif atype in ("boundary", "boundaries", "loop", "loops"):
                (closed, joined) = (True, True)
            else:
                raise ValueError(f"invalid annotation type: {atype}")
            # Okay, it needs to be drawn, so convert the figure points
            # into image coordinates.
            grid_points = self.figure_to_image(points)
            # For all the point-matrices here, we need to draw them.
            for pts in grid_points:
                self.state.draw_path(
                    styletag, pts, canvas,
                    fixed_head = fh, fixed_tail = ft, cursor = cursor,
                    closed = closed, path = joined)
                
        # Next, we step through all the (visible) builtin annotations.
        # if background:
        #     for (annotation_name, dat) in self.builtin_annotations.items():
        #         if dat is None: continue
        #         style = self.state.style(annotation_name)
        #         if not style["visible"]: continue
        #         points_list = dat.get_data()
        #         for points in points_list:
        #             grid_points = self.figure_to_image(points)
        #             for pts in grid_points:
        #                 self.state.draw_path(
        #                     annotation_name, pts, self.builtin_canvas, path = False)
        # That's it.
    
    def _increment_annotation_change(self):
        self._annotation_change += 1
    
    
    def change_annotations(
            self, annots, builtin_annots, redraw = True, allow = True,
            fixed_heads = None, fixed_tails = None, annotation_types = None,
            target = None
        ):
        """Changes the set of currently visible annotations.

        The argument `annots` must be a dictionary whose keys are the annotation
        names and whose values are the `N x 2` matrices of annotation points, in
        figure coordinates. The optional argument `fixed_heads` may be a
        `dict`-like object whose keys are annotation names and whose values are
        the `(x,y)` coordinates of the fixed head position for that particular
        annotation.
        """
        self.annotations = annots
        self.builtin_annotations = builtin_annots
        self.fixed_heads = fixed_heads
        self.fixed_tails = fixed_tails
        self.annotation_types = annotation_types
        self.target = target
        if redraw:
            self.redraw_annotations()
        self.ignore_input = not allow
    
    
    def change_foreground(self, annot, redraw = True):
        """Changes the foreground annotation (the annotation being edited).

        `figure_panel.change_foreground(annot)` changes the current foreground
        annotation to the annotation with the name `annot`. The foreground
        annotation is the annotation that is currently being edited by the user.
        """
        self.foreground = annot
        if redraw:
            self.redraw_annotations()
    
    
    def fixed_head(self, annot = None):
        """Returns the 2D fixed-head point for the given annotation or `None`."""
        if self.fixed_heads is None: return None
        if annot is None: annot = self.foreground
        pt = self.fixed_heads.get(annot)
        if pt is None: return None
        if len(np.shape(pt)) != 1: return None
        if len(pt) != 2: return None
        if np.isfinite(pt).sum() != 2: return None
        return pt
    
    
    def fixed_tail(self, annot = None):
        """Returns the 2D fixed-tail point for the given annotation or `None`."""
        if self.fixed_tails is None: return None
        if annot is None: annot = self.foreground
        pt = self.fixed_tails.get(annot)
        if pt is None: return None
        if len(np.shape(pt)) != 1: return None
        if len(pt) != 2: return None
        if np.isfinite(pt).sum() != 2: return None
        return pt
    
    
    def annotation_type(self, annot = None):
        """Returns the annotation type of the given annotation."""
        if self.annotation_types is None: return "points"
        if annot is None: annot = self.foreground
        at = self.annotation_types.get(annot)
        return "points" if at is None else at


    def _recalc_ends(self, annot):
        """Recalculates the fixed head and fixed tail for the point and updates
        the annotations."""
        points = self.annotations.get(annot)
        if points is None or len(points) == 0:
            return
        (fh, ft) = self.state._calc_fixed_ends(annot, targ=self.target)
        stack = []
        if fh is not None:
            stack.append(fh)
        else:
            stack.append(points[0])
        stack.append(points[1:-1])
        if ft is not None:
            stack.append(ft)
        else:
            stack.append(points[-1])
        points = np.vstack(stack)
        self.annotations[annot] = points
    

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
    
    #TODO: this is a bit of a mess.
    def push_point(self, x, y = None, redraw = True):
        """Push the given image point onto the path at the cursor end.

        The point may be given as `x, y` or as a vector or 1 x 2 matrix. The
        point is added to the head or the tail depending on the cursor.
        """
        if self.foreground is None:
            # We got a click while not accepting clicks. Just ignore it.
            return None
        x = FigurePanel._to_point_matrix(x, y)
        # Add it on!
        points = self.annotations.get(self.foreground)
        if points is None:
            points = np.zeros((0,2), dtype=float)
        # We'll need to know the fixed head and tail conditions.
        fh = self.fixed_head(self.foreground)
        ft = self.fixed_tail(self.foreground)
        at = self.annotation_type(self.foreground)
        # How/where we add the point depends partly on whether there are points
        # and what the fixed head/tail state is.
        if len(points) == 0:
            # If this is the first point, we add the fixed points as well.
            fh = np.zeros((0,2),dtype=float) if fh is None else fh[None,:]
            ft = np.zeros((0,2),dtype=float) if ft is None else ft[None,:]
            points = np.vstack([fh, x, ft])
        else:
            if fh is None:
                fh = np.zeros((0,2), dtype=float)
            else:
                fh = points[[0]]
                points = points[1:]
            if ft is None:
                ft = np.zeros((0,2), dtype=float)
            else:
                ft = points[[-1]]
                points = points[:-1]
            # Where we add depends on the cursor position.
            if at in ( "point", "points" ):
                points = np.reshape(x, (1,2))
            elif self.cursor_position == "head":
                points = np.vstack([fh, x, points, ft])
            else:
                points = np.vstack([fh, points, x, ft])
        self.annotations[self.foreground] = points
        # Anyone who depends on us needs to be potentially redrawn.
        deps = self.state.config.fixed_deps[self.foreground]
        for dep in deps:
            self._recalc_ends(dep)
        # Redraw the annotations.
        if redraw:
            self.redraw_annotations(background = (len(deps) > 0))
            self._increment_annotation_change()
        

    @staticmethod
    def _to_point_matrix(x, y = None):
        x = np.asarray(x) if y is None else np.array([[x, y]])
        if x.shape == (2,):
            x = x[None,:]
        elif x.shape != (1,2):
            raise ValueError(f"Bad point shape: {x.shape}")
        return x


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
        # If we're ignoring input, just ignore it.
        if self.ignore_input: return
        
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
        if self.foreground is None:
            # We got a backspace while not accepting edits; ignore it.
            return None
        # Get the current points.
        points = self.annotations.get(self.foreground)
        if points is None or len(points) == 0:
            # No points to pop!
            return None
        deps = self.state.config.fixed_deps[self.foreground]
        hasdeps = len(deps) > 0
        if len(points) == 1 and hasdeps:
            # Can't pop because something depends on this point!
            return None
        fh = self.fixed_head(self.foreground)
        ft = self.fixed_tail(self.foreground)
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
            self.annotations[self.foreground] = None
        else:
            if self.cursor_position == "head":
                points = points[1:]
            else:
                points = points[:-1]
            points = np.vstack([fh, points, ft])
            self.annotations[self.foreground] = points
        # Update dependant annotations.
        for dep in deps:
            self._recalc_ends(dep)
        # Redraw the annotations.
        if redraw:
            self.redraw_annotations(background = hasdeps)
            self._increment_annotation_change()

    def on_key_press(self, key, shift_down, ctrl_down, meta_down):
        """This method a key is pressed."""
        # If we're ignoring input, just ignore it.
        if self.ignore_input: return

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
