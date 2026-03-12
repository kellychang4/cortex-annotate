# -*- coding: utf-8 -*-
################################################################################
# annotate/_figure.py

"""
Implementation code for the Figure Panel.
"""

# Imports ----------------------------------------------------------------------

import numpy as np
import ipywidgets as ipw
from neuropythy.geometry.util import barycentric_to_cartesian

from ._canvas import CanvasPanel
from ._viewer import CortexViewerPanel

# The Figure Panel State ------------------------------------------------------

class FigurePanelState:
    """Figure panel state for the cortex annotation tool."""

    # Point type constants (used for annotation rendering)
    POINT_FIXED  = 2  # fixed head/tail point
    POINT_USER   = 1  # user-placed point
    POINT_INTERP = 0  # interpolated point (between user/fixed points)

    class CanvasState:
        def __init__(self, annot_state):
            """Initialize the canvas state."""
            self.image      = None # ipw.Image (background image)
            self.grid       = None # figure_grid layout
            self.grid_shape = None # (rows, cols) tuple
            self.xlim       = None # x-axis figure limits
            self.ylim       = None # y-axis figure limits
            self.style      = annot_state.style # get/set canvas style method


    class ViewerState:
        def __init__(self):
            """Initialize the viewer state."""
            self.faces        = None # (n_faces, 3) array of mesh faces
            self._coordinates = None # list of (n_vertices, 3), 1 = no interp, 2 = interp
            self.overlays     = {}   # (n_vertices, 3) array of overlay RGB colors

            # Surface annotations (set by update_surface_annotations).
            # Dict of annotation_name → { "addresses", "coordinates", "point_types" }
            self.surface_annotations = {}

            self.style = {
                "inflation_percent" : 100,
                "overlay"           : "curvature",
                "overlay_alpha"     : 1.0, 
                "point_size"        : 1.5, 
                "line_width"        : 0.25,
                "line_interp"       : 10,
            }


        @property
        def coordinates(self):
            """Compute blended coordinates between internal coordinate values."""
            # If only one set of coordinates, return without interpolation
            if len(self._coordinates) == 1: return self._coordinates[0]

            # Else two set of coordinates, return blended coordinates
            start_coords, end_coords = self._coordinates
            inflation_proportion = self.style["inflation_percent"] / 100.0
            return ((end_coords - start_coords) * inflation_proportion) + start_coords


    def __init__(self, annot_state):
        """Initialize the figure panel state."""
        # Store the state (from the annotation tool).
        self.annot_state = annot_state
        self.annot_cfg   = annot_state.config.annotations
        self.locked      = annot_state.locked

        # Initialize the (shared = canvas & viewer) variables.
        self.target      = None # current target id tuple
        self.active      = None # current active annotation name
        self.annotations = {} # annotation_name -> (N, 2) coordinates
        self.fixed_heads = {} # annotation_name -> (1, 2) coordinates or None
        self.fixed_tails = {} # annotation_name -> (1, 2) coordinates or None
        self.editable    = np.array([]) # editable indices
        self.cursor      = None # cursor index into active annotation

        # Canvas and Viewer specific variables
        self.canvas = self.CanvasState(annot_state)
        self.viewer = self.ViewerState()
       
        # Flatmap mesh for address computation (set by update_cortex).
        # This is the fsaverage flatmap for the current hemisphere, used to 
        # convert 2D flatmap coordinates into barycentric addresses.
        self.flatmap = None # TODO
    
    # Fixed Point Methods ------------------------------------------------------

    @staticmethod
    def empty_point_matrix():
        """Returns an empty point matrix with shape (0, 2) and dtype float."""
        return np.zeros((0, 2), dtype = float)


    def calc_fixed_point(self, annotation, target_annotations, fixed_point):
        """Calculates the fixed head or tail point for the given annotation."""
        # Validate the fixed point type.
        if fixed_point not in ("fixed_head", "fixed_tail"):
            raise ValueError(f"Invalid fixed point: {fixed_point}")

        # Get the fixed head or tail attribute for the given annotation.
        fixed_point = getattr(self.annot_cfg, fixed_point)[annotation]

        # If there is a fixed head, we need to calculate it using the compiled function.
        if fixed_point is not None:
            try:
                fixed_point = fixed_point["calculate"](target_annotations)
                fixed_point = fixed_point.reshape(1, 2)
            except Exception:
                fixed_point = None
        
        # Return the fixed point (coordinates of the fixed point or None).
        return fixed_point

    # Editable Methods ---------------------------------------------------------

    @staticmethod
    def _init_editable(x = None):
        """Initializes the editable points for the given annotation."""
        if x is None: return np.zeros((0,), dtype = int)
        return np.array([x], dtype = int)
    

    def _calc_editable(self, annotation):
        """Calculates the editable points for the active annotation."""
        # Get the points, fixed head, and fixed tail for the given annotation
        points     = self.annotations[annotation]
        fixed_head = self.fixed_heads[annotation]
        fixed_tail = self.fixed_tails[annotation]

        # Determine which points are fixed by comparing them to the fixed head and tail.
        fixed_head = np.all(points == fixed_head, axis = 1)
        fixed_tail = np.all(points == fixed_tail, axis = 1)
        fixed_index = np.logical_or(fixed_head, fixed_tail)

        # Return the indices of the editable points (i.e., non-fixed points).
        return np.where(~fixed_index)[0] 

    # Update Method ------------------------------------------------------------

    def update(self, target_id, annotation, target_annotations):
        """Updates the state to reflect the given target and annotation."""
    
        # If neither the target nor the annotation is changing, we can skip the update.
        if self.target == target_id and self.active == annotation: return
        print("Inside Update....")
        # Store the previous state.
        prev_target     = self.target
        prev_annotation = self.active

        # Update the target, active annotation, and annotations.
        self.target      = target_id
        self.active      = annotation
        self.annotations = target_annotations
        
        print(f"self.target: {self.target}")
        print(f"self.active: {self.active}")
        # print(f"self.annotations: {self.annotations}")
        
        # If the target is changing, we need to reset the fixed heads and tails, 
        # since they are target specific. Recalculating all annotations.
        if self.fixed_heads == {} or self.fixed_tails == {} or \
            prev_target != self.target:
            self.fixed_heads = {}
            self.fixed_tails = {}
            recalc_fixed     = list(self.annot_cfg.names)
        # If the annotation is changing, we need to recalculate the fixed heads
        # tails for dependencies of the previous annotation.
        else:
            prev_deps    = self.annot_cfg.fixed_dependencies[prev_annotation]
            recalc_fixed = { self.active, *prev_deps }
            
        # Recalculate the fixed head and tails of the given annotations.
        for annotation in recalc_fixed:
            self.fixed_heads[annotation] = self.calc_fixed_point(
                annotation, self.annotations, "fixed_head")
            self.fixed_tails[annotation] = self.calc_fixed_point(
                annotation, self.annotations, "fixed_tail")
            
        # Get the points and annotation type for the active annotation.
        points = self.annotations[self.active]
        atype  = self.annot_cfg.type[self.active]

        # If there are no points for the current annotation, initialize.
        if points is None or points.shape[0] == 0:
            points = self.empty_point_matrix()

        # Determine the editable points.
        if atype == "point":
            # Points annotations either have no point or exactly one point.
            if points.shape[0] == 0: self.editable = self._init_editable()
            else: self.editable = self._init_editable(0) # one point

        else: # atype in ( "contour", "boundary" )
            # If points is empty, update the annotations with the fixed points. 
            # Annotations should be saved WITH their fixed heads and tails.
            if points.shape[0] == 0:
                if self.fixed_heads[self.active] is not None:
                    points = np.vstack([self.fixed_heads[self.active], points])
                if self.fixed_tails[self.active] is not None:
                    points = np.vstack([points, self.fixed_tails[self.active]])
                    
                # Update the annotation with the fixed points.
                self.annotations[self.active] = points

            # Calculate the editable points (non-fixed points)
            self.editable = self._calc_editable(self.active)
    
        # If there are no editable points, we set the cursor to None.
        # Otherwise, we set the cursor to the last editable point.
        if self.editable.shape[0] == 0: self.cursor = None
        else: self.cursor = self.editable[-1]

        # Canvas-specific updates
        ## Update the image data, grid shape, and figure limits from the state.
        image_data, meta_data  = self.annot_state.grid(self.target, self.active)
        self.canvas.image      = ipw.Image(value = image_data, format = "png")
        self.canvas.grid       = self.annot_cfg.figure_grid[self.active]
        self.canvas.grid_shape = self.annot_cfg.grid_shape[self.active]
        self.canvas.xlim       = meta_data["xlim"]
        self.canvas.ylim       = meta_data["ylim"]

        # Cortex viewer-specific updates
        ## If the target changed, we need to update the cortex variables.
        if prev_target != self.target: # target change
            # Extract the configuration cortex and target dictionary
            cortex_dict = self.annot_state.config.cortex
            target      = self.annot_state.config.targets[self.target]

            # Prepare the viewer faces values
            self.viewer.faces = cortex_dict["faces"](target, None)

            # Prepare the internal coordinates, depends on inflate_between
            inflate_between = cortex_dict.get("inflate_between", None)
            if inflate_between is None: inflate_between = [ "_default" ]
            self.viewer._coordinates = [
                cortex_dict["coordinates"][coordinate_name](target, None)
                for coordinate_name in inflate_between
            ]
            
            # Prepare the overlay values 
            self.viewer.overlays = {
                key: overlay_fn(target, key)
                for key, overlay_fn in cortex_dict["overlays"].items()
            } 
        
        # # Store the flatmap mesh for address computation.
        # # TODO: The exact access pattern for the flatmap depends on the config
        # # structure. This may need to be adjusted based on how config.cortex_fn
        # # or config.targets provides the fsaverage flatmap for the hemisphere.
        # if "flatmap" in cortex_fn:
        #     self.flatmap = cortex_fn["flatmap"]

    # Recalculate Dependencies -------------------------------------------------

    def _recalculate_deps(self, annotation):
        """Recalculates the dependent annotations for the given annotation."""
        # Get the dependent annotations for the given annotation.
        fixed_deps = self.annot_cfg.fixed_dependencies[annotation]

        # If there are no dependencies, we can skip.
        if len(fixed_deps) == 0: return 

        # We need to recalculate each of the dependent annotations using their
        # provided functions and update them in the state.
        for fd in fixed_deps: 
            # Get the current points for the dependent annotation.
            points = self.annotations[fd]

            # If there are no points, we can skip the recalculation.
            if points is None or points.shape[0] == 0: continue

            # Recalculate and update the fixed head for the dependent annotation.        
            fixed_head = self.calc_fixed_point(fd, self.annotations, "fixed_head")
            if fixed_head is not None:
                points[0,:] = fixed_head

            # Recalculate and update the fixed tail for the dependent annotation.        
            fixed_tail = self.calc_fixed_point(fd, self.annotations, "fixed_tail")
            if fixed_tail is not None:
                points[-1,:] = fixed_tail

            # Update the annotation with the new points.
            self.annotations[fd] = points
    
    # Surface Annotation Methods -----------------------------------------------

    # @staticmethod
    # def _flatmap_to_surface(flatmap_address, mesh_coordinates):
    #     """Convert flatmap annotation coordinates to surface coordinates."""
    #     bary_faces  = flatmap_address["faces"]       # (3, n_faces)
    #     bary_coords = flatmap_address["coordinates"] # (2, n_points)
    #     tx = np.transpose(mesh_coordinates[:, bary_faces], (1, 0, 2)) # (3, 3, n_points)
    #     return barycentric_to_cartesian(tx, bary_coords) # (3, n_points)

    
    # def _interpolate_coordinates(self, coordinates, point_types):
    #     """Interpolate coordinates along the annotation path."""
    #     # Get number of interpolated points
    #     n = self.style["line_interp"] + 2

    #     # Intialize ararys to store interpolated coordinates
    #     x_interp = [];  y_interp = [];  ptype_interp = []
        
    #     # Initialize point type interpolation filler
    #     ptype_filler = [self.POINT_INTERP] * self.style["line_interp"]
        
    #     # Iterate over each segment and interpolate points  
    #     n_interp = coordinates.shape[0] - 1
    #     for i in np.arange(n_interp):
    #         # Extract start and end coordinates and point types for the segment
    #         xs, xe = coordinates[i, 0], coordinates[i + 1, 0]
    #         ys, ye = coordinates[i, 1], coordinates[i + 1, 1]
    #         ps, pe = point_types[i], point_types[i + 1]
            
    #         # Interpolate x and y coordinates and point types for the segment\
    #         xn = np.linspace(xs, xe, n)
    #         yn = np.linspace(ys, ye, n)
    #         pn = [ps, *ptype_filler, pe]

    #         if i == 0:
    #             # for the first segment, include the starting point
    #             x_interp.append(xn)
    #             y_interp.append(yn)
    #             ptype_interp.append(pn)
    #         else:
    #             # for subsequent segments, exclude the starting point to avoid duplicates
    #             x_interp.append(xn[1:])
    #             y_interp.append(yn[1:])
    #             ptype_interp.append(pn[1:])

    #     # Concatenate and prepare interpolated points
    #     x_interp     = np.concatenate(x_interp)
    #     y_interp     = np.concatenate(y_interp)
    #     ptype_interp = np.concatenate(ptype_interp)

    #     # Return interpolated coordinates (as matrix) and point types (as int)
    #     interp_coordinates = np.vstack((x_interp, y_interp, ptype_interp)).T
    #     return interp_coordinates[:, :-1], interp_coordinates[:, -1].astype(int)


    # def update_surface_addresses(self, annotations = None):
    #     """Update cortical surface addresses for each annotation.

    #     Converts 2D flatmap coordinates from FigurePanelState.annotations 
    #     into barycentric addresses on the cortical surface mesh.
    #     """
    #     # Read flatmap annotations from the shared figure state.
    #     #TODO:!!!!
    #     flatmap_annotations = self.figure_state.annotations

    #     # Get the list of annotations to update
    #     if annotations is None:
    #         annotations = list(flatmap_annotations.keys())
    #     elif isinstance(annotations, str):
    #         annotations = [annotations]
    #     else:
    #         raise ValueError(f"Invalid annotations value: {annotations}")

    #     # We need a flatmap mesh to compute addresses.
    #     if self.flatmap is None: return

    #     # Convert each flatmap annotation to surface coordinates
    #     for key in annotations: # for each annotation to update
    #         # Get the current annotaitons flatmap coordinates
    #         flatmap_coordinates = flatmap_annotations.get(key, None)

    #         # If no flatmap coordinates, set surface annotation to None
    #         if flatmap_coordinates is None or flatmap_coordinates.shape[0] == 0:
    #             self.surface_annotations[key] = {
    #                 "addresses"   : None,
    #                 "coordinates" : None,
    #                 "point_types" : None,
    #             }
    #             continue

    #         # Determine point types (fixed vs user points).
    #         n_points    = flatmap_coordinates.shape[0]
    #         point_types = np.full(n_points, self.POINT_USER)
    #         fixed_head  = bool(self.annot_cfg.fixed_head[key])
    #         fixed_tail  = bool(self.annot_cfg.fixed_tail[key])
    #         if fixed_head: point_types[0]  = self.POINT_FIXED
    #         if fixed_tail: point_types[-1] = self.POINT_FIXED

    #         # Interpolate coordinate if there are more than one point (to make a 
    #         # segment) and if the points are NOT all fixed points.
    #         if n_points > 1 and not np.all(point_types == self.POINT_FIXED):
    #             flatmap_coordinates, point_types = \
    #                 self._interpolate_coordinates(flatmap_coordinates, point_types)

    #         # Convert flatmap coordinates to barycentric addresses.
    #         flatmap_address = self.flatmap.address(flatmap_coordinates.T)

    #         # Store surface annotation addresses
    #         # TODO: gotta make sure that the addressing structure is stored?
    #         self.surface_annotations[key] = {
    #             "addresses"   : flatmap_address,
    #             "coordinates" : None,
    #             "point_types" : point_types,
    #         }


    # def update_surface_coordinates(self, annotations = None):
    #     """Update cortical surface coordinates for each annotation."""
    #     #TODO: !!!!
    #     flatmap_annotations = self.figure_state.annotations

    #     # Get the list of annotations to update
    #     if annotations is None:
    #         annotations = list(flatmap_annotations.keys())
    #     elif isinstance(annotations, str):
    #         annotations = [annotations]
    #     else:
    #         raise ValueError(f"Invalid annotations value: {annotations}")
        
    #     # Update surface coordinates for each annotation
    #     for key in annotations: # for each annotation to update
    #         # Get the current annotation's surface addresses
    #         surface_annotation = self.surface_annotations.get(key, {})
    #         flatmap_address = surface_annotation.get("addresses", None)

    #         # If surface addresses, calculate the surface annotation coordinates 
    #         if flatmap_address is not None:
    #             # Calculate surface coordinates from flatmap addresses and store in surface annotations
    #             surface_coordinates = self._flatmap_to_surface(
    #                 flatmap_address, self.coordinates)

    #             # Store surface annotation coordinates
    #             self.surface_annotations[key]["coordinates"] = surface_coordinates


    # def update_surface_annotations(self, annotations = None):
    #     """Update cortical annotations based on current state."""
    #     # Initialize surface annotations dictionary if not present
    #     if annotations is None: self.cortex_annotations = {} 

    #     # Get the list of annotations to update
    #     self.update_surface_addresses(annotations)
    #     self.update_surface_coordinates(annotations)

    # Figure Size Methods ------------------------------------------------------

    # def figure_size(self, new_figure_size = None):
    #     """Returns the figure size from the user's preferences.

    #     `state.figure_size()` returns the current figure size.

    #     `state.figure_size(new_figure_size)` updates the current figure size.
    #     """
    #     if new_figure_size is None:
    #         # Just return the current figure size, or the default if it is not set.
    #         return self.preferences.get("figure_size", 256)
    #     else:
    #         # Update the figure size in the preferences, and return the new value.
    #         self.preferences["figure_size"] = new_figure_size
    #         return new_figure_size


# Figure Panel Class -----------------------------------------------------------

class FigurePanel(ipw.Box):
    """Container widget holding the 2D canvas and 3D cortex viewer."""

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

    def __init__(self, annot_state, width = 512, height = 512):
        """Initialize the figure panel."""
        # Store the annotation state.
        self.annot_state = annot_state

        # Create the figure panels state (NOT annotation state)
        self.figure_state = FigurePanelState(
            annot_state = self.annot_state
        )
        print("Inside FigurePanel...")

        # Make the canvas panel.
        self.canvas_panel = CanvasPanel(
            figure_state = self.figure_state,
            figure_size  = annot_state.preferences["figure_size"]
        )

        # Make the cortex viewer panel. 
        self.viewer_panel = CortexViewerPanel(
            figure_state = self.figure_state,
            width = width, 
            height = height
        )

        # Create the Box (HBox/VBox) figure area.
        super().__init__(
            children = [ self.canvas_panel, self.viewer_panel ],
            layout   = self._HORIZONTAL_LAYOUT
        )


        # Set up event observers for mouse clicks and key presses.
        # self.multicanvas.on_mouse_down(self.on_mouse_click)
        # self.multicanvas.on_key_down(self.on_key_press)


    # Mouse Event Handler Methods ----------------------------------------------

    def on_mouse_click(self, x, y):
        """Handle a mouse click on the canvas."""
        # If the figure is locked, we do not allow events.
        if self.annot_state.locked: return

        # Convert canvas pixel coordinates to figure coordinates.
        point = np.array([[x, y]]) # must be (N, 2) matrix
        figure_point = self.canvas_panel.canvas_to_figure(point)

        # Push annotation to the state. 
        self.figure_state.push_point(figure_point)

    # Key Press Event Handler Methods ------------------------------------------

    def on_key_press(self, key, shift_down, ctrl_down, meta_down):
        """Handle a key press on the canvas."""
        # If the figure is locked, we do not allow events.
        if self.annot_state.locked: return

        # Handle the key press.
        key = key.lower()
        if key == "tab":
            # Toggle the cursor (active) position.
            self.state.toggle_cursor()
        elif key == "backspace":
            # Delete current cursor (active) point.
            self.state.pop_point()
        else: 
            pass

    @property
    def loading_context(self):
        """Expose the canvas loading context for AnnotationTool."""
        return self.canvas_panel.loading_context
    



    # Canvas Resizing Method ---------------------------------------------------

    def resize_canvas(self, new_figure_size = None):
        """Resize the canvas so that each grid cell has the given pixel size.

        Triggers a full redraw because resizing clears the canvas.
        """
        # If there is no new_figure_size give, we just use the current figure size.
        if new_figure_size is None:
            new_figure_size = self.figure_size

        # Update the figure size (pixels per grid cell).
        self.figure_size = np.array([new_figure_size, new_figure_size])

        # The canvas size is a product of the figure size and the grid shape.
        self.canvas_size = self.figure_size * np.array(self.state.grid_shape)
        canvas_width, canvas_height = self.canvas_size.astype(int)

        # Resize the multicanvas (this clears it).
        self.multicanvas.width         = canvas_width
        self.multicanvas.height        = canvas_height
        self.multicanvas.layout.width  = f"{canvas_width}px"
        self.multicanvas.layout.height = f"{canvas_height}px"

        # Redraw everything.
        self.redraw_canvas()

    # Redraw Multicanvas Method ------------------------------------------------

    def redraw_canvas(self, redraw_image = True, redraw_annotations = True):
        """Redraw the entire canvas."""
        print("Redrawing canvas...")
        print(self.state.canvas)
        # If there is no image to draw, skip
        if self.state.canvas.image is None: return
        

        # Redraw the loading canvas.
        if redraw_image or redraw_annotations:
            self.loading_canvas.restore()

        # Redraw layers.
        with ipc.hold_canvas():
            if redraw_image:
                self.redraw_image()
            if redraw_annotations:
                self.redraw_annotations()
                self._increment_annotation_change()

    # Internal Helpers ---------------------------------------------------------

    def _increment_annotation_change(self):
        """Increments the annotation change traitlet after redraw triggers."""
        self._annotation_change += 1        

