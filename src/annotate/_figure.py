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

    def __init__(self, state):
        """Initialize the figure panel state."""
        # Store the state (from the annotation tool).
        self.state  = state
        self.locked = state.locked # expo

        # Initialize the (shared = canvas & viewer) variables.
        self.target      = None # current target id tuple
        self.active      = None  # current active annotation name
        self.annotations = {} # annotation_name -> (N, 2) coordinates
        self.fixed_heads = {} # annotation_name -> (1, 2) or None
        self.fixed_tails = {} # annotation_name -> (1, 2) or None
        self.editable    = np.array([]) # editable indices
        self.cursor      = None# cursor index into active annotation

        # Canvas-specific variables
        self.image      = None # ipywidgets.Image of the grid
        self.grid       = None # figure_grid layout (list of lists)
        self.grid_shape = None # (rows, cols) tuple
        self.xlim       = None # x-axis figure limits
        self.ylim       = None # y-axis figure limits

        # Cortex viewer-specific variables
        self.faces       = None # (n_faces, 3) array of mesh faces
        self.coordinates = None # (n_vertices, 3) array of mesh coordinates
        self.curvature   = None # (n_vertices, 3) array of curvature colors
       
        # Flatmap mesh for address computation (set by update_cortex).
        # This is the fsaverage flatmap for the current hemisphere, used to 
        # convert 2D flatmap coordinates into barycentric addresses.
        self.flatmap = None

        # Overlay data (set by update_overlay).
        self.overlay = None  # (N_vertices, 3) overlay RGB colors, or None

        # Surface annotations (set by update_surface_annotations).
        # Dict of annotation_name → { "addresses", "coordinates", "point_types" }
        self.surface_annotations = {}

        # Corex viewer style settings 
        self.style = {
            "inflation_percent" : 100,
            "overlay"           : "curvature",
            "overlay_alpha"     : 1.0, 
            "point_size"        : 1.5, 
            "line_width"        : 0.25,
            "line_interp"       : 10,
        }

        # -- Observer callbacks -----------------------------------------------
        # Registered via observe() / observe_message().
        #TODO: need to understand this nonsense
        self._observers         = []  # fn(change_type, **kwargs)
        self._message_observers = []  # fn(message, duration)

        # Register as an observer on the figure state to auto-update when 
        # annotations change. (this is for the cortex-viewer)
        self.figure_state.observe(self._on_figure_state_change)

    # Observer Methods ---------------------------------------------------------

    def observe(self, fn):
        """Register an observer callback for annotation state changes."""
        self._observers.append(fn)

    
    def observe_message(self, fn):
        """Register an observer callback for messages to display on the canvas."""
        self._message_observers.append(fn)

    # Notification Helpers -----------------------------------------------------

    def _notify(self, change_type, **kwargs):
        """Notify all state observers of an annotation change."""
        for fn in self._observers:
            fn(change_type, **kwargs)


    def _notify_message(self, message, duration = None):
        """Notify all message observers of a message."""
        for fn in self._message_observers:
            fn(message, duration)


    # Fixed Point Calculation --------------------------------------------------

    #TODO: dunno if this is canvas only or not yet.
    @staticmethod
    def empty_point_matrix():
        return np.zeros((0, 2), dtype = float)


    def calc_fixed_point(self, annotation, target_annotations, fixed_point):
        """Calculates the fixed head or tail point for the given annotation."""
        if fixed_point not in ("fixed_head", "fixed_tail"):
            raise ValueError(f"Invalid fixed point: {fixed_point}")

        # Get the fixed head or tail attribute for the given annotation.
        fixed_point = getattr(self.annot_cfg[annotation], fixed_point)

        # If there is a fixed head, we need to calculate it using the provided function.
        if fixed_point is not None:
            try:
                fixed_point = fixed_point["calculate"](target_annotations)
                fixed_point = fixed_point.reshape(1, 2)
            except Exception:
                fixed_point = None
        
        # Return the fixed point (None or coordinates of the fixed point).
        return fixed_point


    @staticmethod
    def _init_editable(x = None):
        """Initializes the editable points for the given annotation."""
        if x is None: return np.zeros((0,), dtype = int)
        return np.array([x], dtype = int)
    

    def _calc_editable(self):
        """Calculates the editable points for the active annotation."""
        # Get the points, fixed head, and fixed tail for the active annotation
        points = self.annotations[self.active]
        fixed_head = self.fixed_heads[self.active]
        fixed_tail = self.fixed_tails[self.active]

        # Determine which points are fixed by comparing them to the fixed head and tail.
        fixed_head = np.all(points == fixed_head, axis = 1)
        fixed_tail = np.all(points == fixed_tail, axis = 1)
        fixed_index = np.logical_or(fixed_head, fixed_tail)

        # Return the indices of the editable points (i.e., non-fixed points).
        return np.where(~fixed_index)[0]
    

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
    
    def update_state(self, target_id, annotation, target_annotations):
        """Updates the state to reflect the given target and annotation."""
    
        # If neither the target nor the annotation is changing, we can skip the update.
        if self.target == target_id and self.active == annotation: return

        # Store the previous state.
        prev_target     = self.target
        prev_annotation = self.active

        # Update the target, active annotation, and annotations.
        self.target      = target_id
        self.active      = annotation
        self.annotations = target_annotations

        # Update the image data, grid shape, and figure limits from the state.
        image_data, grid_shape, meta_data = self.state.grid(
            self.target, self.active)
        self.image = ipw.Image(value = image_data, format = "png")
        self.grid       = self.annot_cfg.figure_grid[self.active]
        self.grid_shape = grid_shape
        self.xlim = meta_data["xlim"]
        self.ylim = meta_data["ylim"]

        # If the target is changing, we need to reset the fixed heads and tails, 
        # since they are target specific. Recalculating everything.
        if self.fixed_heads == {} or self.fixed_tails == {} or \
            prev_target != self.target: 
            self.fixed_heads = {}
            self.fixed_tails = {}
            recalc_fixed     = list(self.annotations.keys())
        # If the annotation is changing, we need to recalculate the fixed heads
        # tails for dependencies of the previous annotation.
        else:
            prev_deps = self.annot_cfg.fixed_dependencies.get(prev_annotation, [])
            recalc_fixed = { self.active, *prev_deps}
            
        # Recalculate the fixed head and tails of the given fixed annotations.
        for annotation in recalc_fixed:
            self.fixed_heads[annotation] = self.calc_fixed_point(
                annotation, self.annotations, "fixed_head")
            self.fixed_tails[annotation] = self.calc_fixed_point(
                annotation, self.annotations, "fixed_tail")
        
        # Get the points and annotation type for the active annotation.
        points = self.annotations[self.active]
        atype  = self.annot_cfg.types[self.active]

        # If there are no points for the current annotation, initialize.
        if points is None or points.shape[0] == 0:
            points = self.empty_point_matrix()

        # Determine the editable points.
        if atype == "point":
            # Points annotations either have no point or exactly one point.
            if points.shape[0] == 0:
                self.editable = self._init_editable()
            else:
                self.editable = self._init_editable(0)
        else: # atype in ( "contour", "boundary" )
            # If points is empty, update the annotations with the fixed points. 
            # Annotations should be saved WITH their fixed heads and tails.
            if points.shape[0] == 0:
                if self.fixed_heads[self.active] is not None:
                    points = np.vstack([self.fixed_heads[self.active], points])
                if self.fixed_tails[self.active] is not None:
                    points = np.vstack([points, self.fixed_tails[self.active]])
                    
                # Update the annotation with the new points, if necessary.
                self.annotations[self.active] = points

            # Calculate the editable points (non-fixed points)
            self.editable = self._calc_editable()
    
        # If there are no editable points, we set the cursor to None.
        # Otherwise, we set the cursor to the last editable point.
        if self.editable.shape[0] == 0:
            self.cursor = None
        else:
            self.cursor = self.editable[-1]


    # Observer Callback --------------------------------------------------------

    # def _on_figure_state_change(self, change_type, **kwargs):
    #     """Respond to FigurePanelState changes.

    #     ``"state"``
    #         Full selection change. Reload cortex geometry if the target 
    #         changed, then recompute all surface annotations.
    #     ``"annotations"``
    #         Annotation points changed. Recompute surface annotations for 
    #         the active annotation (and background if deps changed).
    #     ``"cursor"``
    #         Cursor moved. No recomputation needed for the 3D viewer.
    #     """
    #     if change_type == "state":
    #         # Check if the target changed (requires full cortex reload).
    #         new_target = self.figure_state.target_id
    #         if new_target != self.target_id:
    #             self.update_cortex(new_target)
    #             self.update_overlay()
    #         # Recompute all surface annotations for the new selection.
    #         self.update_surface_annotations()

    #     elif change_type == "annotations":
    #         # Recompute surface annotations (active + potentially deps).
    #         active = self.figure_state.annotation
    #         redraw_bg = kwargs.get("redraw_background", False)
    #         if redraw_bg:
    #             # Dependent annotations changed too — recompute all.
    #             self.update_surface_annotations()
    #         else:
    #             # Only the active annotation changed.
    #             self.update_surface_annotations(active)

    #     # "cursor" changes don't affect the 3D viewer state.

    # Cortex Geometry Methods --------------------------------------------------

    def update_cortex(self, target_id):
        """Load cortex geometry from config and compute blended coordinates."""

        self.target_id = target_id
        self.target    = self.state.targets[target_id]

        # Load mesh geometry from config.
        cortex_fn = self.state.config.cortex_fn
        self.faces = cortex_fn["faces"]
        midgray    = cortex_fn["midgray"]
        inflated   = cortex_fn["inflated"]

        # Compute blended coordinates between midgray and inflated surfaces.
        inflation_proportion = self.style["inflation_percent"] / 100.0
        self.coordinates = ((inflated - midgray) * inflation_proportion) + midgray

        # Store curvature colors (used as the base mesh coloring).
        self.curvature = cortex_fn["curvature"]

        # Store the flatmap mesh for address computation.
        # TODO: The exact access pattern for the flatmap depends on the config
        # structure. This may need to be adjusted based on how config.cortex_fn
        # or config.targets provides the fsaverage flatmap for the hemisphere.
        if "flatmap" in cortex_fn:
            self.flatmap = cortex_fn["flatmap"]


    def update_overlay(self):
        """Update overlay colors based on the current overlay selection."""
        if self.style["overlay"] == "curvature":
            self.overlay = None
        else:
            overlay_name = self.style["overlay"]
            overlay_fn   = self.state.config.cortex_fn[overlay_name]
            self.overlay = overlay_fn(self.target, overlay_name)

    # Surface Annotation Methods -----------------------------------------------

    @staticmethod
    def _flatmap_to_surface(flatmap_address, mesh_coordinates):
        """Convert flatmap annotation coordinates to surface coordinates."""
        bary_faces  = flatmap_address["faces"]       # (3, n_faces)
        bary_coords = flatmap_address["coordinates"] # (2, n_points)
        tx = np.transpose(mesh_coordinates[:, bary_faces], (1, 0, 2)) # (3, 3, n_points)
        return barycentric_to_cartesian(tx, bary_coords) # (3, n_points)


    def _interpolate_coordinates(self, coordinates, point_types):
        """Interpolate coordinates along the annotation path."""
        # Get number of interpolated points
        n = self.style["line_interp"] + 2

        # Intialize ararys to store interpolated coordinates
        x_interp = [];  y_interp = [];  ptype_interp = []
        
        # Initialize point type interpolation filler
        ptype_filler = [self.POINT_INTERP] * self.style["line_interp"]
        
        # Iterate over each segment and interpolate points  
        n_interp = coordinates.shape[0] - 1
        for i in np.arange(n_interp):
            # Extract start and end coordinates and point types for the segment
            xs, xe = coordinates[i, 0], coordinates[i + 1, 0]
            ys, ye = coordinates[i, 1], coordinates[i + 1, 1]
            ps, pe = point_types[i], point_types[i + 1]
            
            # Interpolate x and y coordinates and point types for the segment\
            xn = np.linspace(xs, xe, n)
            yn = np.linspace(ys, ye, n)
            pn = [ps, *ptype_filler, pe]

            if i == 0:
                # for the first segment, include the starting point
                x_interp.append(xn)
                y_interp.append(yn)
                ptype_interp.append(pn)
            else:
                # for subsequent segments, exclude the starting point to avoid duplicates
                x_interp.append(xn[1:])
                y_interp.append(yn[1:])
                ptype_interp.append(pn[1:])

        # Concatenate and prepare interpolated points
        x_interp     = np.concatenate(x_interp)
        y_interp     = np.concatenate(y_interp)
        ptype_interp = np.concatenate(ptype_interp)

        # Return interpolated coordinates (as matrix) and point types (as int)
        interp_coordinates = np.vstack((x_interp, y_interp, ptype_interp)).T
        return interp_coordinates[:, :-1], interp_coordinates[:, -1].astype(int)


    def update_surface_addresses(self, annotations = None):
        """Update cortical surface addresses for each annotation.

        Converts 2D flatmap coordinates from FigurePanelState.annotations 
        into barycentric addresses on the cortical surface mesh.
        """
        # Read flatmap annotations from the shared figure state.
        #TODO:!!!!
        flatmap_annotations = self.figure_state.annotations

        # Get the list of annotations to update
        if annotations is None:
            annotations = list(flatmap_annotations.keys())
        elif isinstance(annotations, str):
            annotations = [annotations]
        else:
            raise ValueError(f"Invalid annotations value: {annotations}")

        # We need a flatmap mesh to compute addresses.
        if self.flatmap is None: return

        # Convert each flatmap annotation to surface coordinates
        for key in annotations: # for each annotation to update
            # Get the current annotaitons flatmap coordinates
            flatmap_coordinates = flatmap_annotations.get(key, None)

            # If no flatmap coordinates, set surface annotation to None
            if flatmap_coordinates is None or flatmap_coordinates.shape[0] == 0:
                self.surface_annotations[key] = {
                    "addresses"   : None,
                    "coordinates" : None,
                    "point_types" : None,
                }
                continue

            # Determine point types (fixed vs user points).
            n_points    = flatmap_coordinates.shape[0]
            point_types = np.full(n_points, self.POINT_USER)
            fixed_head  = bool(self.annot_cfg.fixed_head[key])
            fixed_tail  = bool(self.annot_cfg.fixed_tail[key])
            if fixed_head: point_types[0]  = self.POINT_FIXED
            if fixed_tail: point_types[-1] = self.POINT_FIXED

            # Interpolate coordinate if there are more than one point (to make a 
            # segment) and if the points are NOT all fixed points.
            if n_points > 1 and not np.all(point_types == self.POINT_FIXED):
                flatmap_coordinates, point_types = \
                    self._interpolate_coordinates(flatmap_coordinates, point_types)

            # Convert flatmap coordinates to barycentric addresses.
            flatmap_address = self.flatmap.address(flatmap_coordinates.T)

            # Store surface annotation addresses
            # TODO: gotta make sure that the addressing structure is stored?
            self.surface_annotations[key] = {
                "addresses"   : flatmap_address,
                "coordinates" : None,
                "point_types" : point_types,
            }


    def update_surface_coordinates(self, annotations = None):
        """Update cortical surface coordinates for each annotation."""
        #TODO: !!!!
        flatmap_annotations = self.figure_state.annotations

        # Get the list of annotations to update
        if annotations is None:
            annotations = list(flatmap_annotations.keys())
        elif isinstance(annotations, str):
            annotations = [annotations]
        else:
            raise ValueError(f"Invalid annotations value: {annotations}")
        
        # Update surface coordinates for each annotation
        for key in annotations: # for each annotation to update
            # Get the current annotation's surface addresses
            surface_annotation = self.surface_annotations.get(key, {})
            flatmap_address = surface_annotation.get("addresses", None)

            # If surface addresses, calculate the surface annotation coordinates 
            if flatmap_address is not None:
                # Calculate surface coordinates from flatmap addresses and store in surface annotations
                surface_coordinates = self._flatmap_to_surface(
                    flatmap_address, self.coordinates)

                # Store surface annotation coordinates
                self.surface_annotations[key]["coordinates"] = surface_coordinates


    def update_surface_annotations(self, annotations = None):
        """Update cortical annotations based on current state."""
        # Initialize surface annotations dictionary if not present
        if annotations is None: self.cortex_annotations = {} 

        # Get the list of annotations to update
        self.update_surface_addresses(annotations)
        self.update_surface_coordinates(annotations)


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

    def __init__(self, state, width = 512, height = 512):
        """Initialize the figure panel."""
        # Create the figure panels state (NOT annotation state)
        self.state = FigurePanelState(state)

        # Make the canvas panel.
        self.canvas_panel = CanvasPanel(self.state)

        # Make the cortex viewer panel. 
        self.viewer_panel = CortexViewerPanel(
            self.state, width = width, height = height
        )

        # Create the Box (HBox/VBox) figure area.
        super().__init__(
            children = [ self.canvas_panel, self.viewer_panel ],
            layout   = self._HORIZONTAL_LAYOUT
        )

    # Public Interface for AnnotationTool --------------------------------------
    # These methods provide a clean API so that _core.py does not need to 
    # reach through to canvas_panel or viewer_panel directly.

    def update(self, target_id, annotation, target_annotations):
        """Update the figure state with a new selection.

        Delegates to FigurePanelState.update_state(), which will notify 
        both panels to refresh.
        """
        self.figure_state.update_state(
            target_id, annotation, target_annotations)


    def write_message(self, message, **kwargs):
        """Display a message on the canvas panel."""
        self.canvas_panel.write_message(message, **kwargs)


    def clear_message(self):
        """Clear any message currently displayed on the canvas panel."""
        self.canvas_panel.clear_message()


    def redraw_annotations(self):
        """Redraw annotation layers on the canvas (e.g., after a style change)."""
        self.canvas_panel.redraw_canvas(redraw_image = False)


    def resize(self, new_figure_size):
        """Resize the canvas panel to a new figure size."""
        self.canvas_panel.resize_canvas(new_figure_size)


    @property
    def loading_context(self):
        """Expose the canvas loading context for AnnotationTool."""
        return self.canvas_panel.loading_context