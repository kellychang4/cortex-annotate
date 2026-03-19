# -*- coding: utf-8 -*-
################################################################################
# annotate/figure/_editor.py

"""
Implementation code for the Figure Panel.
"""

# Imports ----------------------------------------------------------------------

import threading
import numpy as np
import ipywidgets as ipw
from functools import partial

# The Figure Panel State ------------------------------------------------------

class AnnotationEditor:
    """Figure panel state for the cortex annotation tool."""

    # Point type constants (used for annotation rendering)
    POINT_FIXED  = 2  # fixed head/tail point
    POINT_USER   = 1  # user-placed point
    POINT_INTERP = 0  # interpolated point (between user/fixed points)


    class CanvasState:
        def __init__(self):
            """Initialize the canvas state."""
            self.image      = None # ipw.Image (background image)
            self.grid       = None # figure_grid layout
            self.grid_shape = None # (rows, cols) tuple
            self.xlim       = None # x-axis figure limits
            self.ylim       = None # y-axis figure limits


    class ViewerState:
        def __init__(self):
            """Initialize the viewer state."""
            self.faces        = None # (n_faces, 3) array of mesh faces
            self._coordinates = None # list of (n_vertices, 3), 1 = no interp, 2 = interp
            self.overlays     = {}   # (n_vertices, 3) array of overlay RGB colors

            # Viewer annotations (surface!!!)
            self.canvas_to_viewer = None # function to convert canvas coordinates to viewer coordinates
            self.annotations = {} # Dict of annotation_name → { "coordinates", "point_types" }

            # Style settings for the viewer.
            self.style = {
                "morph_percent" : 0,
                "overlay"       : "curvature",
                "overlay_alpha" : 1.0, 
                "point_size"    : 1.5, 
                "line_width"    : 0.25,
                "line_interp"   : 10,
            }


        @property
        def coordinates(self):
            """Compute blended coordinates between internal coordinate values."""
            # If only one set of coordinates, return without interpolation
            if len(self._coordinates) == 1: return self._coordinates[0]

            # Else: there two set of coordinates, return blended coordinates
            start_coords, end_coords = self._coordinates
            morph_proportion = self.style["morph_percent"] / 100.0
            return ((end_coords - start_coords) * morph_proportion) + start_coords


    def __init__(self, annot_state):
        """Initialize the figure panel state."""
        # Store the state (from the annotation tool).
        self.annot_state = annot_state
        self.annot_cfg   = annot_state.config.annotations
        self.style       = annot_state.style # get/set style method

        # Initialize the (shared = canvas & viewer) variables.
        self.target      = None # current target id tuple
        self.active      = None # current active annotation name
        self.annotations = {} # annotation_name -> (N, 2) coordinates
        self.fixed_heads = {} # annotation_name -> (1, 2) coordinates or None
        self.fixed_tails = {} # annotation_name -> (1, 2) coordinates or None
        self.editable    = np.array([]) # editable indices
        self.cursor      = None # cursor index into active annotation

        # Canvas and Viewer specific variables
        self.canvas = self.CanvasState()
        self.viewer = self.ViewerState()
       
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


    # Cortex Viewer Annotation Methods -----------------------------------------
    
    def _interpolate_coordinates(self, coordinates, point_types):
        """Interpolate coordinates along the annotation path."""
        # Get number of interpolated points
        n = self.viewer.style["line_interp"] + 2

        # Intialize ararys to store interpolated coordinates
        x_interp = []; y_interp = []; ptype_interp = []
        
        # Initialize point type interpolation filler
        ptype_filler = [self.POINT_INTERP] * self.viewer.style["line_interp"]
        
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


    def update_viewer_annotations(self, annotations = None):
        """Update cortical viewer annotations for each annotation."""
        # Determine the annotations to update. If None, update all annotations.
        if annotations is None: annotations = self.annot_cfg.names.copy()

        # Convert each canvas (2d) annotation to viewer (3d) coordinates
        for key in annotations: # for each annotation to update
            # Get the current canvas points
            canvas_points = self.annotations.get(key, None)
            print(f"working on: {key}")
            print(f"canvas_points: {canvas_points}")

            # If no points, set viewer annotation to None
            if canvas_points is None or canvas_points.shape[0] == 0:
                self.viewer.annotations[key] = {
                    "coordinates" : None,
                    "point_types" : None,
                }
                continue

            # Determine point types (fixed vs user points).
            n_points    = canvas_points.shape[0]
            point_types = np.full(n_points, self.POINT_USER)
            fixed_head  = bool(self.annot_cfg.fixed_heads[key])
            fixed_tail  = bool(self.annot_cfg.fixed_tails[key])
            if fixed_head: point_types[0]  = self.POINT_FIXED
            if fixed_tail: point_types[-1] = self.POINT_FIXED

            print(f"n_points: {n_points}")
            print(f"before interp: {canvas_points.shape}")

            # Interpolate coordinate if there are more than one point (to make a 
            # segment) and if the points are NOT all fixed points.
            if n_points > 1 and not np.all(point_types == self.POINT_FIXED):
                canvas_points, point_types = self._interpolate_coordinates(
                    canvas_points, point_types)
            
            print(f"after interp: {canvas_points.shape}")
            print(f"point_types: {point_types}")

            # Calculate viewer coordinates from canvas (interpolated) coordinates
            viewer_coordinates = self.viewer.canvas_to_viewer(
                canvas_points, self.viewer.coordinates)

            # Store viewer annotations coordinates and point types
            self.viewer.annotations[key] = {
                "coordinates": viewer_coordinates,
                "point_types": point_types
            }


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
            self.viewer.faces = cortex_dict["faces"](target)

            # Prepare the internal coordinates, depends on morph_between
            morph_between = cortex_dict.get("morph_between", None)
            print(f"morph_between: {morph_between}")
            if morph_between is None: morph_between = [ "_default" ]
            print(f"morph_between: {morph_between}")
            self.viewer._coordinates = [
                cortex_dict["coordinates"][coordinate_name](target)
                for coordinate_name in morph_between
            ]

            # Prepare the canvas to viewer function
            self.viewer.canvas_to_viewer = partial(
                cortex_dict["canvas_to_viewer"], target)
            print(f"self.viewer.canvas_to_viewer: {self.viewer.canvas_to_viewer}")

            # Update the viewer annotatations (all annotations)
            self.update_viewer_annotations()
            
            # Prepare the overlay values 
            self.viewer.overlays = {
                key: overlay_fn(target, key)
                for key, overlay_fn in cortex_dict["overlays"].items()
            } 
        

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
            if fixed_head is not None: points[0,:] = fixed_head

            # Recalculate and update the fixed tail for the dependent annotation.        
            fixed_tail = self.calc_fixed_point(fd, self.annotations, "fixed_tail")
            if fixed_tail is not None: points[-1,:] = fixed_tail

            # Update the annotation with the new points.
            self.annotations[fd] = points

    # Push Point Method --------------------------------------------------------

    def push_point(self, new_point):
        """Pushes a new point to the active annotation dependent on cursor position."""
        # We can only push points if there is an active annotation.
        if self.active is None: return None

        # Get the current points for this annotation. If None, initialize empty.
        points = self.annotations[self.active]
        if points is None: points = self.empty_point_matrix()

        # Get the annotation type for this annotation.
        atype = self.annot_cfg.type[self.active]
        
        # Depending on the annotation type, we add the newest point to the
        # annotation in different ways.
        if atype == "point":
            # For a point annotation, replace the current point with the new point.
            points        = new_point
            self.editable = self._init_editable(0)
            self.cursor   = 0

        else: # atype in ( "contour", "boundary" )
            # If there are no points, we just add the new point.
            if points.shape[0] == 0:
                self.editable = self._init_editable(0)
                self.cursor   = 0

            # If there are no editable points, we add the new point to the head
            # or tail depending on which one is fixed.                
            elif self.editable.shape[0] == 0:
                if self.fixed_heads[self.active] is not None:
                    self.editable = self._init_editable(1)
                elif self.fixed_tails[self.active] is not None:
                    self.editable = self._init_editable(0)
                self.cursor = self.editable[0]   

            # If there are editable points, we add the new point after the 
            # current cursor position and move the cursor to the new point.
            else: 
                # Because we are inserting a point, all the editable points 
                # after the cursor need to be shifted by one index.
                self.editable[self.editable > self.cursor] += 1

                # We add the new cursor position to the editable points.
                self.editable = np.sort(np.append(self.editable, self.cursor + 1))
                
                # Finally, we increment the cursor to move it to the next position.
                self.cursor += 1

            # Insert the new point at the cursor position.
            points = np.insert(points, self.cursor, new_point, axis = 0)
 
        # Update the annotation with the new points.
        self.annotations[self.active] = points

        # Update dependent annotations, if this active annotation has them.
        fixed_deps = self.annot_cfg.fixed_dependencies[self.active]
        if len(fixed_deps) > 0: self._recalculate_deps(self.active)

        # Return fixed dependencies
        return fixed_deps


    # Toggle Cursor Method -----------------------------------------------------
    
    def toggle_cursor(self):
        """Toggles the cursor position of the active annotation."""
        # Extract current annotation type.
        atype = self.annot_cfg.type[self.active]

        # For a point annotation, there is only one point. Toggling the 
        # cursor position does not do anything, so we can skip it.
        if atype == "point": return

        # If there are less than two editable points, we cannot toggle the cursor.
        if self.editable.shape[0] < 2: return

        # For contour or boundary annotations, we toggle the cursor position by 
        # moving it to the next editable point in the annotation.
        if atype in ( "contour", "boundary" ):
            # Get the index of the current cursor position in the editable points.
            current_index = np.where(self.editable == self.cursor)[0][0]

            # Calculate the index of the next editable point with wraparound.
            next_index = np.mod(current_index + 1, self.editable.shape[0])

            # Update the cursor to the next editable point.
            self.cursor = self.editable[next_index]

    
    # Pop Point Method ---------------------------------------------------------
    
    def pop_point(self):
        """Removes the point at the current cursor position of the active annotation."""
        # We can only push points if there is an active annotation.
        if self.active is None: return None

        # Get the current annotation and annotation type.
        points = self.annotations[self.active]
        atype  = self.annot_cfg.type[self.active]

        # If there are no points, we cannot delete anything. Skip.
        if points is None or points.shape[0] == 0 or \
            self.editable.shape[0] == 0: return
        
        # Check if there are any LIVE dependencies on this annotation. If so, 
        # we cannot delete the last point of this annotation because the 
        # dependent annotations rely on it. 
        fixed_deps = self.annot_cfg.fixed_dependencies[self.active]
        if len(fixed_deps) > 0 and self.editable.shape[0] == 1:
            # Determine the number of fixed points for each dependent 
            # annotation. This number is the minimum number of points that the 
            # annotation must have be considered LIVE.
            n_fixed = [ len(self.annot_cfg.fixed_points[fd]) for fd in fixed_deps ]

            live_deps = [
                fd for fd, n in zip(fixed_deps, n_fixed) 
                if self.annotations[fd] is not None
                and self.annotations[fd].shape[0] > n
            ]
        
            if live_deps:
                # Write a warning message to the user about live dependencies. 
                self.write_message(
                    f"Cannot delete: '{self.active}'. It is required by "
                    f"'{', '.join(live_deps)}'. Clear those annotations first."
                )
                # Clear the message after 3 seconds. 
                threading.Timer(3.0, self.clear_message).start()
                return
        
        # If there are points, we delete based on annotation type.
        if atype == "point":
            # For a point annotation, we delete the single point.
            points        = self.empty_point_matrix()
            self.editable = self._init_editable()
            self.cursor   = None
        else: # atype in ( "contour", "boundary" )
            # If there are points to delete, delete at current position.
            points = np.delete(points, self.cursor, axis = 0)

            # Remove the current cursor from the editable points.
            self.editable = self.editable[self.editable != self.cursor]
            if self.editable.shape[0] == 0:
                self.cursor = None
            else:
                # Removing an index causes all the indices larger than the 
                # current position to shift down by one, so we need to decrement
                # the editable points.
                self.editable[self.editable > self.cursor] -= 1

                # When the cursor is at the head of the editable points, we do
                # not need to decrement the cursor because it will just move 
                # down with the shift of the points. However, if the cursor is
                # anywhere else, we need to decrement the cursor.
                if self.cursor != self.editable[0]: 
                    self.cursor -= 1

        # Update the annotation with the new points.
        self.annotations[self.active] = points

        # Update dependent annotations, if this active annotation has them.
        if len(fixed_deps) > 0: self._recalculate_deps(self.active)

        # Return fixed dependencies
        return fixed_deps

    
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
