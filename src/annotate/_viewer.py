# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer.py

"""
Implementation code for the Cortex Viewer.
"""

# Imports ----------------------------------------------------------------------

import k3d
import numpy as np
import ipywidgets as ipw
from matplotlib.colors import to_rgb
from neuropythy.geometry.util import barycentric_to_cartesian

# The Cortex Viewer State ------------------------------------------------------

class CortexViewerState:
    """Viewer-specific state for the 3D cortex viewer.
    
    This class manages the data that is specific to the 3D viewer but not 
    relevant to the 2D canvas or the broader annotation tool. It consumes 
    cortex geometry and overlay data from `AnnotationState.cortex_data()` and
    converts flatmap annotations into 3D surface coordinates.
    """

    # Point type constants (used for annotation rendering)
    POINT_FIXED  = 2  # fixed head/tail point
    POINT_USER   = 1  # user-placed point
    POINT_INTERP = 0  # interpolated point (between user/fixed points)


    def __init__(self, state):
        """Initialize the cortex viewer state.
        
        Parameters
        ----------
        annotation_state : AnnotationState
            The shared annotation state that provides cortex data, annotation
            coordinates, annotation config, and style preferences.
        """
        # Store the state.
        self.state = state

        # Cortex viewer-specific (only) display style options.
        self.style = {
            "inflation_percent" : 100,
            "overlay"           : "curvature",
            "overlay_alpha"     : 1.0, 
            "point_size"        : 1.5, 
            "line_width"        : 0.25,
            "line_interp"       : 10,
        }

        # Current viewer data — populated by update methods, read by the panel.
        self.target_id  = None  # current target id tuple
        self.annotation = None  # current active annotation name 

        # Cortex geometry (set by update_cortex)
        self.faces       = None  # (3, N_faces) face indices
        self.coordinates = None  # (3, N_vertices) blended coordinates
        self.curvature   = None  # (N_vertices, 3) curvature RGB colors

        # Overlay data (set by update_overlay)
        self.overlay = None  # (N_vertices, 3) overlay RGB colors, or None

        # Surface annotations (set by update_surface_annotations)
        # Dict of annotation_name -> { "addresses", "coordinates", "point_types" }
        self.surface_annotations = {}


    # Update Methods -----------------------------------------------------------

    def update_cortex(self, target_id):
        """Load cortex geometry from config and compute blended coordinates.
        
        This evaluates the ``config.cortex`` functions (faces, midgray, 
        inflated) via ``state.cortex_data()`` and computes the blended 
        coordinates based on the current inflation percentage.

        Parameters
        ----------
        target_id : tuple
            The target identifier (e.g., (dataset, participant, hemisphere)).
        """
        # Update the target and get the cortex functions.
        self.target_id = target_id
        self.target    = self.state.targets[target_id]

        # Update faces and coordinates for the mesh.
        self.faces = self.state.config.cortex_fn["faces"]
        midgray    = self.state.config.cortex_fn["midgray"]
        inflated   = self.state.config.cortex_fn["inflated"]

        # Compute blended coordinates between midgray and inflated surfaces.
        inflation_proportion = self.style["inflation_percent"] / 100.0
        self.coordinates = ((inflated - midgray) * inflation_proportion) + midgray

        # Store curvature colors (used as the base mesh coloring).
        self.curvature = self.state.config.cortex_fn["curvature"]


    def update_overlay(self):
        """Update overlay colors based on the current overlay selection.

        If the overlay is ``"curvature"``, no separate overlay is needed (the
        curvature colors are used as the base mesh coloring). Otherwise, the
        overlay color array is fetched from ``state.cortex_data()``.
        """
        if self.style["overlay"] == "curvature":
            self.overlay = None
        else:
            overlay_name = self.style["overlay"]
            overlay_fn   = self.state.config.cortex_fn[overlay_name]
            self.overlay = overlay_fn(self.target, overlay_name)

   # Cortex Annotations Methods -----------------------------------------------

    #TODO: this ssection is a complete mess

    @staticmethod
    def _flatmap_to_surface(flatmap_address, mesh_coordinates):
        """Convert flatmap annotation coordinates to surface coordinates."""
        bary_faces  = flatmap_address["faces"]       # (3, n_faces)
        bary_coords = flatmap_address["coordinates"] # (2, n_points)
        tx = np.transpose(mesh_coordinates[:, bary_faces], (1, 0, 2)) # (3, 3, n_points)
        return barycentric_to_cartesian(tx, bary_coords) # (3, n_points)

    
    def _interpolate_coordinates(self, coordinates, point_types):
        """Interpolate coordinates along the path."""
        # Get number of interpolated points
        n = self.style["line_interp"] + 2

        # Intialize ararys to store interpolated coordinates
        x_interp = []; y_interp = []; ptype_interp = []

        # Initialize point type interpolation filler
        ptype_filler = [self.POINT_INTERP] * self.style["line_interp"]

        # Iterate over each segment and interpolate points  
        n_interp = coordinates.shape[0] - 1 
        for i in np.arange(n_interp): # for each pair of coordinates
            # Extract start and end coordinates and point types for the segment
            xs, xe = coordinates[i, 0], coordinates[i+1, 0]
            ys, ye = coordinates[i, 1], coordinates[i+1, 1]
            ps, pe = point_types[i], point_types[i+1]

            # Interpolate x and y coordinates and point types for the segment
            xn = np.linspace(xs, xe, n)
            yn = np.linspace(ys, ye, n)
            pn = [ps, *ptype_filler, pe]

            if i == 0: # for the first segment, include the starting point
                x_interp.append(xn)
                y_interp.append(yn)
                ptype_interp.append(pn)
            else: # for subsequent segments, exclude the starting point to avoid duplicates
                x_interp.append(xn[1:])
                y_interp.append(yn[1:])
                ptype_interp.append(pn[1:])

        # Concatenate and prepare interpolated points
        x_interp = np.concatenate(x_interp)
        y_interp = np.concatenate(y_interp)
        ptype_interp = np.concatenate(ptype_interp)

        # Return interpolated coordinates (as matrix) and point types (as int)
        interp_coordinates = np.vstack((x_interp, y_interp, ptype_interp)).T
        return interp_coordinates[:,:-1], interp_coordinates[:,-1].astype(int)


    def update_surface_addresses(self, annotations = None): 
        """Update cortical surface addresses for each annotation."""
        # Get the list of annotations to update
        if annotations is None:
            annotations = list(self.flatmap_annotations.keys())
        elif isinstance(annotations, str):
            annotations = [annotations, ]
        else:
            raise ValueError(f"Invalid annotations value: {annotations}")

        # Get current fsaverage hemisphere flatmap
        fsa_flatmap = self.fsaverage[self.hemisphere]["flatmap"]

        # Convert each flatmap annotation to surface coordinates
        for key in annotations: # for each annotation to update
            # Get the current annotaitons flatmap coordinates
            flatmap_coordinates = self.flatmap_annotations[key]

            # If no flatmap coordinates, set surface annotation to None
            if flatmap_coordinates is None or flatmap_coordinates.shape[0] == 0: 
                self.cortex_annotations[key] = {
                    "addresses"   : None,
                    "coordinates" : None,
                    "point_types" : None,
                }
                continue
            
            # If there are flatmap coordinates, figure out each point type
            n_points    = flatmap_coordinates.shape[0]
            point_types = np.full(n_points, self.POINT_USER)
            fixed_head  = bool(self.annot_cfg.fixed_head[key])
            fixed_tail  = bool(self.annot_cfg.fixed_tail[key])
            if fixed_head: point_types[0]  = self.POINT_FIXED
            if fixed_tail: point_types[-1] = self.POINT_FIXED

            # Interpolate coordinate if there are more than 1 point (to make a 
            # segment) and if the points are NOT all fixed points.
            if n_points > 1 and not np.all(point_types == self.POINT_FIXED):
                flatmap_coordinates, point_types = \
                    self._interpolate_coordinates(flatmap_coordinates, point_types)
            
            # Convert flatmap coordinates to addresses
            flatmap_address = fsa_flatmap.address(flatmap_coordinates.T)
        
            # Store surface annotation addresses
            self.cortex_annotations[key] = {
                "addresses"   : flatmap_address,
                "coordinates" : None,
                "point_types" : point_types,
            }


    def update_surface_coordinates(self, annotations = None):
        """Update cortical surface coordinates for each annotation."""
        # Get the list of annotations to update
        if annotations is None:
            annotations = list(self.flatmap_annotations.keys())
        elif isinstance(annotations, str):
            annotations = [annotations, ]
        else:
            raise ValueError(f"Invalid annotations value: {annotations}")

        # Update surface coordinates for each annotation
        for key in annotations:
            # Get the current annotation's surface addresses
            surface_annotation = self.cortex_annotations.get(key, {})
            flatmap_address = surface_annotation.get("addresses", None)

            # If no surface addresses, set surface annotation coordinates to None
            if flatmap_address is not None:
                # Calculate surface coordinates
                surface_coordinates = (
                    self._flatmap_to_surface(flatmap_address, self.coordinates))
                
                # Store surface annotation coordinates
                self.cortex_annotations[key]["coordinates"] = surface_coordinates


    def update_surface_annotations(self, annotations = None):
        """Update cortical annotations based on current state."""
        # Initialize surface annotations dictionary if not present
        if annotations is None: self.cortex_annotations = {} 

        # Get the list of annotations to update
        self.update_surface_addresses(annotations)
        self.update_surface_coordinates(annotations)


# Cortex Viewer Figure Panel ---------------------------------------------------

class CortexViewerPanel(ipw.VBox):
    """Cortex Figure Panel.

    The panel that contains the 3D cortex plot for the Cortex Viewer tool.
    """
    
    def __init__(self, state, width = 512, height = 512):
        # Store the viewer state
        self.viewer_state = CortexViewerState(state)

        # Create a figure background (k3d plot)
        self.figure = k3d.plot(
            height            = height, 
            grid_visible      = False,
            camera_auto_fit   = False,
            menu_visibility   = False,
            camera_fov        = 60,
            axes_helper       = 0, # remove axes direction helper
            camera_zoom_speed = 1.5,
        )

        # Initialize all k3d layers (start empty/invisible).
        self.k3dmesh_cortex       = self._init_mesh()
        self.k3dmesh_overlay      = self._init_mesh()
        self.k3dline_active       = self._init_line()
        self.k3dpoints_active     = self._init_points()
        self.k3dline_background   = self._init_line()
        self.k3dpoints_background = self._init_points()

        # Add all layers to the figure.
        self.figure += self.k3dmesh_cortex
        self.figure += self.k3dmesh_overlay 
        self.figure += self.k3dline_active
        self.figure += self.k3dpoints_active 
        self.figure += self.k3dline_background
        self.figure += self.k3dpoints_background 

        # Set initial camera values
        self.figure.camera = [-160, -10, -6, 15, -30, 0, 0, 0, 1]

        # Initialize the cortex variables
        self.target = None
        self.active = None
        self.annotations = {} 
        self.cortex_annotations = {}
        self.fixed_head = {} 
        self.fixed_tail = {} 
        self.editable   = {} 
        self.cursor     = None

        # Initialize the VBox with the figure as the child 
        super().__init__(
            children = [ self.figure ], 
            layout = {
                "width"   : f"{width}px", 
                "height"  : f"{height}px", 
                "border"  : "1px solid magenta",
                "overflow": "hidden"
            }
        )

    # Update State Method ------------------------------------------------------

    def update_state(self, target_id, annotation, flatmap_annotations):
        """Updates the state to reflect the given target and annotation."""

        # If neither the target nor the annotation is changing, we can skip the update.
        if self.target == target_id and self.active == annotation: return

        # Store the previous state.
        prev_target     = self.target
        prev_annotation = self.active

        # Update the target, active annotation, and annotations.
        self.target      = target_id
        self.active      = annotation

        # Update the flatmap annotation, fixed heads, and fixed tails
        self.annotations = flatmap_annotations
        self.fixed_head  = self.fixed_head
        self.fixed_tail  = self.fixed_tail
        self.editable    = self.editable
        self.cursor      = self.cursor

        # Update the cortex annotations based on the updated flatmap annotations
        self.cortex_annotations = self.viewer_state.surface_annotations

        # Refresh the full 3D figure.
        self.refresh_figure(clear = True, cortex = True, points = True)
    

    # k3d Color Helper Method --------------------------------------------------

    def _rgb_to_k3dcolor(self, colors):
        """Converts a matplotlib color (RGB) into a hex integer for k3d.

        If the given color is a matrix of RGB triples, then a list of
        integers, one per row, is returned. 
        """
        # Convert to numpy array for easier processing
        colors = np.array(colors)
        
        # Handle string color inputs (e.g. "red", "#ff0000", etc.)
        if np.issubdtype(colors.dtype, np.str_):
            if colors.ndim == 0: colors = colors.reshape(-1,)
            colors = np.array([to_rgb(x) for x in colors], dtype = float)

        # Handle single RGB or RGBA triple input (e.g. [1, 0, 0] or [1, 0, 0, 1])
        if colors.ndim == 1: colors = colors.reshape((1, -1))
            
        # Handle floating point inputs by converting to uint8
        if np.issubdtype(colors.dtype, np.floating):
            if colors.max() > 1.0: # if max is greater than 1, assume 0-255 range
                colors = colors.astype(np.uint8)
            else: # else assume 0-1 range and convert to 0-255
                colors = (colors * 255).astype(np.uint8)

        # Handle integer inputs by checking if they are within uint8 range and converting to uint8
        if np.issubdtype(colors.dtype, np.integer):
            if colors.min() < 0 or colors.max() > 255:
                raise ValueError("Color values must be within uint8 range [0-255].")    
            colors = colors.astype(np.uint8)

        # Check that colors are now uint8 and converted to 2D array
        if not np.issubdtype(colors.dtype, np.uint8):
            raise ValueError("Color values must be convertible from float [0,1] or uint8 [0,255].")
        if colors.ndim != 2:
            raise ValueError("Color input must be scalar, 1D RGB/RGBA, or 2D Nx3/Nx4.")
        
        # Convert RGB/RGBA values to k3d color integers
        colors = colors.astype(np.uint32) # ensure uint32 for bitwise operations
        if colors.shape[1] == 3: # if RGB, convert to k3d color integer
            return np.array(
                [ ((r << 16) | (g << 8) | b) for r, g, b in colors ], 
                dtype = np.uint32
            )
        elif colors.shape[1] == 4: # if RGBA, ignore the alpha channel
            # NOTE: k3d does not support alpha in the color integer, ignor
            return np.array(
                [ ((r << 16) | (g << 8) | b) for r, g, b, _ in colors ], 
                dtype = np.uint32
            )
        else:
            raise ValueError("Color matrices must be RGB (Nx3) or RGBA (Nx4).")

    # Empty Value Methods ------------------------------------------------------

    def _empty_coordinates(self):
        """Helper method to create an empty matrix for initializing empty plots."""
        return np.array([[0, 0, 0]], dtype = np.float32)


    def _empty_indices(self):
        """Helper method to create an empty matrix for initializing empty meshes."""
        return np.array([[0, 0, 0]], dtype = np.uint32)


    def _empty_colors(self):
        """Helper method to create an empty color for initializing empty plots."""
        return np.array([0x000000], dtype = np.uint32)
    
    # Initialize Methods -------------------------------------------------------

    def _init_mesh(self):
        """Initialize an empty and invisible mesh."""
        mesh = k3d.mesh(
            vertices     = self._empty_coordinates(), 
            indices      = self._empty_indices(),
            colors       = self._empty_colors(),
            wireframe    = False,
            flat_shading = False
        )
        mesh.visible = False
        return mesh


    def _init_cortex(self):
        """Initialize the cortex mesh."""
        cortex_kwargs = self._prep_cortex()
        return k3d.mesh(**cortex_kwargs, wireframe = False, flat_shading = False)          
    

    def _init_overlay(self):
        """Initialize the cortex overlay mesh."""
        overlay_kwargs = self._prep_overlay()
        if overlay_kwargs is None:
            return self._init_mesh()
        return k3d.mesh(**overlay_kwargs, wireframe = False, flat_shading = False)


    def _init_points(self):
        """Initialize an empty and invisible points plot."""
        points = k3d.points(
            positions = self._empty_coordinates(),
            colors    = self._empty_colors(), 
            shader    = "3d"
        )
        points.visible = False
        return points


    def _init_line(self):
        """Initialize an empty and invisible line plot."""
        line = k3d.line(
            vertices = self._empty_coordinates(),
            colors   = self._empty_colors(), 
            width    = 0.1, 
            shader   = "mesh"
        )
        line.visible = False
        return line


    def _init_active_annotation(self):
        """Initialize the line and points for the active annotation."""
        active_kwargs = self._prep_active_annotation()
        if active_kwargs is None:
            return ( self._init_line(), self._init_points() )
        return (
            k3d.line(**active_kwargs["line"], shader = "mesh"),
            k3d.points(**active_kwargs["points"], shader = "3d"), 
        )


    def _init_background_annotations(self):
        """Initialize the points plot for the background annotations."""
        background_kwargs = self._prep_background_annotations()
        if background_kwargs is None:
            return ( self._init_line(), self._init_points() )
        return (
            k3d.line(**background_kwargs["line"], shader = "mesh"),
            k3d.points(**background_kwargs["points"], shader = "3d")
        )

    # Prepare Cortex Methods ---------------------------------------------------

    def _prep_cortex(self):
        """Prepare the data dict for the cortex mesh k3d object."""
        vertices  = self.viewer.coordinates.T
        indices   = self.viewer.faces
        curvature = self._rgb_to_k3dcolor(self.viewer.curvature)
        return { 
            "vertices" : vertices.astype(np.float32), 
            "indices"  : indices.T.astype(np.uint32), 
            "colors"   : curvature.astype(np.uint32) 
        }
    

    # Prepare Overlay Methods --------------------------------------------------

    def _prep_overlay(self):
        """Prepare the data dict for the cortex overlay mesh k3d object."""
        if self.viewer.style["overlay"] == "curvature":
            return None
        return {
            **self._prep_cortex(),
            "colors"  : self._rgb_to_k3dcolor(self.viewer.overlay),
            "opacity" : float(self.viewer.style["overlay_alpha"])
        }
    
    # Prepare Active Points Methods ---------------------------------------------------

    def _prep_active_annotation(self):
        """Prepare the data for the active annotation."""
        # Get the currnet active surface annotation
        annotation         = self.state.annotation
        surface_annotation = self.state.surface_annotations[annotation]

        # If no coordinates, return None to skip plotting.
        coordinates = surface_annotation.get("coordinates", None)
        if coordinates is None or coordinates.shape[1] == 0: return None

        # Get the annotation style from the styler (active = None)
        # If not visible, return None to skip plotting.
        annotation_style = self.state.styler(None)
        if not annotation_style["visible"]: return None

        # Get number of annotation vertex (line) coordinates and point types
        vertices    = coordinates.T.astype(np.float32)
        positions   = vertices.copy() # copy!
        point_types = surface_annotation.get("point_types", None)

        # Check if vertices are all fixed points, skip lines if so. 
        if np.all(point_types == self.state.POINT_FIXED):
            vertices = self._empty_coordinates() # set vertices to empty to skip line plotting

        # Get annotation positions (for points) and point types
        positions   = positions[point_types != self.state.POINT_INTERP]
        point_types = point_types[point_types != self.state.POINT_INTERP]
        n_points    = positions.shape[0]

        # Prepare scatter sizes by points type (slightly larger fixed points)
        point_sizes = np.full(n_points, self.state.style["point_size"])
        point_sizes[point_types == self.state.POINT_FIXED] = self.state.style["point_size"] * 1.25

        # Prepare colors for each annotation point
        annotation_color = self._rgb_to_k3dcolor(annotation_style["color"])

        # Return the active annotation plot keyword arguments by plot type
        return { 
            "line": {
                "vertices" : vertices.astype(np.float32),
                "width"    : float(self.state.style["line_width"]),
                "colors"   : np.full(vertices.shape[0], annotation_color, dtype = np.uint32)
            },
            "points": {
                "positions"   : positions.astype(np.float32), 
                "point_sizes" : point_sizes.astype(np.float32), 
                "colors"      : np.full(n_points, annotation_color, dtype = np.uint32)
            }
        }

    # Prepare Background Points Methods ----------------------------------------

    def _prep_background_annotations(self):
        """Prepare the data for the background annotations."""
        # Get the list of annotations excluding the selected one
        annotation      = self.state.annotation
        annotation_list = list(self.state.surface_annotations.keys())
        annotation_list.remove(annotation)
        
        # Initialize empty arrays for all coordinates and colors
        all_vertices  = np.empty((0, 3)) 
        all_positions = np.empty((0, 3))
        all_lcolors   = np.empty((0,), dtype = np.uint32)
        all_pcolors   = np.empty((0,), dtype = np.uint32)

        # Initailize NaN array to separate annotations (for line plotting)
        coord_sep = np.full((1, 3), np.nan)
        color_sep = np.array([0], dtype = np.uint32)

        for annotation in annotation_list: # for each annotation
            # Get the surface annotation and style for the annotation
            surface_annotation = self.state.surface_annotations[annotation]

            # If no coordinates, skip processing.
            coordinates = surface_annotation.get("coordinates", None)
            if coordinates is None or coordinates.shape[1] == 0: continue

            # Get the annotation style from the styler (active = None)
            # If not visible, return None to skip plotting.
            annotation_style = self.state.styler(annotation)
            if not annotation_style["visible"]: continue

            # Get annotation color and point types for the current annotation
            annotation_color = self._rgb_to_k3dcolor(annotation_style["color"])
            point_types = surface_annotation.get("point_types", None)

            # Get number of annotation vertex (line) coordinates and point types
            vertices  = coordinates.T.astype(np.float32)
            positions = vertices.copy() # copy!

            # Check if not all vertices are all fixed points, all to lines.
            if not np.all(point_types == self.state.POINT_FIXED):
                # Prepare the vertices and line colors arrays
                all_vertices  = np.vstack((all_vertices, vertices, coord_sep))
                vertex_colors = np.full(vertices.shape[0], annotation_color)
                all_lcolors   = np.hstack((all_lcolors, vertex_colors, color_sep))

            # Get annotation positions (for points) and point types
            positions   = positions[point_types != self.state.POINT_INTERP]
            point_types = point_types[point_types != self.state.POINT_INTERP]

            # Prepare the positions and point colors arrays
            all_positions = np.vstack((all_positions, positions))
            point_colors  = np.full(positions.shape[0], annotation_color)
            all_pcolors   = np.hstack((all_pcolors, point_colors))
    
        # If no coordinates, return None to skip plotting.
        if all_vertices.shape[0] == 0: return None

        return { 
            "line": {
                "vertices" : all_vertices.astype(np.float32),
                "width"    : float(self.state.style["line_width"] * 0.5), 
                "colors"   : all_lcolors.astype(np.uint32)
            },
            "points": {
                "positions"  : all_positions.astype(np.float32), 
                "point_size" : float(self.state.style["point_size"] * 0.5),
                "colors"     : all_pcolors.astype(np.uint32)
            }
        }
    
    # Figure Clear Method ------------------------------------------------------

    def clear_figure(self):
        """Clear the figure by setting layers to invisible."""
        self.k3dmesh_cortex.visible       = False
        self.k3dmesh_overlay.visible      = False
        self.k3dline_active.visible       = False
        self.k3dpoints_active.visible     = False
        self.k3dline_background.visible   = False
        self.k3dpoints_background.visible = False 

    # Figure Refresh Methods ---------------------------------------------------

    def refresh_cortex(self):
        """Refresh the cortex mesh and overlay layers."""
        # Update the cortex mesh.
        cortex_kwargs = self._prep_cortex()
        for key, val in cortex_kwargs.items():
            setattr(self.k3dmesh_cortex, key, val)
        self.k3dmesh_cortex.visible = True

        # Update the overlay mesh.
        overlay_kwargs = self._prep_overlay()
        if overlay_kwargs is None:
            self.k3dmesh_overlay.visible = False
        else:
            for key, val in overlay_kwargs.items():
                setattr(self.k3dmesh_overlay, key, val)
            self.k3dmesh_overlay.visible = True
    

    def refresh_points(self):
        """Refresh the active and background annotation layers."""
        # Active annotation.
        active_kwargs = self._prep_active_annotation()
        if active_kwargs is None:
            self.k3dline_active.visible   = False
            self.k3dpoints_active.visible = False
        else:
            # Active annotation lines
            for key, val in active_kwargs["line"].items(): 
                setattr(self.k3dline_active, key, val)
            self.k3dline_active.visible = True

            # Active annotation points
            for key, val in active_kwargs["points"].items():
                setattr(self.k3dpoints_active, key, val)
            self.k3dpoints_active.visible = True
    
        # Background annotations.
        background_kwargs = self._prep_background_annotations()
        if background_kwargs is None:
            self.k3dline_background.visible   = False
            self.k3dpoints_background.visible = False
        else:
            # Background annotation lines
            for key, val in background_kwargs["line"].items():
                setattr(self.k3dline_background, key, val)
            self.k3dline_background.visible = True

            # Background annotation points
            for key, val in background_kwargs["points"].items():
                setattr(self.k3dpoints_background, key, val)
            self.k3dpoints_background.visible = True


    def refresh_figure(self, clear = False, cortex = True, points = True):
        """Refresh the 3D figure."""
        # Disable auto-rendering to batch updates and avoid intermediate renders.
        self.figure.auto_rendering = False

        # Apply updates to the figure layers based on the specified flags.
        if clear:  self.clear_figure()
        if cortex: self.refresh_cortex()
        if points: self.refresh_points()
        
        # Re-enable auto-rendering and trigger a single render after all updates are applied.
        self.figure.auto_rendering = True
        self.figure.render()


# The Cortex Viewer Widget -----------------------------------------------------



    #     # Assign information box observers
    #     for key in self._infobox_observers.keys():
    #         self._infobox_observers[key](partial(self.on_selection_change, key))

    #     # Assign user annotation input observers
    #     self.state.observe_annotation_change(self.on_annotation_change)

    #     # Assign style option observers
    #     for key in self._style_observers.keys():
    #         self._style_observers[key](partial(self.on_style_change, key)) 


    # @property
    # def _infobox_observers(self):
    #     """Return a list of observer functions for the Cortex Viewer state."""
    #     return {
    #         "targets"    : self.state.observe_targets,
    #         "annotation" : self.state.observe_annotation,
    #     } 


    # @property
    # def _style_observers(self):
    #     """Return a list of observer functions for the Cortex Viewer style."""
    #     return {
    #         # "inflation_percent" : self.control_panel.observe_inflation_slider, 
    #         # "overlay"           : self.control_panel.observe_overlay_dropdown, 
    #         # "overlay_alpha"     : self.control_panel.observe_overlay_slider, 
    #         # "point_size"        : self.control_panel.observe_point_size_slider, 
    #         # "line_width"        : self.control_panel.observe_line_width_slider,
    #         # "line_interp"       : self.control_panel.observe_line_interp_slider, 
    #         "annotation_style"  : self.state.observe_annotation_styles,
    #     }
    

    # def on_selection_change(self, key, change):
    #     """Handle changes to the dataset selection."""
    #     # Update the control panel information
    #     if key == "targets":
    #         self.state.targets    = self.state.get_targets()
    #         self.state.annotation = self.state.get_annotation()
    #     else: # key == "annotation":
    #         self.state.annotation = self.state.get_annotation()


    #     # Update the cortex viewer state based on selection change
    #     if key == "targets":
    #         self.state.update_annot_cfg()
    #         self.state.update_styler()
    #         self.state.update_participant()
    #         self.state.update_coordinates()
    #         self.state.update_mesh()
    #         self.state.update_overlay()
    #         self.state.update_flatmap_annotations()
    #         self.state.update_surface_annotations()
    #         clear, cortex, points = True, True, True
    #     else: # key == "annotation"
    #         clear, cortex, points = False, False, True

    #     # Refresh the figure.
    #     self.figure_panel.refresh_figure(
    #         clear = clear, cortex = cortex, points = points)


    # def on_annotation_change(self, _):
    #     """Handle update when the user changes the annotation data."""
    #     # Update the surface annotations based on the new annotation data
    #     self.state.update_surface_annotations()
        
    #     # Refresh the figure with annotation changes
    #     self.figure_panel.refresh_figure(
    #         clear = False, cortex = False, points = True)
        

    # def on_style_change(self, key, change):
    #     """Handle changes to the style option changes."""
    #     # Change style based on key value
    #     if key != "annotation_style":
    #         self.state.style[key] = change.new

    #     # Update the mesh color based on the new overlay
    #     if key == "inflation_percent":
    #         self.state.update_coordinates()
    #         self.state.update_mesh()
    #         self.state.update_overlay()
    #         self.state.update_surface_coordinates()
    #         clear, cortex, points = False, True, True
    #     elif key == "overlay":
    #         self.state.update_overlay()
    #         clear, cortex, points = False, True, False
    #     elif key == "overlay_alpha":
    #         clear, cortex, points = False, True, False
    #     elif key in ( "point_size", "line_width" ):
    #         clear, cortex, points = False, False, True
    #     elif key == "line_interp":
    #         self.state.update_surface_annotations()
    #         clear, cortex, points = False, False, True
    #     elif key == "annotation_style":
    #         clear, cortex, points = False, False, True
    #     else: 
    #         raise ValueError(f"Invalid style key: {key}")
            
    #     # Update the figure with updated state
    #     self.figure_panel.refresh_figure(
    #         clear = clear, cortex = cortex, points = points)
