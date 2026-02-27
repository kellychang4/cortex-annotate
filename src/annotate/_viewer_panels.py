# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer_panels.py

"""
Core implementation code for the CortexViewer's subpanels. 

CortexViewer is implemented in terms of two main subpanels:
1) CortexControlPanel: The control panel that contains viewer information about 
    the dataset, target keys, and annotations, as well as user controls for 
    adjusting the inflation level of the cortex plot.
2) CortexFigurePanel: The figure panel that contains the 3D cortex plot.

Each panel is implemented as an ipywidget, and the CortexViewer class
assembles these two panels into a single interface.

CortexViewer is implmented in _viewer.py.
"""

# Imports ----------------------------------------------------------------------

import k3d
import numpy as np
import ipywidgets as ipw
from matplotlib.colors import to_rgb

# The Cortex Figure Panel ------------------------------------------------------

class CortexFigurePanel(ipw.GridBox):
    """Cortex Figure Panel.

    The panel that contains the 3D cortex plot for the Cortex Viewer tool.
    """
    
    def __init__(self, state, width = None, height = 512):
        # Store the state.
        self.state = state

        # Create a figure background (k3d plot)
        self.figure = k3d.plot(
            # K3d does not use width, just fills the space
            height            = height, 
            grid_visible      = False,
            camera_auto_fit   = False,
            menu_visibility   = False,
            camera_fov        = 60,
            axes_helper       = 0, # remove axes direction helper
            camera_zoom_speed = 1.5,
        )

        # Create and add the cortex mesh to the figure
        self.k3dmesh_cortex = self._init_cortex()

        # Create and add the overlay mesh to the figure
        self.k3dmesh_overlay = self._init_overlay()
        
        # Create active annotation to figure 
        self.k3dline_active, self.k3dpoints_active = \
            self._init_active_annotation()

        # Create background annotations to figure
        self.k3dline_background, self.k3dpoints_background = \
            self._init_background_annotations()

        # Add the meshes and points to the figure and initial render
        self.figure += self.k3dmesh_cortex
        self.figure += self.k3dmesh_overlay 
        self.figure += self.k3dline_active
        self.figure += self.k3dpoints_active 
        self.figure += self.k3dline_background
        self.figure += self.k3dpoints_background 

        # Set initial camera values
        self.figure.camera = [-160, -10, -6, 15, -30, 0, 0, 0, 1]

        # Define layout (GridBox as HBox)
        layout = ipw.Layout(
            grid_template_columns = "1fr",
            height = f"{height}px",
            width  = f"{width}px" if width is not None else "100%"
        )
        super().__init__([ self.figure ], layout = layout)
        self.figure.camera_auto_fit = False


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
        """Prepare the data for the cortex mesh."""
        vertices  = self.state.coordinates.T # (n_vertices, 3)
        indices   = self.state.fsaverage[self.state.hemisphere]["tesselation"]
        curvature = self._rgb_to_k3dcolor(self.state.curvature)
        return { 
            "vertices" : vertices.astype(np.float32), 
            "indices"  : indices.T.astype(np.uint32), 
            "colors"   : curvature.astype(np.uint32) 
        }
    
    # Prepare Overlay Methods --------------------------------------------------

    def _prep_overlay(self):
        """Prepare the data for the cortex overlay mesh."""
        # If no overlay, return None
        if self.state.style["overlay"] == "curvature":
            return None

        # Else return the overlay mesh keyword arguments
        return {
            **self._prep_cortex(), # same vertices + indices
            "colors"   : self._rgb_to_k3dcolor(self.state.overlay),
            "opacity"  : float(self.state.style["overlay_alpha"])
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
        """Refresh the cortex mesh."""
        # Update the cortex mesh values
        cortex_kwargs = self._prep_cortex()
        for key in cortex_kwargs.keys():
            setattr(self.k3dmesh_cortex, key, cortex_kwargs[key])
        self.k3dmesh_cortex.visible = True

        # Update the overlay mesh values 
        overlay_kwargs = self._prep_overlay()
        if self.state.style["overlay"] == "curvature":
            self.k3dmesh_overlay.visible = False
        else:
            for key in overlay_kwargs.keys():
                setattr(self.k3dmesh_overlay, key, overlay_kwargs[key])
            self.k3dmesh_overlay.visible = True

    
    def refresh_points(self):
        """Refresh the annotation points."""
        # Update the surface active annotation
        active_kwargs = self._prep_active_annotation()
        if active_kwargs is None:
            self.k3dline_active.visible   = False
            self.k3dpoints_active.visible = False
        else:
            # Update the active line layer (interpolated between points)
            line_kwargs = active_kwargs["line"]
            for key in line_kwargs.keys(): 
                setattr(self.k3dline_active, key, line_kwargs[key])
            self.k3dline_active.visible = True

            # Update the active points layer (fixed + user points)
            points_kwargs = active_kwargs["points"]
            for key in points_kwargs.keys():
                setattr(self.k3dpoints_active, key, points_kwargs[key])
            self.k3dpoints_active.visible = True
    
        # Update the surface background annotation
        background_kwargs = self._prep_background_annotations()
        if background_kwargs is None:
            self.k3dline_background.visible   = False
            self.k3dpoints_background.visible = False
        else:
            # Update the background line layer (interpolated between points)
            line_kwargs = background_kwargs["line"]
            for key in line_kwargs.keys():
                setattr(self.k3dline_background, key, line_kwargs[key])
            self.k3dline_background.visible = True

            # Update the background points layer (fixed + user points)
            points_kwargs = background_kwargs["points"]
            for key in points_kwargs.keys():
                setattr(self.k3dpoints_background, key, points_kwargs[key])
            self.k3dpoints_background.visible = True
            

    def refresh_figure(self, clear = False, cortex = True, points = True):
        """Refresh the entire figure."""
        # Disable auto rendering for performance
        self.figure.auto_rendering = False

        # Figure refresh, dependent on which layers need to be updated
        if clear:
            self.clear_figure()
        if cortex:
            self.refresh_cortex()
        if points:
            self.refresh_points()
        
        # Re-enable auto rendering after cortex and overlay values
        self.figure.auto_rendering = True
        self.figure.render()
