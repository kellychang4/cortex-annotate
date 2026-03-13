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

# Cortex Viewer Figure Panel ---------------------------------------------------

class CortexViewerPanel(ipw.VBox):
    """The 3D cortex viewer that displays cortical mesh and annotations.

    The CortexViewerPanel manages a multi-layer k3d figure for rendering:
        Layer 0: cortex mesh with curvature
        Layer 1: cortex overlays (optional)
        Layer 2: active annotation (lines)
        Layer 3: active annotation (points)
        Layer 4: background annotations (lines)
        Layer 5: background annotations (points)
    """

    def __init__(self, figure_state, width = 512, height = 512):
        """Initialize the cortex viewer panel."""
        # Store figure state.
        self.state = figure_state 

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
            # NOTE: k3d does not support alpha in the color integer, ignore
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
            width    = float(self.state.viewer.style["line_width"]),
            shader   = "mesh"
        )
        line.visible = False
        return line


    # Prepare Cortex Methods ---------------------------------------------------

    def _prep_cortex(self):
        """Prepare the data dict for the cortex mesh k3d object."""
        curvature = self._rgb_to_k3dcolor(self.state.viewer.overlays["curvature"])
        return { 
            "vertices" : self.state.viewer.coordinates.astype(np.float32), 
            "indices"  : self.state.viewer.faces.astype(np.uint32), 
            "colors"   : curvature.astype(np.uint32) 
        }
    
    # Prepare Overlay Methods --------------------------------------------------

    def _prep_overlay(self):
        """Prepare the data dict for the cortex overlay mesh k3d object."""
        # If overlay style is curvature, no additional overlay
        overlay_name = self.state.viewer.style["overlay"]
        if overlay_name == "curvature": return None

        # Else, get overlay values and return with opactity.
        overlay_values = self.state.viewer.overlays[overlay_name]
        return {
            **self._prep_cortex(),
            "colors"  : self._rgb_to_k3dcolor(overlay_values),
            "opacity" : float(self.state.viewer.style["overlay_alpha"])
        }
    
    # Prepare Active Points Methods ---------------------------------------------------

    def _prep_active_annotation(self):
        """Prepare the data for the active annotation."""
        # Get the current active viewer annotation
        annotation        = self.state.active
        viewer_annotation = self.state.viewer.annotations[annotation]

        # If no coordinates, return None to skip plotting.
        coordinates = viewer_annotation.get("coordinates", None)
        if coordinates is None or coordinates.shape[0] == 0: return None

        # Get the annotation style from the styler (active = None)
        # If not visible, return None to skip plotting.
        annotation_style = self.state.style(None)
        if not annotation_style["visible"]: return None

        # Get number of annotation vertex (line) coordinates and point types
        vertices    = coordinates.astype(np.float32)
        positions   = vertices.copy() # copy!
        point_types = viewer_annotation.get("point_types", None)

        # Check if vertices are all fixed points, skip lines if so. 
        if np.all(point_types == self.state.POINT_FIXED):
            vertices = self._empty_coordinates() # set vertices to empty to skip line plotting

        # Get annotation positions (for points) and point types
        positions   = positions[point_types != self.state.POINT_INTERP]
        point_types = point_types[point_types != self.state.POINT_INTERP]
        n_points    = positions.shape[0]

        # Prepare scatter sizes by points type (slightly larger fixed points)
        point_sizes = np.full(n_points, self.state.viewer.style["point_size"])
        point_sizes[point_types == self.state.POINT_FIXED] = \
            self.state.viewer.style["point_size"] * 1.25

        # Prepare colors for each annotation point
        annotation_color = self._rgb_to_k3dcolor(annotation_style["color"])

        # Return the active annotation plot keyword arguments by plot type
        return { 
            "line": {
                "vertices" : vertices.astype(np.float32),
                "width"    : float(self.state.viewer.style["line_width"]),
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
        # Get the list of annotations excluding the active one
        annotation      = self.state.active
        annotation_list = self.state.annot_cfg.names.copy()
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
            # Get the current viewer annotation.
            viewer_annotation = self.state.viewer.annotations[annotation]

            # If no coordinates, skip processing.
            coordinates = viewer_annotation.get("coordinates", None)
            if coordinates is None or coordinates.shape[0] == 0: continue

            # Get the annotation style from the styler (active = None)
            # If not visible, return None to skip plotting.
            annotation_style = self.state.style(annotation)
            if not annotation_style["visible"]: continue

            # Get annotation color and point types for the current annotation
            annotation_color = self._rgb_to_k3dcolor(annotation_style["color"])
            point_types = viewer_annotation.get("point_types", None)

            # Get number of annotation vertex (line) coordinates and point types
            vertices  = coordinates.astype(np.float32)
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
                "width"    : float(self.state.viewer.style["line_width"] * 0.5), 
                "colors"   : all_lcolors.astype(np.uint32)
            },
            "points": {
                "positions"  : all_positions.astype(np.float32), 
                "point_size" : float(self.state.viewer.style["point_size"] * 0.5),
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
    

    def refresh_points(self, active = True, background = False):
        """Refresh the active and background annotation layers."""
        # Active annotation.
        if active:
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
        if background: 
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


    # Redraw Viewer Method -----------------------------------------------------

    def redraw_viewer(self, clear = False, cortex = False, active = True, background = False):
        """Redraw the entire canvas panel."""
        # Disable auto-rendering to batch updates and avoid intermediate renders.
        self.figure.auto_rendering = False

        # Apply updates to the figure layers based on the specified flags.
        if clear:  self.clear_figure()
        if cortex: self.refresh_cortex()
        if active or background:
            self.refresh_points(active = active, background = background)

        # Re-enable auto-rendering and trigger render after all updates are applied.
        self.figure.auto_rendering = True
        self.figure.render()