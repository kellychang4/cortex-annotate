# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer.py

"""
Implementation code for the Cortex Viewer.
"""

# Imports ----------------------------------------------------------------------

import k3d
import glob
import numpy as np
import os.path as op
import neuropythy as ny
import ipywidgets as ipw
from struct import unpack
from functools import partial
from matplotlib.colors import to_rgb
from neuropythy.geometry.util import barycentric_to_cartesian

# The Cortex Viewer State ------------------------------------------------------

class CortexViewerState:
    """
    The state object for the Cortex Viewer tool.
    """
    
    __FLATMAP_KWARGS = {
        "mask"      : ( "parcellation", 43 ), 
        "map_right" : "right", 
        "radius"    : np.pi / 2
    }

    __PROPERTY_KWARGS = {
        "angle" : { "func": lambda prop, h: { f"polar_angle_{h}": prop }, 
                     "kwargs": { } },
        "eccen" : { "func": lambda prop, _: { "eccentricity": prop }, 
                     "kwargs": { } },
        "vexpl" : { "func": lambda prop, _: prop, 
                     "kwargs": { "cmap": "hot", "vmin": 0, "vmax": 100 } },
    }

    # Point type constants
    POINT_FIXED  = 2  # fixed head/tail
    POINT_USER   = 1  # user-placed
    POINT_INTERP = 0  # interpolated


    def __init__(
            self, 
            annotation_tool, 
            dataset_directory = "/home/jovyan/datasets", 
        ):
        
        # Store intialization variables
        self.annotation_tool   = annotation_tool 
        self.dataset_directory = dataset_directory

        # Prepare fsaverage da_ta 
        self.fsaverage = self._get_fsaverage()

        # Prepare initial values from annotation widget
        self.targets    = self.get_targets()
        self.annotation = self.get_annotation()

        # Prepare dataset dependent annotation information and styler
        self.update_annot_cfg()
        self.update_styler() 

        # Set cortex viewer style (default) options
        self.style = {
            "inflation_percent" : 100,
            "overlay"           : "curvature",
            "overlay_alpha"     : 1.0, 
            "point_size"        : 1.5, 
            "line_width"        : 0.25,
            "line_interp"       : 10,
        }

        # Prepare participant 3d coordinate and mesh data
        self.update_participant()
        self.update_coordinates()
        self.update_mesh()
        self.update_overlay()

        # Extract the flatmap and annotation data
        self.update_flatmap_annotations()

        # Prepare surface annotations and path 
        self.update_surface_annotations() 


    # [Cortex Viwer] fsaverage Method ------------------------------------------

    def _get_fsaverage(self):
        """load fsaverage tesselations and inflated surface coordinates."""
        # Load the fsaverage object
        fsa = ny.freesurfer_subject("fsaverage")
        
        # Return fsaverage dictionary values
        return {
            h: {
                "tesselation": fsa.hemis[h].tess.faces, 
                "inflated"   : fsa.hemis[h].surface("inflated").coordinates, 
                "flatmap"    : fsa.hemis[h].mask_flatmap(
                    **CortexViewerState.__FLATMAP_KWARGS)
            } for h in ( "lh", "rh" )
        }

    # [Annotation Tool] Get Active Widget Methods ------------------------------

    def _get_active_selection_panel(self):
        """Get the active selection panel widget."""
        return self.annotation_tool.control_panel.selection_panel
    

    def _get_active_figure_panel(self):
        """Get the active figure panel widget."""
        return self.annotation_tool.figure_panel
    
    # [Annotation Tool] Get Selected Information Methods -----------------------    

    def get_targets(self):
        """Get the active targets from the active annotation tool."""
        selection_panel = self._get_active_selection_panel()
        return { 
            key.lower(): value.value for key, value 
            in selection_panel.target_dropdowns.items() 
        }
    
    @property
    def dataset(self):
        """Get the current dataset selection."""
        return self.targets.get("dataset", None)
    

    @property
    def participant(self):
        """Get the current participant selection."""
        return self.targets.get("participant", None)


    @staticmethod
    def convert_hemisphere(hemisphere):
        """Convert hemisphere string to code or vice versa."""
        str_to_code = { "Left Hemisphere"  : "lh", "Right Hemisphere" : "rh" }
        code_to_str = { v: k for k, v in str_to_code.items() }
        if hemisphere in str_to_code.keys():
            return str_to_code[hemisphere]
        elif hemisphere in code_to_str.keys():
            return code_to_str[hemisphere]
        else: 
            raise ValueError(f"Invalid hemisphere value: {hemisphere}")


    @property
    def hemisphere(self):
        """Get the current hemisphere selection."""
        hemisphere = self.targets.get("hemisphere", None)
        if hemisphere is not None:
            return self.convert_hemisphere(hemisphere)
        return hemisphere
    

    def get_annotation(self):
        """Get the annotation from the active annotation tool."""
        selection_panel = self._get_active_selection_panel()
        return selection_panel.annotation


    def get_style_annotation(self):
        """Get the current style annotation selection widget."""
        return self.style_panel.annotation
    
    # [Annotation Tool] Update State Methods -----------------------------------

    def update_annot_cfg(self):
        """Update the annotation configuration based on current state."""
        self.annot_cfg = self.annotation_tool.state.config.annotations


    def update_styler(self):
        """Update the styler options based on current state."""
        self.styler = self.annotation_tool.state.style


    def update_flatmap_annotations(self):
        """Get the annotation tool's annotation dictionary"""
        figure_panel = self._get_active_figure_panel()
        self.flatmap_annotations = figure_panel.annotations    

    # [Cortex Viewer] Update Participant ---------------------------------------

    def _read_coordinates(self, filename):
        """Read cortical mesh coordinates from a <hemisphere>.3d.coordinates file."""
        with open(filename, "rb") as f:
            values = f.read() # load file content
            fstr = "e" * (len(values) // 2)
            coordinates = np.array(unpack(fstr, values)).reshape((3, -1))
        return coordinates
    

    def _read_property(self, filename):
        """Read cortical property data from a <hemisphere>.3d.<property> file."""
        with open(filename, "rb") as f:
            values = f.read() # load file content
            fstr = "e" * (len(values) // 2)
            prop = np.array(unpack(fstr, values)).reshape((-1, ))
        return prop
    

    def update_participant(self): 
        """Load participant dataset based on current state."""
        # Locate participant directory and files
        participant_dir = op.join(
            self.dataset_directory, self.dataset.lower(), self.participant)
        filenames = glob.glob(op.join(participant_dir, f"{self.hemisphere}.3d.*"))

        # Load participant midgray coordinates
        coordinates_file = [x for x in filenames if x.endswith("coordinates")][0]
        self.midgray = self._read_coordinates(coordinates_file)

        # Load remaining property files
        property_files = [x for x in filenames if x != coordinates_file]
        self.properties = {
            fname.split(".")[-1]: self._read_property(fname)
            for fname in property_files
        }


    def update_coordinates(self):
        """Update the cortical mesh coordinates based on the inflation value."""
        inflated_coordinates = self.fsaverage[self.hemisphere]["inflated"]
        self.coordinates = ((inflated_coordinates - self.midgray) * \
             (self.style["inflation_percent"] / 100.0)) + self.midgray


    def update_mesh(self):
        """Update the cortical mesh object based on the current state."""
        # Update the mesh with current coordinates and properties
        self.mesh = ny.geometry.Mesh(
            faces       = self.fsaverage[self.hemisphere]["tesselation"],
            coordinates = self.coordinates,
            properties  = self.properties
        )

        # Assign curvature color (for default coloring)
        self.curvature = ny.graphics.cortex_plot_colors(self.mesh)[:, :3]    

    
    def update_overlay(self):
        """Update the cortical mesh color based on curvature."""
        if self.style["overlay"] == "curvature":
            self.overlay = None
        else: # Get property kwargs and values 
            overlay_style = self.style["overlay"]
            prop_kwargs   = CortexViewerState.__PROPERTY_KWARGS[overlay_style]
            prop_values   = self.mesh.properties[overlay_style]
            prop_color    = prop_kwargs["func"](prop_values, self.hemisphere)
            self.overlay  = ny.graphics.cortex_plot_colors(self.mesh, 
                color = prop_color, **prop_kwargs["kwargs"])[:, :3]

    # [Cortex Viewer] Update Surface Coordinates -------------------------------

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
                self.surface_annotations[key] = {
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
            self.surface_annotations[key] = {
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
            surface_annotation = self.surface_annotations.get(key, {})
            flatmap_address = surface_annotation.get("addresses", None)

            # If no surface addresses, set surface annotation coordinates to None
            if flatmap_address is not None:
                # Calculate surface coordinates
                surface_coordinates = (
                    self._flatmap_to_surface(flatmap_address, self.coordinates))
                
                # Store surface annotation coordinates
                self.surface_annotations[key]["coordinates"] = surface_coordinates

        
    def update_surface_annotations(self, annotations = None):
        """Update cortical surface annotations based on current state."""
        # Initialize surface annotations dictionary if not present
        if annotations is None: self.surface_annotations = {} 

        # Get the list of annotations to update
        self.update_surface_addresses(annotations)
        self.update_surface_coordinates(annotations)

    # [Cortex Viewer] Observer Methods -----------------------------------------

    def observe_targets(self, callback):
        """Assign a callback function to target value changes."""
        selection_panel = self.annotation_tool.control_panel.selection_panel
        for target_dropdown in selection_panel.target_dropdowns.values():
            target_dropdown.observe(callback, names = "value")


    def observe_annotation(self, callback):
        """Assign a callback function to annotation data changes."""
        selection_panel = self.annotation_tool.control_panel.selection_panel
        annotations_dropdown = selection_panel.annotations_dropdown
        annotations_dropdown.observe(callback, names = "value")


    def observe_annotation_change(self, callback):
        """Assign a callback function to annotation data changes."""
        annotation_figure = self.annotation_tool.figure_panel
        annotation_figure.observe(callback, names = "_annotation_change")


    def observe_annotation_styles(self, callback):
        """Assign a callback function to style option changes."""
        style_panel = self.annotation_tool.control_panel.style_panel
        style_panel.visible_checkbox.observe(callback, names = "value")
        style_panel.color_picker.observe(callback, names = "value")


# Cortex Viewer Figure Panel ---------------------------------------------------

class CortexViewerPanel(ipw.VBox):
    """Cortex Figure Panel.

    The panel that contains the 3D cortex plot for the Cortex Viewer tool.
    """
    
    def __init__(self, state, width = 512, height = 512):
        # Store the state.
        self.state = state

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

        # Initialize the VBox with the figure as the child 
        super().__init__(
            children = [ self.figure ], 
            layout = {
                "width" : f"{width}px", 
                "height": f"{height}px", 
                "border": "1px solid magenta"
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
