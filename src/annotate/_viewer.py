# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer.py

"""
Implementation code for the Cortex Viewer.
"""

# Imports ----------------------------------------------------------------------

import glob
import numpy as np
import os.path as op
import neuropythy as ny
import ipywidgets as ipw
from struct import unpack
from functools import partial
from neuropythy.geometry.util import barycentric_to_cartesian

from ._viewer_panels import CortexControlPanel, CortexFigurePanel

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


    def __init__(
            self, 
            annotation_widgets, 
            dataset_directory = "/home/jovyan/datasets", 
            inflation_value   = 100
        ):
        
        # Store intialization variables
        self.annotation_widgets = annotation_widgets 
        self.dataset_directory  = dataset_directory

        # Prepare initial values from annotation widget
        self.dataset_index    = self.get_dataset_index()
        self.dataset          = self.get_dataset_value()
        self.participant      = self.get_participant_value()
        self.hemisphere       = self.get_hemisphere_value() 
        self.annotation       = self.get_annotation_value()

        # Prepare fsaverage data 
        self.fsaverage = self._get_fsaverage()

        # Set cortex viewer control panel values
        self.inflation_value  = inflation_value 
        self.overlay          = "curvature" # None = curvature by default
        
        #TODO: optimize everything below this!!!
        # Prepare participant 3d coordinate and mesh data
        self.midgray, self.properties = self.load_participant()
        self.coordinates = self.update_coordinates()
        self.mesh        = self.update_mesh()
        self.color       = self.update_color()

        # Prepare the annotations data
        self.flatmap_annotations = self.get_flatmap_annotations()
        self.surface_annotations = self.update_surface_annotations()
        self.surface_paths       = self.update_surface_paths()


    def get_dataset_index(self):
        """Get the current dataset selection index."""
        return self.annotation_widgets.selected_index
    

    def get_dataset_value(self):
        """Get the current dataset selection widget."""
        return self.annotation_widgets.titles[self.dataset_index]
    
    
    def _get_active_annotation_tool(self):
        """Get the active multicanvas widget."""
        return self.annotation_widgets.children[self.dataset_index]
  

    def _get_active_selection(self):
        """Get the active selection panel widget."""
        active_widget = self._get_active_annotation_tool()
        return active_widget.control_panel.selection_panel
    
  
    def get_participant_value(self):
        """Get the current participant selection widget."""
        active_selection = self._get_active_selection()
        return active_selection.children[0].value
    

    def format_hemisphere(self, hemisphere):
        """Format hemisphere value for internal usage."""
        if hemisphere.lower().startswith("l"): return "lh" 
        return "rh" # else return "rh"
    

    def get_hemisphere_value(self):
        """Get the current hemisphere selection widget."""
        active_selection = self._get_active_selection()
        hemisphere = active_selection.children[1].value.lower()
        return self.format_hemisphere(hemisphere)
    

    def get_annotation_value(self):
        """Get the current annotation selection widget."""
        active_selection = self._get_active_selection()
        return active_selection.children[2].value
    
    
    def get_flatmap_annotations(self):
        """Get the annotation tool's annotation dictionary"""
        active_widget = self._get_active_annotation_tool()
        return active_widget.figure_panel.annotations
    

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


    @classmethod
    def _read_coordinates(cls, filename):
        """Read cortical mesh coordinates from a <hemisphere>.3d.coordinates file."""
        with open(filename, "rb") as f:
            values = f.read() # load file content
            fstr = "e" * (len(values) // 2)
            coordinates = np.array(unpack(fstr, values)).reshape((3, -1))
        return coordinates
    

    @classmethod
    def _read_property(cls, filename):
        """Read cortical property data from a <hemisphere>.3d.<property> file."""
        with open(filename, "rb") as f:
            values = f.read() # load file content
            fstr = "e" * (len(values) // 2)
            prop = np.array(unpack(fstr, values)).reshape((-1, ))
        return prop


    def load_participant(self): 
        """Load participant dataset based on current state."""
        # Locate participant directory and files
        participant_dir = op.join(
            self.dataset_directory, self.dataset.lower(), self.participant)
        filenames = glob.glob(op.join(participant_dir, f"{self.hemisphere}.3d.*"))

        # Load participant midgray coordinates
        coordinates_file = [x for x in filenames if x.endswith("coordinates")][0]
        midgray = self._read_coordinates(coordinates_file)

        # Load remaining property files
        property_files = [x for x in filenames if x != coordinates_file]
        properties = {} # initialize 
        for fname in property_files: # for each property file
            property_name = fname.split(".")[-1] # get property name
            properties[property_name] = self._read_property(fname)
        return ( midgray, properties )
    

    def update_coordinates(self):
        """Update the cortical mesh coordinates based on the inflation value."""
        return ((self.fsaverage[self.hemisphere]["inflated"] - self.midgray) * \
                (self.inflation_value / 100.0)) + self.midgray


    def update_mesh(self):
        """Update the cortical mesh object based on the current state."""
        return ny.geometry.Mesh(
            faces       = self.fsaverage[self.hemisphere]["tesselation"],
            coordinates = self.coordinates,
            properties  = self.properties
        )
    

    def update_color(self):
        """Update the cortical mesh color based on curvature."""
        if self.overlay == "curvature":
            return ny.graphics.cortex_plot_colors(self.mesh)[:, :3]
        else: # Get property kwargs and values 
            prop_kwargs = CortexViewerState.__PROPERTY_KWARGS[self.overlay]
            prop_value  = self.mesh.properties[self.overlay]
            prop_color  = prop_kwargs["func"](prop_value, self.hemisphere)
            return ny.graphics.cortex_plot_colors(
                self.mesh, color = prop_color, **prop_kwargs["kwargs"])[:, :3]


    #TODO: personally dislike the lack of a return...
    def update_figure(self):
        """Update the cortical mesh rendering based on the current state."""
        self.coordinates = self.update_coordinates()
        self.mesh        = self.update_mesh()
        self.color       = self.update_color()
        

    def update_surface_annotations(self):
        """Update the cortical surface annotation coordinates."""
        # Get flatmap annotation coordinates
        flatmap_coordinates = self.flatmap_annotations[self.annotation]

        # Get flatmap addresses 
        flatmap_addresses = self.fsaverage[self.hemisphere]["flatmap"].address(
            flatmap_coordinates)
        
        # Extract faces and barycentric coordinates
        bary_faces  = flatmap_addresses["faces"] # (3, n_points)
        bary_coords = flatmap_addresses["coordinates"]  # (n_points, 2)

        # Extract bary-relevant vertices from the current mesh
        tx = self.mesh.coordinates[:, bary_faces].T # (n_points, 3, 3)
 
        # Return surface annotation coordinates
        return barycentric_to_cartesian(tx, bary_coords) # (3, n_points)
        
    
    def update_surface_paths(self, n = 10):
        # Initalize surface path matrix
        surface_paths = np.array([[], [], []])

        # Get flatmap annotation coordinates
        flatmap_coords = self.flatmap_annotations[self.annotation]

        if flatmap_coords.shape[0] < 2: # one point = no path
            return surface_paths

        for i in np.arange(flatmap_coords.shape[0] - 1): 
            # Interpolate between the current flatmap coodinates
            curr_coords = flatmap_coords[i:(i+2), :] 
            xs = np.linspace(curr_coords[0, 0], curr_coords[1, 0], n + 2)
            ys = np.linspace(curr_coords[0, 1], curr_coords[1, 1], n + 2)

            # Get flatmap addresses 
            flatmap_addresses = self.fsaverage[self.hemisphere]["flatmap"].address([xs, ys])

            # Extract faces and barycentric coordinates
            bary_faces  = flatmap_addresses["faces"] # (3, n_points)
            bary_coords = flatmap_addresses["coordinates"]  # (n_points, 2)

            # Extract bary-relevant vertices from the current mesh
            tx = self.mesh.coordinates[:, bary_faces].T # (n_points, 3, 3)
    
            # Calculate Barycentric to Cartesian coordinates
            surface_coordinates = barycentric_to_cartesian(tx, bary_coords)

            # Append to coordinates to surface paths matrix
            surface_paths = np.concatenate((surface_paths, surface_coordinates), axis = 1)

        # Return surface paths without duplicated points
        return np.unique(surface_paths, axis = 1)


    def _observe_dataset(self, callback):
        """Assign a callback function to dataset value changes."""
        self.annotation_widgets.observe(callback, names = "selected_index")


    def _observe_participant(self, callback):
        """Assign a callback function to participant value changes."""
        for annotation_widget in self.annotation_widgets.children:
            participant_dropdown = annotation_widget.control_panel.selection_panel.children[0]
            participant_dropdown.observe(callback, names = "value")


    def _observe_hemisphere(self, callback):
        """Assign a callback function to hemisphere value changes."""
        for annotation_widget in self.annotation_widgets.children:
            hemisphere_dropdown = annotation_widget.control_panel.selection_panel.children[1]
            hemisphere_dropdown.observe(callback, names = "value")

    
    def _observe_annotation(self, callback):
        """Assign a callback function to annotation value changes."""
        for annotation_widget in self.annotation_widgets.children:
            annotation_dropdown = annotation_widget.control_panel.selection_panel.children[2]
            annotation_dropdown.observe(callback, names = "value")


    @property
    def _observer_functions(self):
        """Return a list of observer functions for the Cortex Viewer state."""
        return {
            "dataset"     : self._observe_dataset,
            "participant" : self._observe_participant,
            "hemisphere"  : self._observe_hemisphere,
            "annotation"  : self._observe_annotation,
        }   


# The Cortex Viewer Widget -----------------------------------------------------

class CortexViewer(ipw.HBox):
    """
    The cortex viewer tool for the `cortex-annotate` project.

    The `CortexViewer` type handles the 3D Cortex figure that renders the 
    cortical mesh that assists the flatmap viewer.
    """

    def __init__(self, annotation_widgets, dataset_directory, inflation_value = 100):
        # Initialize the Cortex Viewer state
        self.state = CortexViewerState(
            annotation_widgets = annotation_widgets,
            dataset_directory  = dataset_directory,
            inflation_value    = inflation_value,
        )

        # Create the Cortex Viewer control panel
        self.control_panel = CortexControlPanel(self.state)

        # Create the Cortex Viewer figure panel
        self.figure_panel = CortexFigurePanel(self.state)

        # Initialize the HBox with the control panel and figure panel
        super().__init__([self.control_panel, self.figure_panel])

        # Assign dataset, participant, and hemisphere observers
        for k in self.control_panel.infobox.keys():
            self.state._observer_functions[k](partial(self.on_selection_change, k))

        # Assign inflation slider observer
        self.control_panel.observe_inflation_slider(self.on_inflation_slider)

        # Assign overlay dropdown observer
        self.control_panel.observe_overlay_dropdown(self.on_overlay_change)


    def on_selection_change(self, key, change):
        """Handle changes to the dataset selection."""
        # Update the control panel information
        #TODO: this is a mess, please keep up the logic
        if key == "dataset":
            self.state.dataset_index = change.new
            self.state.dataset       = self.state.get_dataset_value()
            self.state.participant   = self.state.get_participant_value()
            self.state.hemisphere    = self.state.get_hemisphere_value() 
            self.state.annotation    = self.state.get_annotation_value()
            self.state.flatmap_annotations = self.state.get_flatmap_annotations()
            self.state.midgray, self.state.properties = self.state.load_participant()
        elif key == "participant":
            self.state.participant   = change.new
            self.state.annotation    = self.state.get_annotation_value()
            self.state.flatmap_annotations = self.state.get_flatmap_annotations()
            self.state.midgray, self.state.properties = self.state.load_participant()
        elif key == "hemisphere":
            self.state.hemisphere    = self.state.format_hemisphere(change.new)
            self.state.annotation    = self.state.get_annotation_value()
            self.state.flatmap_annotations = self.state.get_flatmap_annotations()
            self.state.midgray, self.state.properties = self.state.load_participant()
        elif key == "annotation": 
            self.state.annotation          = change.new
            self.state.flatmap_annotations = self.state.get_flatmap_annotations()
            
        # Update the infobox displays
        for k in self.control_panel.infobox.keys():
            self.control_panel.refresh_infobox(self.state, k)

        # Update the figure panel's mesh values
        # self.state.update_figure()
        self.state.coordinates = self.state.update_coordinates()
        self.state.mesh        = self.state.update_mesh()
        self.state.color       = self.state.update_color()
        self.state.surface_annotations = self.state.update_surface_annotations()
        self.state.surface_paths       = self.state.update_surface_paths()

        # Update the figure
        self.figure_panel.refresh_figure(self.state)


    def on_inflation_slider(self, change):
        """Handle changes to the inflation slider value."""
        # Update the inflation value (internal state)
        self.state.inflation_value = change.new # percent, 0-100%

        # Update the figure panel's mesh values
        # self.state.update_figure()
        self.state.coordinates = self.state.update_coordinates()
        self.state.mesh        = self.state.update_mesh()
        self.state.color       = self.state.update_color()
        
        # Update the annotation coordinates
        self.state.surface_annotations = self.state.update_surface_annotations()
        self.state.surface_paths       = self.state.update_surface_paths()

        # Update the figure with new coordinates
        self.figure_panel.refresh_figure(self.state)


    def on_overlay_change(self, change):
        """Handle changes to the overlay dropdown value."""
        # Change dropdown value 
        self.state.overlay = change.new  

        # Update the mesh color based on the new overlay
        self.state.color = self.state.update_color()
 
        # Update the figure with new mesh properties
        self.figure_panel.refresh_figure(self.state)

