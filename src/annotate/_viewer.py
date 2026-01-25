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

from ._viewer_panels import CortexControlPanel, CortexFigurePanel

# The Cortex Viewer State ------------------------------------------------------

class CortexViewerState:
    """
    The state object for the Cortex Viewer tool.
    """

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

        # Prepare fsaverage data 
        self.fsaverage = ny.freesurfer_subject("fsaverage")
        self.tesselation, self.inflated = self._get_fsaverage()

        # Set cortex viewer control panel values
        self.inflation_value  = inflation_value 
        self.overlay          = "curvature" # None = curvature by default

        # Prepare participant 3d coordinate and mesh data
        self.midgray, self.properties = self.load_participant()
        self.coordinates = self.update_coordinates()
        self.mesh        = self.update_mesh()
        self.color       = self.update_color()

        # Store multicanvas and annotations
        # self.multicanvas = self.get_multicanvas() # get current dataset multicanvas
        # self.annotations = self.get_annotations() # get current dataset annotations


    def get_dataset_index(self):
        """Get the current dataset selection index."""
        return self.annotation_widgets.selected_index
    

    def get_dataset_value(self):
        """Get the current dataset selection widget."""
        return self.annotation_widgets.titles[self.get_dataset_index()]
    
    
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
    

    def _get_fsaverage(self):
        """load fsaverage tesselations and inflated surface coordinates."""
        tesselation = { 
            h: self.fsaverage.hemis[h].tess.faces 
            for h in ( "lh", "rh" )
        }

        inflated = {
            h: self.fsaverage.hemis[h].surface("inflated").coordinates 
            for h in ( "lh", "rh" )
        }
        return tesselation, inflated


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
        return ((self.inflated[self.hemisphere] - self.midgray) * \
                (self.inflation_value / 100.0)) + self.midgray


    def update_mesh(self):
        """Update the cortical mesh object based on the current state."""
        return ny.geometry.Mesh(
            faces       = self.tesselation[self.hemisphere],
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


    def update_figure(self):
        """Update the cortical mesh rendering based on the current state."""
        self.coordinates = self.update_coordinates()
        self.mesh        = self.update_mesh()
        self.color       = self.update_color()


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


    @property
    def observer_functions(self):
        """Return a list of observer functions for the Cortex Viewer state."""
        return {
            "dataset"     : self._observe_dataset,
            "participant" : self._observe_participant,
            "hemisphere"  : self._observe_hemisphere,
        }   


    # def get_multicanvas(self):
    #     """Get the canvas widget for the current dataset."""
    #     active_widget = self._get_active_annotation_tool()
    #     return active_widget.figure_panel.multicanvas


    # def get_annotations(self):
    #     """Get the annotation widgets for the current dataset."""
    #     active_widget = self._get_active_annotation_tool()
    #     return active_widget.figure_panel.annotations


    # def observe_multicanvas_mouse(self, callback):
    #     """Assign a callback function to multicanvas mouse events."""
    #     self.multicanvas.on_mouse_down(callback)


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
            self.state.observer_functions[k](partial(self.on_selection_change, k))

        # Assign inflation slider observer
        self.control_panel.observe_inflation_slider(self.on_inflation_slider)

        # Assign overlay dropdown observer
        self.control_panel.observe_overlay_dropdown(self.on_overlay_change)

        # # Assign multicanvas mouse observer
        # self.state.observe_multicanvas_mouse(self.on_multicanvas_event)


    def on_selection_change(self, key, change):
        """Handle changes to the dataset selection."""
        # Update the control panel information
        if key == "dataset":
            self.state.dataset_index = change.new
            self.state.dataset       = self.state.get_dataset_value()
            self.state.participant   = self.state.get_participant_value()
            self.state.hemisphere    = self.state.get_hemisphere_value() 
        elif key == "participant":
            self.state.participant   = change.new
        elif key == "hemisphere":
            self.state.hemisphere    = self.state.format_hemisphere(change.new)

        # Update the infobox displays
        for k in self.control_panel.infobox.keys():
            self.control_panel.refresh_infobox(self.state, k)

        # Load the new participant data
        self.state.midgray, self.state.properties = self.state.load_participant()

        # Update the figure panel's mesh values
        self.state.update_figure()

        # Update the figure
        self.figure_panel.refresh_figure(self.state)


    def on_inflation_slider(self, change):
        """Handle changes to the inflation slider value."""
        # Update the inflation value (internal state)
        self.state.inflation_value = change.new # percent, 0-100%

        # Update the figure panel's mesh values
        self.state.update_figure()

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


    # def on_multicanvas_event(self, change):
    #     """Handle multicanvas mouse events."""
    #     annt = self.state.annotations

    #     self.figure_panel.handle_multicanvas_event(self.state, change)