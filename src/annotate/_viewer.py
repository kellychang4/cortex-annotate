# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer.py

"""
Implementation code for the Cortex Viewer.
"""

# Imports ----------------------------------------------------------------------


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

    def __init__(
            self, 
            annotation_widgets, 
            dataset_directory = "/home/jovyan/datasets", 
            inflation_value   = 100
        ):
        
        # Store intialization variables
        self.annotation_widgets = annotation_widgets 
        self.dataset_directory  = dataset_directory

        # Prepare initial control panel
        self.dataset_index    = self.get_dataset_index()
        self.dataset          = self.get_dataset_value()
        self.participant      = self.get_participant_value()
        self.hemisphere       = self.get_hemisphere_value() 
        self.inflation_value  = inflation_value 

        # Prepare fsaverage data 
        self.fsaverage = ny.freesurfer_subject("fsaverage")
        self.tesselation, self.inflated = self._get_fsaverage()

        # Prepare participant data 
        self.midgray, self.curvature = self.load_participant()
        self.coordinates = self.update_coordinates()
        self.mesh        = self.update_mesh()
        self.color       = self.update_color()


    def get_dataset_index(self):
        """Get the current dataset selection index."""
        return self.annotation_widgets.selected_index
    

    def get_dataset_value(self):
        """Get the current dataset selection widget."""
        return self.annotation_widgets.titles[self.get_dataset_index()]
    
    
    def _get_active_selection(self):
        """Get the active selection panel widget."""
        active_widget = self.annotation_widgets.children[self.dataset_index]
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
    def _read_curvature(cls, filename):
        """Read cortical curvature data from a <hemisphere>.3d.curvature file."""
        with open(filename, "rb") as f:
            values = f.read() # load file content
            fstr = "e" * (len(values) // 2)
            curvature = np.array(unpack(fstr, values)).reshape((-1, ))
        return curvature


    def load_participant(self): 
        """Load participant dataset based on current state."""
        # Locate participant directory and files
        data_dir = op.join(self.dataset_directory, self.dataset.lower(), self.participant)
        coordinates_file = op.join(data_dir, f"{self.hemisphere}.3d.coordinates")
        curvature_file   = op.join(data_dir, f"{self.hemisphere}.3d.curvature")

        # Read participant cortical midgray coordinates and curvature data
        return ( self._read_coordinates(coordinates_file),
                 self._read_curvature(curvature_file) )
    

    def update_coordinates(self):
        """Update the cortical mesh coordinates based on the inflation value."""
        return ((self.inflated[self.hemisphere] - self.midgray) * \
                (self.inflation_value / 100.0)) + self.midgray


    def update_mesh(self):
        """Update the cortical mesh object based on the current state."""
        return ny.geometry.Mesh(
            faces       = self.tesselation[self.hemisphere],
            coordinates = self.coordinates,
            properties  = { "curvature" : self.curvature }
        )
    

    def update_color(self):
        """Update the cortical mesh color based on curvature."""
        return ny.graphics.cortex_plot_colors(self.mesh)[:, :3]


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
        self.state.midgray, self.state.curvature = self.state.load_participant()

        # Update the figure panel's mesh coordinates
        self.state.update_figure()

        # Update the figure
        self.figure_panel.refresh_figure(self.state)


    def on_inflation_slider(self, change):
        """Handle changes to the inflation slider value."""
        # Update the inflation value (internal state)
        self.state.inflation_value = change.new # percent, 0-100%

        # Update the figure panel's mesh coordinates
        self.state.update_figure()

        # Update the figure with new coordinates
        self.figure_panel.refresh_figure(self.state)
