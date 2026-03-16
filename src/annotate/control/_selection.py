# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_selection.py
#
# DOCSTRING
    
# Imports ----------------------------------------------------------------------

import os.path as op
import ipywidgets as ipw
from functools import partial

from .._widgets import make_section_title, make_hline, darken_color

# The Selection Subpanel -------------------------------------------------------

class SelectionPanel(ipw.VBox):
    """The subpanel of the control panel for target selection."""
    
    __slots__ = (
        "state", "target_dropdowns", "annotations_dropdown", 
        "target_observers", "annotation_observers"
    )
    
    def __init__(self, state):
        # Store the state.
        self.state = state

        # We have to manage an "updating" state to avoid firing observers while
        # in the middle of updating dependent dropdowns.
        self._updating = False

        # Initialize the dropdowns.
        self.target_dropdowns = {}
        
        # Create the dropdown widgets as children (excluding annotation).
        children = [ 
            make_section_title("Selection"),
        ]
        dd_layout = { "width": "94%", "margin": "1% 3% 1% 3%" }
        for key in state.config.targets.concrete_keys:
            dropdown_values = state.config.targets.items[key]

            # If dynamic target (dict), dropdown values depend on the parent 
            # key's dropdown selection.
            if isinstance(dropdown_values, dict):        
                # Look up the parent key that this target depends on.
                parent_key = dropdown_values["depends_on"]

                # Look up the first parent selection and return options.
                parent0 = state.config.targets.items[parent_key][0]
                
                # Look up the dropdown values for this parent selection.
                dropdown_values = dropdown_values[parent0]

            dropdown_widget = ipw.Dropdown(
                options     = dropdown_values, 
                value       = dropdown_values[0], 
                layout      = dd_layout, 
                description = (key + ":")
            )
            children.append(dropdown_widget)
            self.target_dropdowns[key] = dropdown_widget
            
        # We also need the annotation dropdown.
        self.annotations_dropdown = ipw.Dropdown(
            options     = [], 
            layout      = dd_layout, 
            description = "Annotation:"
        )
        children.append(self.annotations_dropdown)

        super().__init__(children)
        
        # Because we want to control the order of a few things, we actually
        # listen to our selection items ourselves, then update them and pass
        # them along to our listeners. This is important so that, for example,
        # the Figure panel's listener does not get updated before the annotation
        # selection dropbox is changed when the user changes the target
        # selection.

        # FIRST: Wire up dependent dropdown updates.
        # These must fire before on_target_change so that downstream
        # dropdowns have valid options when the target is read.
        for key in state.config.targets.concrete_keys:
            target_item = state.config.targets.items[key]
            if isinstance(target_item, dict):
                parent_key = target_item["depends_on"]
                self.target_dropdowns[parent_key].observe(
                    partial(self.on_parent_change, key), names = "value"
                )

        # SECOND: Wire up the (non-dependent) target change observers.
        # By the time these fire, dependent dropdowns are already updated.
        for key in state.config.targets.concrete_keys:
            self.target_dropdowns[key].observe(
                partial(self.on_target_change, key), names = "value"
        )   

        # THIRD: Wire up the annotation change observer.
        # By the time this fires, the target observers have already fired.
        self.annotations_dropdown.observe(
            self.on_annotation_change, names = "value")
        
        # Initialize the observer lists.
        self.target_observers     = []
        self.annotation_observers = []
        
        # Initialize the annotations menu.
        self.refresh_annotations()

    # Property Methods ---------------------------------------------------------

    @property
    def target(self):
        """Compute the current target selection."""
        return tuple( dd.value for dd in self.target_dropdowns.values() )
    

    @property
    def annotation(self):
        """Compute the current annotation selection."""
        return self.annotations_dropdown.value
    

    @property
    def selection(self):
        """Compute the current selection (target + annotation)."""
        return self.target + (self.annotation, )

    # Refresh Methods ----------------------------------------------------------

    def refresh_annotations(self):
        """Refreshes the annotations menus based on the current target selection."""
        # Get the new target selection entirely.
        target_id = self.target
    
        # Look up the target for this selection.
        target = self.state.config.targets[target_id]
    
        # Recalculate the annotations for this target and update the menu.
        annotation_options = [ 
            annotation for ( annotation, annotation_data ) 
            in self.state.config.annotations.items()
            if annotation_data.filter is None or annotation_data.filter(target) 
        ]
        self.annotations_dropdown.options = annotation_options
        self.annotations_dropdown.value   = annotation_options[0]

    # Handler Methods ----------------------------------------------------------

    def on_parent_change(self, key, change):
        """Handles the change in a parent dropdown for a dynamic target."""
        # Set "updating" state to avoid firing dependent dropdown observers.
        self._updating = True
        try: 
            # Get the new parent selection and dropdown values.
            dependent_items = self.state.config.targets.items[key]
            dropdown_values = dependent_items[change.new]

            # Update the dependent dropdown's options and value.
            dependent_dropdown = self.target_dropdowns[key]
            dependent_dropdown.options = dropdown_values
            dependent_dropdown.value   = dropdown_values[0]
        finally: 
            # Undo "updating" state.
            self._updating = False


    def on_target_change(self, key, change):
        """Alert our observers that the target selection has changed."""
        # Prevent firing observers if we are updating dependent dropdowns.
        if self._updating: return

        # Set "updating" state to avoid annotation dropdown observers.
        self._updating = True
        try: 
            # Refresh the annotations menu.
            self.refresh_annotations()
        finally:
            # Undo "updating" state.
            self._updating = False

        # Alert our target observers, now that our updates are finished.
        for fn in self.target_observers:
            fn(key, change)


    def on_annotation_change(self, change):
        """Alert our observers that the annotation selection has changed."""
        # Alert our observers.
        for fn in self.annotation_observers:
            fn(change)


    # Observer Methods ---------------------------------------------------------

    def observe_target(self, fn):
        """Registers the given function to be called when the target changes.

        The selection target refers to the selection of all the concrete keys in
        the `config.yaml` file's `targets` section. In other words, the
        selection target changes when any of the selection dropdowns are changed
        except for the annotation dropdown.

        When the selection target changes, the given function is called with two
        arguments: `fn(concrete_key, change)` where `concrete_key` is the
        (string) name of one of the concrete keys and `change` is the change
        object typically used in the `ipywidget` `observe` pattern.
        """
        self.target_observers.append(fn)


    def observe_annotation(self, fn):
        """Registers the argument to be called when the annotation changes.

        The annotation selection is the currently selected annotation in the
        annotations dropdown menu of the `SelectionPanel` component of the
        `ControlPanel`.

        When the annotation selection changes, the given function is called with
        the argument `change` where `change` is the `change` object typically
        used in the `ipywidget` `observe` pattern.
        """
        self.annotation_observers.append(fn)


    def observe_selection(self, fn):
        """Registers the given function to be called when the selection changes.

        The selection refers to the combination of target and annotation
        selection; see the `observe_target` and `observe_annotation` methods for
        more information.

        When the selection changes, the given function is called with two
        arguments: `fn(concrete_key, change)` where `concrete_key` is the
        (string) concrete key that has changed and `change` is the change object
        typically used in the `ipywidget` `observe` pattern. If the annotation
        has changed, then the `key` will be `None`.
        """
        self.observe_target(fn)
        self.observe_annotation(partial(fn, None))
