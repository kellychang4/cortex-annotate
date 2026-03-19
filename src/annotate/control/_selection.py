# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_selection.py

"""Target and annotation selection subpanel.
 
Provides the ``SelectionPanel`` widget, which manages a set of dropdown
menus for choosing annotation targets (derived from the ``config.yaml``
``targets`` section) and the active annotation.  Dependent dropdowns
(whose options change based on a parent dropdown's value) are updated
automatically.
 
Internal widget observers fire in three tiers:
 
1. **Parent-dependent updates** — dependent dropdowns receive valid options.
2. **Target observers** — registered callbacks fire with consistent dropdown
   state.
3. **Annotation observers** — fire only after all target updates are
complete.
 
An ``_updating`` guard prevents observers from firing during internal changes.
"""
 
# Imports ----------------------------------------------------------------------

import ipywidgets as ipw
from functools import partial
 
from .._widgets import make_section_title

# The Selection Subpanel -------------------------------------------------------

class SelectionPanel(ipw.VBox):
    """Target and annotation selection panel.
 
    Parameters
    ----------
    config : Config
        The configuration object. Used to enumerate targets and
        annotations and to evaluate annotation filters.

    Attributes
    ----------
    target_dropdowns : dict[str, ipywidgets.Dropdown]
        Mapping of concrete target key names to their dropdown widgets.

    annotations_dropdown : ipywidgets.Dropdown
        Dropdown for choosing the active annotation.

    target_observers : list[callable]
        Callbacks registered via ``observe_target``.

    annotation_observers : list[callable]
        Callbacks registered via ``observe_annotation``.
    """

    __slots__ = (
        "config",
        "target_dropdowns",
        "annotations_dropdown",
        "target_observers",
        "annotation_observers",
        "_updating",
    )
    
    def __init__(self, config):
        # Store the config.
        self.config = config    
 
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
        for key in config.targets.concrete_keys:
            # Look up the dropdown values for the current concrete key.
            dropdown_values = config.targets.item_generators[key]
 
            # If dynamic target (dict), dropdown values depend on the parent
            # key's dropdown selection.
            if isinstance(dropdown_values, dict):
                # Look up the parent key that this target depends on.
                parent_key = dropdown_values["depends_on"]
 
                # Look up the first parent selection and return options.
                parent0 = config.targets.item_generators[parent_key][0]
 
                # Look up the dropdown values for this parent selection.
                dropdown_values = dropdown_values[parent0]

            # Create the dropdown widget, append to children list and target_dropdowns.
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

        # Initialize the VBox with the dropdown children.
        super().__init__(children)
 
        # Because we want to control the order of a few things, we actually
        # listen to our selection items ourselves, then update them and pass
        # them along to our listeners. This is important so that, for example,
        # the Figure panel's listener does not get updated before the annotation
        # selection dropdown is changed when the user changes the target
        # selection.
 
        # FIRST: Wire up parent of dependent dropdown updates.
        # These must fire before on_target_change so that downstream
        # dropdowns have valid options when the target is read.
        for key in config.targets.concrete_keys:
            target_itemgen = config.targets.item_generators[key]
            if isinstance(target_itemgen, dict):
                parent_key = target_itemgen["depends_on"]
                self.target_dropdowns[parent_key].observe(
                    partial(self._on_parent_change, key), names = "value"
                )
 
        # SECOND: Wire up the (non-dependent) target change observers.
        # By the time these fire, dependent dropdowns are already updated.
        for key, widget in self.target_dropdowns.items():
            widget.observe(
                partial(self._on_target_change, key), names = "value")
 
        # THIRD: Wire up the annotation change observer.
        # By the time this fires, the target observers have already fired.
        self.annotations_dropdown.observe(
            self._on_annotation_change, names = "value"
        )
 
        # Initialize the observer lists.
        self.target_observers     = []
        self.annotation_observers = []
        
        # Initialize the annotations menu.
        self.refresh_annotations() 

    # Properties ---------------------------------------------------------------
 
    @property
    def target(self):
        """Return the current target selection as a tuple of dropdown values.
 
        Returns
        -------
        tuple[str, ...]
            One value per concrete target key, in ``concrete_keys`` order.
        """
        return tuple(dd.value for dd in self.target_dropdowns.values())
 

    @property
    def annotation(self):
        """Return the currently selected annotation name.
 
        Returns
        -------
        str or None
            The value of the annotations dropdown widget.
        """
        return self.annotations_dropdown.value
 

    @property
    def selection(self):
        """Return the full selection as ``target + (annotation,)``.
 
        Returns
        -------
        tuple[str, ...]
            The target tuple with the annotation name appended.
        """
        return self.target + (self.annotation,)

    # Refresh ------------------------------------------------------------------
 
    def refresh_annotations(self):
        """Rebuild the annotations dropdown for the current target.
 
        Filters the full annotation list from ``config.annotations``
        using each annotation's ``filter`` predicate (if any) against
        the resolved target object.  The dropdown value is reset to
        the first matching annotation.
        """ 
        # Look up the target for this selection.
        target = self.config.targets[self.target]
 
        # Recalculate the annotations for this target and update the menu.
        annotation_options = [
            name for ( name, annotation )
            in self.config.annotations.items()
            if annotation.filter is None or annotation.filter(target)
        ]
        self.annotations_dropdown.options = annotation_options
        self.annotations_dropdown.value   = annotation_options[0]
        
    # Internal Handlers --------------------------------------------------------
 
    def _on_parent_change(self, key, change):
        """Update a dependent dropdown when its parent value changes.
 
        Parameters
        ----------
        key : str
            The concrete key of the *dependent* dropdown (not the parent).

        change : traitlets.Bunch
            The ipywidgets change object from the parent dropdown.
        """
        # Set "updating" state to avoid firing dependent dropdown observers.
        self._updating = True
        try:
            # Get the new parent selection and dropdown values.
            dependent_items = self.config.targets.item_generators[key]
            dropdown_values = dependent_items[change.new]
 
            # Update the dependent dropdown's options and value.
            dependent_dropdown = self.target_dropdowns[key]
            dependent_dropdown.options = dropdown_values
            dependent_dropdown.value   = dropdown_values[0]
        finally:
            # Undo "updating" state.
            self._updating = False
 

    def _on_target_change(self, key, change):
        """Refresh annotations, then notify target observers.
 
        Parameters
        ----------
        key : str
            The concrete key whose dropdown changed.

        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Prevent firing observers if we are internally updating dropdowns.
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
 

    def _on_annotation_change(self, change):
        """Notify annotation observers.
 
        Suppressed while ``_updating`` is True (e.g., during a
        target-driven annotation refresh).
 
        Parameters
        ----------
        change : traitlets.Bunch
            The ipywidgets change object.
        """
        # Prevent firing observers if we are internally updating dropdowns.
        if self._updating: return

        # Alert our annotation observers, now that our updates are finished.
        for fn in self.annotation_observers:
            fn(change)
 
    # Observer Registration ----------------------------------------------------
 
    def observe_target(self, fn):
        """Register a callback for target dropdown changes.
 
        Parameters
        ----------
        fn : callable
            Called as ``fn(concrete_key, change)`` where *concrete_key*
            is the ``str`` name of the dropdown that changed and *change*
            is the ipywidgets change object.
        """
        self.target_observers.append(fn)
 

    def observe_annotation(self, fn):
        """Register a callback for annotation dropdown changes.
 
        Parameters
        ----------
        fn : callable
            Called as ``fn(change)`` where *change* is the ipywidgets
            change object.
        """
        self.annotation_observers.append(fn)