# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_style.py

"""Annotation and viewer style tab for cortex-annotate.

Provides the ``StyleTab`` widget, which manages two groups of
controls:

    Annotation style : visibility, color, linewidth, linestyle,
                       and markersize, for each annotation.  A dropdown
                       selects which annotation's style is being edited;
                       index 0 (``"Active Annotation"``) maps to the
                       style applied to whichever annotation is
                       currently selected in the ``SelectionPanel``.

    Viewer style     : cortex viewer rendering controls (morph percentage,
                       overlay, overlay alpha, point size, line width,
                       line interpolation).  Created only when a 3D
                       viewer is active.
"""

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw
from functools import partial

from .._widgets import make_section_title, make_hline

# The Style tab -----------------------------------------------------------

class StyleTab(ipw.VBox):
    """Annotation and viewer style controls.

    Parameters
    ----------
    config : Config
        Parsed project configuration.  Used to enumerate annotation
        names and cortex overlay names.

    prefs : PrefsManager
        Preferences manager, used to read current style values for
        populating widgets.
    """

    __slots__ = (
        "config", 
        "prefs", 
        "has_viewer", 
        "_updating",
        # Annotation style widgets
        "style_dropdown",
        "visible_checkbox", 
        "color_picker", 
        "markersize_slider",
        "linewidth_slider", 
        "linestyle_dropdown",
        "_annotation_style_widgets", 
        "annotation_style_observers",
        # Viewer style widgets (depends on has_viewer)
        "morph_slider", 
        "overlay_dropdown",
        "overlay_slider",
        "point_size_slider", 
        "line_width_slider", 
        "line_interp_slider",
        "_viewer_style_widgets", 
        "viewer_style_observers",
    )

    # Shared widget layout.
    _WIDGET_LAYOUT = { "width": "94%", "margin": "0% 3% 0% 3%" }

    _SLIDER_KWARGS = {
        "readout"           : False,
        "continuous_update" : False,
        "orientation"       : "horizontal",
        "layout"            : _WIDGET_LAYOUT,
    }

    _ANNOT_SLIDER_KWARGS = {
        **_SLIDER_KWARGS,
        "value": 1, "min": 1, "max": 8, "step": 1,
        "readout": False,
    }

    def __init__(self, config, prefs, has_viewer):
        # Store arguments and determine if has_viewer from config information.
        self.prefs      = prefs
        self.config     = config
        self.has_viewer = has_viewer
        # self.has_morph  = at least get the flag 
        self._updating  = False

        # Initialize annotation style widgets.
        self.style_dropdown     = self._init_style_dropdown()
        self.visible_checkbox   = self._init_visible_checkbox()
        self.color_picker       = self._init_color_picker()
        self.markersize_slider  = self._init_markersize_slider()
        self.linewidth_slider   = self._init_linewidth_slider()
        self.linestyle_dropdown = self._init_linestyle_dropdown()

        self._annotation_style_widgets = {
            "visible":    self.visible_checkbox,
            "color":      self.color_picker,
            "markersize": self.markersize_slider,
            "linewidth":  self.linewidth_slider,
            "linestyle":  self.linestyle_dropdown,
        }

        # Initialize viewer style widgets (if has_viewer)
        if self.has_viewer:
            self.morph_slider       = self._init_morph_slider()
            self.overlay_dropdown   = self._init_overlay_dropdown()
            self.overlay_slider     = self._init_overlay_alpha_slider()
            self.point_size_slider  = self._init_point_size_slider()
            self.line_width_slider  = self._init_line_width_slider()
            self.line_interp_slider = self._init_line_interp_slider()

            self._viewer_style_widgets = {
                "morph_percent": self.morph_slider,
                "overlay":       self.overlay_dropdown,
                "overlay_alpha": self.overlay_slider,
                "point_size":    self.point_size_slider,
                "line_width":    self.line_width_slider,
                "line_interp":   self.line_interp_slider,
            }

        # Build the widget panel.
        children = [
            make_section_title("Style Options"),
            self.style_dropdown,
            make_hline(),
            make_section_title("Annotation Canvas Options"),
            self.visible_checkbox,
            self.color_picker,
            self.markersize_slider,
            self.linewidth_slider,
            self.linestyle_dropdown,
        ]

        # If has_viewer, add viewer style widgets below annotation style widgets.
        if self.has_viewer:
            children += [
                make_hline(),
                make_section_title("Cortex Viewer Options"),
                self.morph_slider,
                self.overlay_dropdown,
                self.overlay_slider,
                self.point_size_slider,
                self.line_width_slider,
                self.line_interp_slider,
            ]

        # Initialize the VBox with the assembled children and layout.
        super().__init__(children, layout = { "margin": "0% 0% 3% 0%" })

        # Wire style dropdown observer.
        self.style_dropdown.observe(
            self._on_style_dropdown_change, names = "index",
        )

        # Wire annotation style widget observers.
        for key, widget in self._annotation_style_widgets.items():
            widget.observe(
                partial(self._on_annotation_style_change, key),
                names = "value"
            )
        
        # If has_viewer, wire viewer style widget observers.
        if self.has_viewer:
            for key, widget in self._viewer_style_widgets.items():
                widget.observe(
                    partial(self._on_viewer_style_change, key),
                    names = "value"
                )

        # Initialize observer lists.
        self.annotation_style_observers = []
        self.viewer_style_observers     = [] 

        # Initialize annotation and viewer style widgets.
        self._refresh_annotation_style()
        if self.has_viewer: self._refresh_viewer_style()

    # Properties ---------------------------------------------------------------

    @property
    def annotation(self):
        """Return the annotation selected in the style dropdown.

        Returns
        -------
        str or None
            The annotation name, or ``None`` when the ``"Active
            Annotation"`` option (index 0) is selected.
        """
        dd = self.style_dropdown
        return dd.value if dd.index > 0 else None

    # Refresh Methods ----------------------------------------------------------

    def _refresh_annotation_style(self):
        """Refresh annotation style widgets with style preferences of current 
        annotation.

        Suppresses observer dispatch via the ``_updating`` guard so
        that widget value changes do not fire external callbacks.
        """
        self._updating = True
        try:
            # Get current preferences for the selected annotation for styling
            style = self.prefs.get_annotation_style(self.annotation)
            for key, widget in self._annotation_style_widgets.items():
                # Update widget value with style value
                widget.value = style[key] 
        finally:
            self._updating = False


    def _refresh_viewer_style(self):
        """Refresh viewer style widgets with style preferences.

        Only called when ``has_viewer`` is ``True``.  Suppresses
        observer dispatch via the ``_updating`` guard.
        """
        self._updating = True
        try:
            # Get current preferences for the viewer style
            style = self.prefs.get_viewer_style()
            for key, widget in self._viewer_style_widgets.items():
                # Update widget value with style value
                widget.value = style[key]
        finally:
            self._updating = False

    # Internal Handlers --------------------------------------------------------

    def _on_style_dropdown_change(self, _):
        """Refresh annotation style widgets when the dropdown changes."""
        self._refresh_annotation_style()


    def _on_annotation_style_change(self, key, change):
        """Notify annotation style observers.

        Suppressed while ``_updating`` is ``True``.

        Parameters
        ----------
        key : str
            The annotation style key that changed.

        change : traitlets.Bunch
            The ipywidgets change object.
        """
        if self._updating: return
        for fn in self.annotation_style_observers:
            fn(self.annotation, key, change)


    def _on_viewer_style_change(self, key, change):
        """Notify viewer style observers.

        Suppressed while ``_updating`` is ``True``.

        Parameters
        ----------
        key : str
            The viewer style key that changed.
        change : traitlets.Bunch
            The ipywidgets change object.
        """
        if self._updating: return
        for fn in self.viewer_style_observers:
            fn(key, change)


    # Observer Registration ----------------------------------------------------

    def observe_annotation_style(self, fn):
        """Register a callback for annotation style changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(annotation, key, change)`` where
            *annotation* is the name of the annotation being styled
            (``None`` for the active annotation), *key* is the
            style key that changed (``"visible"``, ``"color"``,
            ``"linewidth"``, ``"linestyle"``, or ``"markersize"``),
            and *change* is the ipywidgets change object.
        """
        self.annotation_style_observers.append(fn)


    def observe_viewer_style(self, fn):
        """Register a callback for viewer style changes.

        Will only work when ``has_viewer`` is ``True``. 

        Parameters
        ----------
        fn : callable
            Called as ``fn(key, change)`` where *key* is the
            viewer style key that changed and *change* is the
            ipywidgets change object.
        """
        if self.has_viewer: 
            self.viewer_style_observers.append(fn)

    # Lock / Unlock ------------------------------------------------------------

    def lock(self):
        """Disable all style controls."""
        # Style dropdown dropdown menu is locked
        self.style_dropdown.disabled = True
        for widget in self._annotation_style_widgets.values():
            widget.disabled = True

        # If has_viewer, viewer style widgets are also locked
        if self.has_viewer:
            for widget in self._viewer_style_widgets.values():
                widget.disabled = True


    def unlock(self):
        """Enable all style controls."""
        # Style dropdown dropdown menu is unlocked
        self.style_dropdown.disabled = False
        for widget in self._annotation_style_widgets.values():
            widget.disabled = False
        
        # If has_viewer, viewer style widgets are also unlocked
        if self.has_viewer:
            for widget in self._viewer_style_widgets.values():
                widget.disabled = False

    # Annotation Style Widget Initialisers -------------------------------------

    def _init_style_dropdown(self):
        """Create the annotation-selector dropdown."""
        options = [ "Active Annotation" ] + list(self.config.annotations.keys())
        return ipw.Dropdown(
            options     = options,
            value       = options[0],
            description = "Annotation:",
            layout      = { **self._WIDGET_LAYOUT, "margin": "3% 3% 3% 3%" },
        )


    def _init_visible_checkbox(self):
        """Create the annotation visibility checkbox."""
        return ipw.Checkbox(
            description = "Visible",
            value       = True,
            layout      = self._WIDGET_LAYOUT,
        )


    def _init_color_picker(self):
        """Create the annotation color picker."""
        return ipw.ColorPicker(
            concise     = False,
            description = "Color:",
            value       = "blue",
            layout      = self._WIDGET_LAYOUT,
        )


    def _init_markersize_slider(self):
        """Create the annotation markersize slider."""
        return ipw.IntSlider(
            **{ **self._ANNOT_SLIDER_KWARGS, "max": 12 },
            description = "Point Size:",
        )


    def _init_linewidth_slider(self):
        """Create the annotation linewidth slider."""
        return ipw.IntSlider(
            **self._ANNOT_SLIDER_KWARGS,
            description = "Line Width:",
        )


    def _init_linestyle_dropdown(self):
        """Create the annotation linestyle dropdown."""
        return ipw.Dropdown(
            options     = [ "solid", "dashed", "dot-dashed", "dotted" ],
            description = "Line Style:",
            layout      = self._WIDGET_LAYOUT,
        )

    # Viewer Style Widget Initialisers -----------------------------------------

    def _init_morph_slider(self):
        """Create the cortex morph percentage slider."""
        return ipw.IntSlider(
            **self._SLIDER_KWARGS,
            value       = 0,
            min         = 0,
            max         = 100,
            step        = 1,
            description = "Morph %:",
        )


    def _init_overlay_dropdown(self):
        """Create the cortex overlay selection dropdown.

        Overlay names are read from ``config.viewer["overlays"]``.
        ``"curvature"`` is treated as the default (labeled ``"None"``
        in the dropdown) and is excluded from the sorted option list.
        """
        # Get the viewer overlay names, exclude curvature (it is default)
        overlay_names = list(self.config.viewer["overlays"].keys())
        if "curvature" in overlay_names: overlay_names.remove("curvature")
        overlay_names = sorted(overlay_names)

        # Build dropdown options, with human readable labels.
        dd_options = [("None", "curvature")]
        for name in overlay_names: # for each overlay
            # Edit variable name to be more human readable (e.g. "sulc_depth" -> "Sulc Depth")
            dd_options.append((name.replace("_", " ").title(), name))

        # Return the dropdown widget
        return ipw.Dropdown(
            options     = dd_options,
            value       = "curvature",
            description = "Overlay:",
            layout      = { **self._WIDGET_LAYOUT, "margin": "0% 3% 3% 3%" },
        )


    def _init_overlay_alpha_slider(self):
        """Create the cortex overlay alpha slider."""
        return ipw.FloatSlider(
            **self._SLIDER_KWARGS,
            value       = 1.0,
            min         = 0.0,
            max         = 1.0,
            step        = 0.01,
            description = "Alpha:",
        )


    def _init_point_size_slider(self):
        """Create the cortex point size slider."""
        return ipw.FloatSlider(
            **self._SLIDER_KWARGS,
            value       = 0.5,
            min         = 0.5,
            max         = 5.0,
            step        = 0.1,
            description = "Point Size:",
        )


    def _init_line_width_slider(self):
        """Create the cortex linewidth slider."""
        return ipw.FloatSlider(
            **self._SLIDER_KWARGS,
            value       = 0.2,
            min         = 0.10,
            max         = 0.50,
            step        = 0.05,
            description = "Line Width:",
        )


    def _init_line_interp_slider(self):
        """Create the cortex line interpolation slider."""
        return ipw.IntSlider(
            **self._SLIDER_KWARGS,
            value       = 10,
            min         = 5,
            max         = 20,
            step        = 1,
            description = "Line Interp.:",
        )