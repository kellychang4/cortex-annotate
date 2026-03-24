# -*- coding: utf-8 -*-
################################################################################
# annotate/control/_control.py
 
"""ControlPanel facade for the cortex-annotate annotation tool.
 
``ControlPanel`` composes the subpanels (``SelectionPanel``,
``DisplayPanel``, ``LegendPanel``, ``ButtonPanel``, ``InfoPanel``,
``StylePanel``) into a tabbed interface.
 
Tab Registration
----------------
``ControlPanel`` maintains an internal tab registry.  The two default
tabs (Selection, Style) are registered during construction.  Additional
tabs can be added after construction via ``register_tab(panel, title)``.
 
Any widget can serve as a tab panel.  If the panel exposes ``lock()``
and ``unlock()`` methods, ``ControlPanel`` will call them when the
tool-wide lock state changes.  Panels without those methods are simply
skipped during lock/unlock cycles.
 
Facade Contract
---------------
Properties
    ``target``, ``annotation``
 
Actions
    ``lock()``, ``unlock()``, ``update_legend(target_id, annotation)``
    
Observers
    ``observe_target``, ``observe_annotation``,
    ``observe_annotation_style``, ``observe_viewer_style``,
    ``observe_figure_size``, ``observe_layout``,
    ``observe_save``, ``observe_clear_current``, ``observe_clear_all``
"""

# Imports ----------------------------------------------------------------------

import ipywidgets as ipw

from ._annotate import AnnotateTab
from ._style    import StyleTab
from .._widgets import darken_color

# The Control Panel Widget -----------------------------------------------------

class ControlPanel(ipw.VBox):
    """Facade widget and tab host for all control subtab/subpanels.
 
    Parameters
    ----------
    config : Config
        Parsed project configuration, passed through to
        ``SelectionPanel``, ``LegendPanel``, and ``StylePanel``.
 
    prefs : PrefsManager
        Preferences manager, passed through to ``DisplayPanel`` and
        ``StylePanel``.
 
    background_color : str, optional
        CSS color for the active tab and content area.
        
    button_color : str, optional
        CSS color for the Save button background.
 
    Attributes
    ----------
    _annotate_tab : AnnotateTab

    _style_tab : StyleTab
 
    _tabs : list[tuple[str, Widget, bool]]
        Internal tab registry.  Each entry is
        ``(title, panel, lockable)``.
 
    _tab_widget : ipywidgets.Tab
        The rendered tab bar.
 
    _accordion : ipywidgets.Accordion
        Collapsible wrapper around the tab widget.
    """
 
    __slots__ = (
        "_annotate_tab",
        "_style_tab",
        "_tabs",
        "_tab_widget",
        "_accordion",
    )
 
    def __init__(
            self,
            config,
            prefs,
            background_color = "#f0f0f0",
            button_color     = "#e0e0e0",
        ):
        # Create the required subtabs/subpanels widgets.
        self._annotate_tab = AnnotateTab(config, prefs, button_color)
        self._style_tab    = StyleTab(config, prefs)
 
        # Declare the tab registry and register the default tabs.
        self._tabs = []

        # Populate the tab register with the defaults. 
        # The loop allows for easy extension to more default tabs in the future, if desired.
        for tab in [ self._annotate_tab, self._style_tab ]:
            # TODO: I dop not like how the tab name is derived from the class. 
            tab_name = type(tab).__name__.removesuffix("Tab").title()
            self._add_tab(tab_name, tab)
 
        # Build the tab widget from the registry.
        self._tab_widget = self._build_tab_widget()
 
        # Declare the accordion wrapper and add the tab widget as its child. 
        # This creates a collapsible wrapper around the tabs, so users can hide
        # the entire control panel when desired.
        self._accordion = ipw.Accordion(
            children       = [ self._tab_widget ],
            selected_index = 0,
        )
 
        # Finally, initialize the VBox with the header and accordion.
        super().__init__(
            children = [
                self._make_html_header(
                    background_color = background_color, 
                    button_color     = button_color
                ),
                self._accordion,
            ],
            layout = { "border": "0px" },
        )
 

    # Tab Registry -------------------------------------------------------------

    def _add_tab(self, title, panel):
        """Append a tab to the internal registry.
 
        Parameters
        ----------
        title : str
            Tab title shown in the tab bar.
 
        panel : ipywidgets.Widget
            The widget displayed as the tab body.
        """
        # Determine if the panel lock/unlock by checking for attributes.
        lockable = hasattr(panel, "lock") and hasattr(panel, "unlock")
        
        # Append the new tab to the registry.
        self._tabs.append((title, panel, lockable))
 
 
    def _build_tab_widget(self):
        """Create a fresh ``ipw.Tab`` from the current registry."""
        # Unzip the registry into separate lists of panels and titles.
        children = [ panel  for (_, panel, _) in self._tabs ]
        titles   = [ title  for (title, _, _) in self._tabs ]
 
        # Create the tab widget and add a class for styling.
        tab = ipw.Tab(
            children       = children,
            titles         = titles,
            selected_index = 0,
        )
        tab.add_class("annotate-control-tabs")
 
        # Return the tab widget.
        return tab

    # HTML Helper --------------------------------------------------------------

    def _make_html_header(
            self, 
            background_color = "#f0f0f0", 
            button_color     = "#e0e0e0",
            darkened_amount  = 0.10, 
        ):
        """Return an ``ipw.HTML`` widget containing scoped CSS for the panel.
 
        The CSS targets JupyterLab class names to style the accordion,
        tab bar, and horizontal-rule dividers.
 
        Parameters
        ----------
        background_color : str
            CSS color for the active tab and content area.
        
        button_color : str
            CSS color for the layout toggle button and its active
            darkened variant.
 
        darkened_amount : float
            Fraction by which to darken *background_color* and *button_color* 
            for alternative color.
 
        Returns
        -------
        ipywidgets.HTML
        """
        inactive_rgb  = darken_color(background_color, darkened_amount)
        toggle_active = darken_color(button_color, darkened_amount)
        return ipw.HTML(f"""
            <style>
                .jupyter-widget-Collapse-open {{
                    background-color: white;
                    width: 300px;
                }}
                .jupyter-widget-Collapse-header {{
                    background-color: white;
                    border: 0px;
                    padding: 0px;
                }}
                .jupyter-widget-Collapse-contents {{
                    background-color: white;
                    border: 0px;
                    padding: 0px;
                }}
                .annotate-control-tabs.jupyter-widget-tab.widget-tab {{
                    max-width: 300px;
                    min-height: 850px;
                }}
                .annotate-control-tabs
                    > .jupyter-widget-TabPanel-tabContents.widget-tab-contents
                {{
                    background-color: {background_color};
                    margin: 0px;
                    padding: 5px;
                }}
                .annotate-control-tabs.jupyter-widgets.jupyter-widget-tab
                    > .lm-TabBar .lm-TabBar-tab
                {{
                    background-color: rgb{inactive_rgb};
                    flex: 1 1 auto;
                }}
                .annotate-control-tabs.jupyter-widgets.jupyter-widget-tab
                    > .lm-TabBar .lm-TabBar-tab.lm-mod-current
                {{
                    background-color: {background_color};
                }}
                .annotate-control-panel-hline {{
                    border: 1px lightgray solid;
                    height: 0px;
                    width: 94%;
                    margin: 3%;
                }}
                .annotate-layout-toggle.jupyter-widgets {{
                    background-color: {button_color};
                    border: 1px solid rgb{toggle_active};
                }}
                .annotate-layout-toggle.jupyter-widgets.mod-active {{
                    background-color: rgb{toggle_active};
                }}
            </style>
        """)
 
    # Properties ---------------------------------------------------------------

    @property
    def target(self):
        """The current target selection as a tuple of dropdown values."""
        return self._annotate_tab.target


    @property
    def annotation(self):
        """The currently selected annotation name."""
        return self._annotate_tab.annotation

    # Lock / Unlock ------------------------------------------------------------

    def lock(self):
        """Disable all interactive widgets in every registered tab.

        Iterates the tab registry and calls ``lock()`` on each panel
        that supports it.
        """
        for _, panel, lockable in self._tabs:
            if lockable: panel.lock()


    def unlock(self):
        """Unlock all interactive widgets in every registered tab.

        Iterates the tab registry and calls ``unlock()`` on each panel
        that supports it.
        """
        for _, panel, lockable in self._tabs:
            if lockable: panel.unlock()

    # Annotation Tab: Selection Panel Observers --------------------------------
    
    def observe_selection(self, fn):
        """Register a callback for target and annotation selection changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(concrete_key, change)``.
        """
        self._annotate_tab._selection.observe_selection(fn)

    # Annotation Tab: Display Panel Observers ----------------------------------

    def observe_canvas_size(self, fn):
        """Register a callback for canvas size changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(change)``.
        """
        self._annotate_tab._display.observe_canvas_size(fn)


    def observe_viewer_size(self, fn):
        """Register a callback for viewer size changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(change)``.
        """
        self._annotate_tab._display.observe_viewer_size(fn)


    def observe_layout(self, fn):
        """Register a callback for layout-toggle changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(change)``.
        """
        self._annotate_tab._display.observe_layout(fn)

    # Annotation Tab: Legend Panel Update --------------------------------------

    def update_legend(self, target_id, annotation):
        """Update the legend image for the given target and annotation.

        Parameters
        ----------
        target_id : tuple[str, ...]
            The full target-ID tuple.

        annotation : str
            The annotation name.
        """
        self._annotate_tab._legend.update(target_id, annotation)

    # Annotation Tab: Button Panel Observers -----------------------------------

    def observe_save(self, fn):
        """Register a callback for the Save button.
        
        Parameters
        ----------
        fn : callable
            Called as ``fn()`` when the Save button is clicked.
        """
        self._annotate_tab._button.save_button.on_click(fn)


    def observe_clear_current(self, fn):
        """Register a callback for the Clear Current button.

        Parameters
        ----------
        fn : callable
            Called as ``fn()`` when the Clear Current button is clicked.
        """
        self._annotate_tab._button.clear_current_button.on_click(fn)


    def observe_clear_all(self, fn):
        """Register a callback for the Clear All button.

        Parameters
        ----------
        fn : callable
            Called as ``fn()`` when the Clear All button is clicked.
        """
        self._annotate_tab._button.clear_all_button.on_click(fn)

    # Style Tab: Style Panel Observers -----------------------------------------

    def observe_annotation_style(self, fn):
        """Register a callback for annotation style changes.

        Parameters
        ----------
        fn : callable
            Called as ``fn(annotation, key, change)`` where
            *annotation* is the name of the affected annotation (or
            ``None`` for the active contour), *key* is one of
            ``"visible"``, ``"color"``, ``"linewidth"``,
            ``"linestyle"``, ``"markersize"``, and *change* is the
            ipywidgets change object.
        """
        self._style_tab.observe_annotation_style(fn)
 
 
    def observe_viewer_style(self, fn):
        """Register a callback for viewer style changes.
 
        No-op when the panel was created with ``has_viewer = False``.
 
        Parameters
        ----------
        fn : callable
            Called as ``fn(key, change)`` where *key* is the
            viewer style key that changed.
        """
        self._style_tab.observe_viewer_style(fn)
 