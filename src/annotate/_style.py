# -*- coding: utf-8 -*-
################################################################################
# annotate/_style.py

"""Style definitions and validation for annotation and viewer styles.
 
This module is the single source of truth for all style constants and
validation in the cortex-annotate toolkit. It defines two style categories:
 
    annotation_style : visual properties for 2D canvas annotations
                       (color, linewidth, linestyle, markersize, visible)
                       
    viewer_style     : 3D cortex viewer display settings
                       (morph_percent, overlay, overlay_alpha, point_size,
                       line_width, line_interp)
 
Each category has a DEFAULT_*_STYLE dict, a *_STYLE_KEYS tuple, and a
validate_*_style() function.
"""

# Imports ----------------------------------------------------------------------

import matplotlib as mpl

# Annotation Style Utilities ---------------------------------------------------

# Default annotation style values. 
DEFAULT_ANNOTATION_STYLE = {
    "visible"    : True,
    "color"      : "black",
    "linestyle"  : "solid",
    "linewidth"  : 1,
    "markersize" : 1,
}


# Annotation style key names.
ANNOTATION_STYLE_KEYS = tuple(DEFAULT_ANNOTATION_STYLE.keys())
 

def validate_annotation_style(style_dict):
    """Validate an annotation style dictionary.
 
    Checks that all keys are recognized annotation style keys and that
    values are within valid ranges. Mutates the input dict in place by
    converting color values to hex strings.
 
    Parameters
    ----------
    style_dict : dict
        A partial or full annotation style dict. Only the keys present
        are validated; missing keys are not required.
 
    Returns
    -------
    dict
        The same *style_dict* object, with colors normalized to hex.
    """
    # Check that all the keys are valid style keys.
    for key in style_dict.keys():
        if key not in ANNOTATION_STYLE_KEYS:
            raise RuntimeError(f"Invalid annotation style key: {key}")
        
    # Check that the linewidth is a valid number.
    if "linewidth" in style_dict:
        linewidth = style_dict["linewidth"]
        if linewidth < 0 or linewidth > 20:
            raise RuntimeError(f"Invalid linewidth: {linewidth}")
    
    # Check that the linestyle is valid.
    if "linestyle" in style_dict:
        linestyle = style_dict["linestyle"]
        if linestyle not in ("solid", "dashed", "dot-dashed", "dotted"):
            raise RuntimeError(f"Invalid linestyle: {linestyle}")
        
    # Check that the color is valid.
    if "color" in style_dict:
        color = style_dict["color"]
        try: color = mpl.colors.to_hex(color)
        except Exception as e: 
            raise RuntimeError(f"Invalid color: {color}") from e
        style_dict["color"] = color # store as hex, if valid

    # Check that the markersize is a valid number.
    if "markersize" in style_dict:
        markersize = style_dict["markersize"]
        if markersize < 0 or markersize > 20:
            raise RuntimeError(f"Invalid markersize: {markersize}")
    
    # Check that the visible is a boolean.
    if "visible" in style_dict:
        visible = style_dict["visible"]
        if not isinstance(visible, bool):
            raise RuntimeError(f"Invalid visible: {visible}")
    
    # Return the style dictionary, if valid.
    return style_dict

# Viewer Style Utilities -------------------------------------------------------
 
# Default viewer style values.
DEFAULT_VIEWER_STYLE = {
    "morph_percent" : 0,
    "overlay"       : "curvature",
    "overlay_alpha" : 1.0, 
    "point_size"    : 1.5, 
    "line_width"    : 0.25,
    "line_interp"   : 10,
}
 
 
# Viewer style key names.
VIEWER_STYLE_KEYS = tuple(DEFAULT_VIEWER_STYLE.keys())

# Button Style Variables -------------------------------------------------------

SAVE_TIMER = 2.0 # seconds
