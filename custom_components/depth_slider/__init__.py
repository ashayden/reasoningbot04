"""Custom depth slider component for Streamlit."""

import os
import streamlit.components.v1 as components
import streamlit as st

# Check if we're in development or production mode
_RELEASE = True

if not _RELEASE:
    _depth_slider = components.declare_component(
        "depth_slider",
        url="http://localhost:3001",
    )
else:
    # Get the build directory relative to this file
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    
    # Fallback to using st.select_slider if build directory doesn't exist
    if not os.path.exists(build_dir):
        def _depth_slider(value, options, key=None):
            return st.select_slider(
                "Analysis Depth",
                options=options,
                value=value,
                key=key
            )
    else:
        _depth_slider = components.declare_component("depth_slider", path=build_dir)

def depth_slider(
    value="Balanced",
    options=["Quick", "Balanced", "Deep", "Comprehensive"],
    key=None
):
    """Create a custom depth slider component.
    
    Parameters
    ----------
    value : str
        The initial selected value
    options : list
        List of available options
    key : str or None
        An optional key that uniquely identifies this component
    
    Returns
    -------
    str
        The selected value
    """
    component_value = _depth_slider(
        value=value,
        options=options,
        key=key,
        default=value
    )

    return component_value 