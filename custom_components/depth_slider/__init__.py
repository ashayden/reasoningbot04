"""Custom depth slider component for Streamlit."""

import os
import streamlit.components.v1 as components
import streamlit as st

# Check if we're in development or production mode
_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "depth_slider",
        url="http://localhost:3001",
    )
else:
    # Get the build directory relative to this file
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    
    if not os.path.exists(build_dir):
        raise RuntimeError(
            "Build directory not found. Run 'npm run build' in the frontend directory first."
        )
    
    # Declare the component with an explicit name and path
    _component_func = components.declare_component(
        "depth_slider",
        path=build_dir
    )

def depth_slider(
    value="Balanced",
    options=["Quick", "Balanced", "Deep", "Comprehensive"],
    key=None,
    label="Analysis Depth"
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
    label : str
        The label to display above the slider
    
    Returns
    -------
    str
        The selected value
    """
    # Ensure options are strings
    options = [str(opt) for opt in options]
    
    # Ensure value is in options
    if value not in options:
        value = options[0]
    
    # Call the component function
    component_value = _component_func(
        value=value,
        options=options,
        key=key,
        label=label,
        default=value  # Provide a default value
    )
    
    return component_value or value  # Return the current value or default 