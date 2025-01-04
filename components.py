"""UI components for the MARA application."""

import streamlit as st
from typing import Dict, List, Callable

def display_logo() -> None:
    """Display the application logo."""
    st.image("assets/logo.png", width=120)

def input_form(state, handle_submit: Callable) -> None:
    """Display the topic input form."""
    with st.form("topic_form"):
        topic = st.text_area(
            "Enter a topic to begin the research process",
            value=state.last_topic if state.last_topic else "",
            key="topic_input"
        )
        
        col1, col2 = st.columns([4, 1])
        with col1:
            submit = st.form_submit_button("Start Analysis", type="primary")
        with col2:
            iterations = st.number_input(
                "Iterations",
                min_value=1,
                max_value=5,
                value=2,
                key="iterations_input"
            )
            
        if submit and topic:
            handle_submit(topic, iterations)

def display_insights(insights: Dict[str, str]) -> None:
    """Display the initial insights."""
    if not insights:
        return
        
    st.markdown("## 💡 Initial Insights")
    
    with st.expander("View Overview", expanded=True):
        st.info(f"**Did you know?** {insights.get('did_you_know', '')}")
        st.markdown("### Overview")
        st.markdown(insights.get('eli5', ''))

def display_focus_areas(state, handle_continue: Callable, handle_skip: Callable) -> None:
    """Display the focus area selection."""
    if not state.focus_areas:
        return
        
    st.markdown("## 🎯 Research Focus Areas")
    
    with st.expander("Select Focus Areas", expanded=state.focus_container_expanded):
        st.markdown("""
        Select up to 5 areas to focus the research on, or skip to analyze all areas.
        """)
        
        # Create columns for focus area selection
        cols = st.columns(2)
        selected = []
        
        for i, area in enumerate(state.focus_areas):
            with cols[i % 2]:
                if st.checkbox(area, key=f"focus_{i}"):
                    selected.append(area)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Skip", key="skip_focus", type="secondary"):
                handle_skip()
        with col2:
            if st.button("Continue", key="continue_focus", type="primary", disabled=len(selected) > 5):
                handle_continue(selected)
                
        if len(selected) > 5:
            st.warning("Please select no more than 5 focus areas.") 