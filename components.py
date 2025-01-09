"""UI components for the MARA application."""

import streamlit as st
from typing import Dict, List, Callable

from state import AppState

def display_logo() -> None:
    """Display the application logo."""
    st.image("assets/mara-logo.png", use_container_width=True)

def input_form(state: AppState, on_submit: Callable) -> None:
    """Display the main input form."""
    with st.form("topic_form", clear_on_submit=False):
        topic = st.text_area(
            "What do you want to explore?",
            value=state.last_topic,
            help="Enter your research topic or question.",
            placeholder="e.g., 'Examine the impact of artificial intelligence on healthcare...'"
        )
        
        iterations = st.number_input(
            "Number of Analysis Iterations",
            min_value=1,
            max_value=5,
            value=state.iterations,
            step=1,
            help="Choose 1-5 iterations. More Iterations = Deeper Insights & Longer Wait."
        )
        
        if state.stage == 'input':
            if st.form_submit_button("🚀 Start Analysis", use_container_width=True, type="primary"):
                on_submit(topic, iterations)
        else:
            if st.form_submit_button("❌ Cancel", use_container_width=True, type="secondary"):
                state.soft_reset()
                st.rerun()

def display_insights(insights: Dict[str, str]) -> None:
    """Display the initial insights."""
    if not insights:
        return
        
    # Display Did You Know section
    with st.expander("💡 Did You Know?", expanded=True):
        st.markdown(insights.get('did_you_know', ''))
    
    # Display TL;DR section
    with st.expander("📝 TL;DR", expanded=True):
        st.markdown(insights.get('eli5', ''))

def display_focus_areas(state, handle_continue: Callable, handle_skip: Callable) -> None:
    """Display the focus area selection."""
    if not state.focus_areas:
        return
        
    with st.expander("🎯 Research Focus Areas", expanded=state.focus_container_expanded):
        if state.selected_focus_areas:
            # Display selected focus areas as a list
            st.markdown("## Selected Focus Areas")
            for area in state.selected_focus_areas:
                st.markdown(f"• {area}")
        else:
            st.markdown("## Select Focus Areas")
            st.markdown("Select up to 5 areas to focus the research on, or skip to analyze all areas.")
            
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