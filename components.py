"""UI components for the MARA application."""

import streamlit as st
from typing import Dict, List, Optional, Tuple, Callable
import markdown
import pdfkit
import re

from state import AppState

# Cache the markdown conversion
@st.cache_data(ttl=3600)
def convert_markdown_to_html(content: str) -> str:
    """Convert markdown to HTML with caching."""
    return markdown.markdown(content)

def display_logo():
    """Display the application logo."""
    st.image("assets/mara-logo.png", use_container_width=True)

def input_form(state: AppState, on_submit: Callable) -> None:
    """Display the main input form."""
    with st.form("topic_form", clear_on_submit=False):
        topic = st.text_area(
            "What would you like to explore?",
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
            help="Choose 1-5 iterations. More iterations = deeper insights = longer wait."
        )
        
        if state.stage == 'input':
            if st.form_submit_button("🚀 Start Analysis", use_container_width=True, type="primary"):
                on_submit(topic, iterations)
        else:
            if st.form_submit_button("❌ Cancel", use_container_width=True, type="secondary"):
                state.soft_reset()
                st.rerun()

def display_insights(insights: Dict[str, str]) -> None:
    """Display insights in expandable containers."""
    if not insights:
        return
        
    with st.container():
        with st.expander("💡 Did You Know?", expanded=True):
            st.markdown(insights['did_you_know'])
        
        with st.expander("⚡ Overview", expanded=True):
            st.markdown(insights['eli5'])

def display_focus_areas(state: AppState, on_continue: Callable, on_skip: Callable) -> None:
    """Display focus areas selection interface."""
    if not state.focus_areas:
        st.error("Failed to load focus areas. Please try again.")
        return
    
    with st.expander("🎯 Focus Areas", expanded=state.focus_container_expanded):
        st.write("Choose up to 5 areas to focus your analysis on (optional):")
        
        cols = st.columns(2)
        selected = []
        
        for i, area in enumerate(state.focus_areas):
            col_idx = i % 2
            with cols[col_idx]:
                key = f"focus_area_{i}"
                is_selected = area in state.selected_areas
                
                if len(selected) < 5 or is_selected:
                    if st.checkbox(area, value=is_selected, key=key):
                        selected.append(area)
                else:
                    st.checkbox(area, value=False, key=key, disabled=True)
        
        state.selected_areas = selected
        
        st.markdown("---")
        num_selected = len(selected)
        
        if num_selected > 0:
            st.success(f"✅ Selected {num_selected} focus area{'s' if num_selected > 1 else ''}")
            st.write("Selected areas:")
            for area in selected:
                st.write(f"- {area}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Skip", type="secondary", key="skip_button"):
                on_skip()
        with col2:
            if st.button("Continue", type="primary", key="continue_button"):
                on_continue(selected)

def display_research_analysis(analysis: Dict[str, str]) -> None:
    """Display a single research analysis output."""
    if not analysis:
        return
        
    with st.expander("📚 " + analysis["title"], expanded=False):
        st.markdown(f"### {analysis['title']}")
        st.markdown(f"*{analysis['subtitle']}*")
        st.markdown("---")
        st.markdown(analysis["content"])

def generate_download_content(synthesis: Dict[str, str], format: str) -> Tuple[str, str, bytes]:
    """Generate downloadable content in specified format."""
    if format == "markdown":
        content = f"# {synthesis['title']}\n\n"
        content += f"*{synthesis['subtitle']}*\n\n"
        content += synthesis['content']
        return "text/markdown", "synthesis_report.md", content.encode('utf-8')
    elif format == "txt":
        content = f"{synthesis['title']}\n"
        content += f"{synthesis['subtitle']}\n\n"
        content += re.sub(r'\*\*|•|\n###', '', synthesis['content'])
        return "text/plain", "synthesis_report.txt", content.encode('utf-8')
    else:  # PDF
        html_content = f"<h1>{synthesis['title']}</h1>"
        html_content += f"<p><em>{synthesis['subtitle']}</em></p>"
        html_content += convert_markdown_to_html(synthesis['content'])
        pdf_content = pdfkit.from_string(html_content, False)
        return "application/pdf", "synthesis_report.pdf", pdf_content

def display_synthesis(synthesis: Dict[str, str]) -> None:
    """Display synthesis report with download options."""
    if not synthesis:
        return
        
    with st.expander("📑 " + synthesis["title"], expanded=False):
        st.markdown(f"### {synthesis['title']}")
        st.markdown(f"*{synthesis['subtitle']}*")
        st.markdown("---")
        st.markdown(synthesis["content"])
        
        st.markdown("---")
        st.markdown("### Download Report")
        
        formats = {
            "PDF": "pdf",
            "Markdown": "markdown",
            "Text": "txt"
        }
        
        col1, col2 = st.columns([1, 2])
        with col1:
            format_choice = st.selectbox("Format:", list(formats.keys()))
        with col2:
            mime_type, filename, content = generate_download_content(synthesis, formats[format_choice])
            st.download_button(
                f"📥 Download {format_choice}",
                data=content,
                file_name=filename,
                mime=mime_type,
                use_container_width=True
            ) 