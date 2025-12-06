"""
Streamlit App for Multi-Agent Research Assistant
================================================
Beautiful UI with real-time progress tracking and interactive results
"""

import streamlit as st
import sys
import os
from datetime import datetime
import json

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the multi-agent system
from multi_agent_system import MultiAgentSystem

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .agent-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .researcher-box {
        background-color: #e3f2fd;
        border-color: #2196f3;
    }
    .analyst-box {
        background-color: #f3e5f5;
        border-color: #9c27b0;
    }
    .writer-box {
        background-color: #e8f5e9;
        border-color: #4caf50;
    }
    .critic-box {
        background-color: #fff3e0;
        border-color: #ff9800;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None


def initialize_system(token):
    """Initialize the multi-agent system"""
    try:
        with st.spinner("🤖 Initializing AI agents..."):
            system = MultiAgentSystem(token=token, max_iterations=2)
        st.success("✅ System initialized successfully!")
        return system
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return None


def display_progress(step):
    """Display progress bar based on current step"""
    steps = {
        "researcher": (25, "🔍 Researching..."),
        "analyst": (50, "📊 Analyzing..."),
        "writer": (75, "✍️ Writing report..."),
        "critic": (90, "🎯 Quality check..."),
        "complete": (100, "✅ Complete!")
    }
    
    if step in steps:
        progress, text = steps[step]
        st.progress(progress)
        st.info(text)


def display_agent_output(agent_name, content, box_class):
    """Display agent output in a styled box"""
    st.markdown(f'<div class="agent-box {box_class}">', unsafe_allow_html=True)
    st.markdown(f"### {agent_name}")
    st.markdown(content)
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Multi-Agent Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by LangGraph & Advanced AI Agents</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Token input
        token = st.text_input(
            "HuggingFace API Token",
            type="password",
            help="Enter your HuggingFace API token"
        )
        
        # Initialize button
        if st.button("🚀 Initialize System"):
            if token:
                st.session_state.system = initialize_system(token)
            else:
                st.error("Please enter your HuggingFace token")
        
        st.divider()
        
        # System status
        st.header("📊 System Status")
        if st.session_state.system:
            st.success("🟢 System Active")
            st.metric("Research Queries", len(st.session_state.research_history))
        else:
            st.warning("🔴 System Inactive")
        
        st.divider()
        
        # Example questions
        st.header("💡 Example Questions")
        examples = [
            "what is 2+2",
            "calculate (15*3)+7",
            "what is artificial intelligence",
            "what is machine learning",
            "what is python programming"
        ]
        
        for example in examples:
            if st.button(f"📝 {example}", key=f"ex_{example}"):
                st.session_state.example_question = example
        
        st.divider()
        
        # History
        st.header("📚 Research History")
        if st.session_state.research_history:
            for i, item in enumerate(reversed(st.session_state.research_history[-5:])):
                with st.expander(f"🔍 {item['question'][:30]}..."):
                    st.write(f"**Time:** {item['timestamp']}")
                    st.write(f"**Score:** {item['score']}/10")
        else:
            st.info("No research history yet")
        
        st.divider()
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.research_history = []
            st.session_state.current_result = None
            st.rerun()
    
    # Main content
    if not st.session_state.system:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👈 Please initialize the system using the sidebar")
            
            st.markdown("### 🌟 Features")
            features = [
                "🔍 **Smart Research**: Automatic tool selection and execution",
                "📊 **Deep Analysis**: AI-powered insight extraction",
                "✍️ **Professional Reports**: Well-structured documentation",
                "🎯 **Quality Assurance**: Automated review and refinement",
                "🔄 **Iterative Improvement**: Multiple revision cycles"
            ]
            for feature in features:
                st.markdown(feature)
            
            st.markdown("### 🛠️ Technology Stack")
            tech = [
                "LangGraph for agent orchestration",
                "Meta Llama 3.1 8B Instruct",
                "Pydantic for structured outputs",
                "NumExpr for safe calculations"
            ]
            for item in tech:
                st.markdown(f"- {item}")
    
    else:
        # Research interface
        st.markdown("## 🔍 Ask Your Question")
        
        # Check if example was clicked
        default_question = st.session_state.get('example_question', '')
        if default_question:
            st.session_state.example_question = ''
        
        question = st.text_input(
            "Enter your research question:",
            value=default_question,
            placeholder="e.g., what is 2+2, what is artificial intelligence...",
            key="question_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            research_button = st.button("🚀 Start Research", type="primary")
        with col2:
            clear_button = st.button("🔄 Clear Results")
        
        if clear_button:
            st.session_state.current_result = None
            st.rerun()
        
        if research_button and question:
            # Create progress container
            progress_container = st.container()
            result_container = st.container()
            
            with progress_container:
                st.markdown("### 🔄 Research in Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Capture output
                import io
                from contextlib import redirect_stdout
                
                output_capture = io.StringIO()
                
                try:
                    with redirect_stdout(output_capture):
                        # Run research
                        final_state = st.session_state.system.research(question)
                    
                    # Update progress
                    progress_bar.progress(100)
                    status_text.success("✅ Research Complete!")
                    
                    if final_state:
                        # Store result
                        st.session_state.current_result = final_state
                        
                        # Add to history
                        st.session_state.research_history.append({
                            'question': question,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'score': final_state['critique_output'].score
                        })
                        
                        st.rerun()
                    
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.error(f"❌ Error: {str(e)}")
        
        # Display results
        if st.session_state.current_result:
            st.markdown("---")
            result = st.session_state.current_result
            
            # Metrics row
            st.markdown("## 📊 Research Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Quality Score", f"{result['critique_output'].score}/10")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Iterations", result['report_iterations'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{result['research_output'].confidence*100:.0f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                sources = ", ".join(result['research_output'].sources_used)
                st.metric("Sources", len(result['research_output'].sources_used))
                st.caption(sources)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Tabbed interface for results
            tab1, tab2, tab3, tab4 = st.tabs(["📄 Final Report", "🔍 Research", "📊 Analysis", "🎯 Quality Review"])
            
            with tab1:
                report = result['report_output']
                st.markdown(f"# {report.title}")
                st.markdown(report.content)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report.content,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with tab2:
                research = result['research_output']
                display_agent_output(
                    "🔍 Research Agent",
                    f"""
**Answer:** {research.answer}

**Sources Used:** {', '.join(research.sources_used)}

**Confidence:** {research.confidence*100:.1f}%
                    """,
                    "researcher-box"
                )
            
            with tab3:
                analysis = result['analysis_output']
                display_agent_output(
                    "📊 Analysis Agent",
                    f"""
**Key Points:**
{chr(10).join(f'• {point}' for point in analysis.key_points)}

**Implications:**
{analysis.implications}
                    """,
                    "analyst-box"
                )
            
            with tab4:
                critique = result['critique_output']
                
                # Score gauge
                score = critique.score
                color = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
                
                st.markdown(f"### {color} Quality Score: {score}/10")
                st.progress(score / 10)
                
                st.markdown(f"""
**Status:** {"✅ Approved" if not critique.needs_revision else "🔄 Needs Revision"}
                """)
            
            # Export all data
            st.markdown("---")
            if st.button("📦 Export Full Research Data (JSON)"):
                export_data = {
                    'question': result['question'],
                    'timestamp': datetime.now().isoformat(),
                    'research': {
                        'answer': result['research_output'].answer,
                        'sources': result['research_output'].sources_used,
                        'confidence': result['research_output'].confidence
                    },
                    'analysis': {
                        'key_points': result['analysis_output'].key_points,
                        'implications': result['analysis_output'].implications
                    },
                    'report': {
                        'title': result['report_output'].title,
                        'content': result['report_output'].content
                    },
                    'quality': {
                        'score': result['critique_output'].score,
                        'needs_revision': result['critique_output'].needs_revision
                    }
                }
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()