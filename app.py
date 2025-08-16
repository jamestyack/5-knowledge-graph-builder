import streamlit as st
import os
from dotenv import load_dotenv
import networkx as nx
from src.ingestion import DocumentIngester
from src.graph_builder import KnowledgeGraphBuilder
from src.visualization import GraphVisualizer
from src.qa_engine import QAEngine

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Knowledge Graph Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'graph' not in st.session_state:
    st.session_state.graph = nx.Graph()
if 'graph_builder' not in st.session_state:
    st.session_state.graph_builder = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def main():
    st.title("🧠 Knowledge Graph Builder")
    st.markdown("Transform document archives into an interactive, visual knowledge graph with natural language Q&A capabilities.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key for GPT-4 processing"
        )
        
        if not api_key:
            st.error("Please enter your OpenAI API key to proceed")
            return
        
        st.divider()
        
        # Document ingestion section
        st.header("📄 Document Ingestion")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Text Files",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload up to 100MB total of text files"
        )
        
        # URL input
        st.subheader("Or add URLs")
        url_input = st.text_area(
            "URLs (one per line)",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            help="Enter URLs to scrape content from"
        )
        
        # Process documents button
        if st.button("🚀 Build Knowledge Graph", type="primary"):
            if uploaded_files or url_input.strip():
                process_documents(api_key, uploaded_files, url_input)
            else:
                st.error("Please upload files or enter URLs")
    
    # Main content area
    if st.session_state.processing_complete:
        show_main_interface()
    else:
        show_welcome_screen()

def process_documents(api_key: str, uploaded_files, url_input: str):
    """Process uploaded files and URLs to build knowledge graph."""
    with st.spinner("Processing documents and building knowledge graph..."):
        try:
            # Initialize components
            ingester = DocumentIngester()
            st.session_state.graph_builder = KnowledgeGraphBuilder(api_key)
            
            # Process documents
            documents = []
            
            # Process uploaded files
            if uploaded_files:
                file_docs = ingester.process_uploaded_files(uploaded_files)
                documents.extend(file_docs)
            
            # Process URLs
            if url_input.strip():
                urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                url_docs = ingester.process_urls(urls)
                documents.extend(url_docs)
            
            if not documents:
                st.error("No documents were successfully processed")
                return
            
            # Build knowledge graph
            st.session_state.documents = documents
            st.session_state.graph = st.session_state.graph_builder.build_graph(documents)
            st.session_state.processing_complete = True
            
            st.success(f"Knowledge graph built successfully! Processed {len(documents)} document chunks.")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

def show_welcome_screen():
    """Show welcome screen with instructions."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🚀 Get Started")
        st.markdown("""
        **Follow these steps to build your knowledge graph:**
        
        1. **Enter your OpenAI API key** in the sidebar
        2. **Upload text files** (TXT format) or **enter URLs** to scrape
        3. **Click "Build Knowledge Graph"** to start processing
        4. **Explore the interactive visualization** and **ask questions**
        
        ---
        
        **Example use cases:**
        - 📚 Research paper analysis
        - 📋 Policy document exploration  
        - 📰 News article clustering
        - 📖 Book/literature analysis
        - 🏢 Company knowledge base
        """)
        
        st.info("💡 **Tip:** Start with 3-5 related documents for best results")

def show_main_interface():
    """Show main interface with graph visualization and Q&A."""
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🕸️ Knowledge Graph", "❓ Ask Questions", "📊 Graph Statistics"])
    
    with tab1:
        show_graph_visualization()
    
    with tab2:
        show_qa_interface()
    
    with tab3:
        show_graph_statistics()

def show_graph_visualization():
    """Show interactive graph visualization."""
    st.header("Interactive Knowledge Graph")
    
    if st.session_state.graph.number_of_nodes() == 0:
        st.warning("No graph data available. Please process some documents first.")
        return
    
    # Graph controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Controls")
        
        # Node highlighting
        if st.button("🔄 Refresh Layout"):
            st.rerun()
        
        # Node selection for highlighting
        all_nodes = list(st.session_state.graph.nodes())
        selected_nodes = st.multiselect(
            "Highlight Nodes",
            options=all_nodes,
            help="Select nodes to highlight in the visualization"
        )
    
    with col1:
        # Create and display visualization
        visualizer = GraphVisualizer()
        fig = visualizer.create_interactive_plot(
            st.session_state.graph, 
            highlighted_nodes=selected_nodes
        )
        
        # Display the plot
        selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        
        # Show node details if clicked
        if selected_data and 'selection' in selected_data:
            points = selected_data['selection'].get('points', [])
            if points:
                show_node_details(points[0])

def show_node_details(point_data):
    """Show details for a selected node."""
    if 'text' in point_data:
        node_name = point_data['text'].replace("...", "")
        
        # Find the actual node name
        matching_nodes = [node for node in st.session_state.graph.nodes() 
                         if node.startswith(node_name) or node_name in node]
        
        if matching_nodes:
            node = matching_nodes[0]
            details = st.session_state.graph_builder.get_node_details(node)
            
            st.subheader(f"📍 Node Details: {node}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Type", details.get('type', 'Unknown'))
            with col2:
                st.metric("Frequency", details.get('frequency', 0))
            with col3:
                st.metric("Connections", len(details.get('neighbors', [])))
            
            # Show connected nodes
            if details.get('neighbors'):
                st.write("**Connected to:**")
                for neighbor in details['neighbors'][:10]:
                    weight = details['edge_weights'].get(neighbor, 0)
                    st.write(f"- {neighbor} (strength: {weight})")
            
            # Show source documents
            if details.get('source_documents'):
                st.write("**Appears in:**")
                for doc in details['source_documents'][:3]:
                    source = doc.get('source', 'Unknown')
                    if len(source) > 50:
                        source = source[:47] + "..."
                    st.write(f"- {source}")

def show_qa_interface():
    """Show Q&A interface."""
    st.header("Ask Questions About Your Documents")
    
    if not st.session_state.processing_complete:
        st.warning("Please build a knowledge graph first using the sidebar.")
        return
    
    # Initialize QA engine
    api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key", type="password")
    qa_engine = QAEngine(api_key)
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the main themes discussed in the documents?",
        help="Ask any question about the content of your documents"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        ask_button = st.button("🔍 Ask Question", type="primary")
    
    if ask_button and question.strip():
        with st.spinner("Searching knowledge graph and generating answer..."):
            try:
                # Get answer
                result = qa_engine.answer_question(
                    question, 
                    st.session_state.graph, 
                    st.session_state.graph_builder,
                    st.session_state.documents
                )
                
                # Display answer
                st.subheader("💡 Answer")
                st.write(result['answer'])
                
                # Display confidence and metadata
                col1, col2 = st.columns(2)
                with col1:
                    confidence_color = "green" if result['confidence'] > 0.7 else "orange" if result['confidence'] > 0.4 else "red"
                    st.markdown(f"**Confidence:** :{confidence_color}[{result['confidence']:.1%}]")
                
                with col2:
                    st.write(f"**Sources used:** {len(result['sources'])}")
                
                # Show relevant concepts
                if result['relevant_nodes']:
                    st.subheader("🔗 Relevant Concepts")
                    for node in result['relevant_nodes'][:8]:
                        st.badge(node)
                
                # Show sources
                if result['sources']:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(result['sources'], 1):
                            st.write(f"**Source {i}:** {source['source']}")
                            st.write(source['content_preview'])
                            st.divider()
                
                # Show reasoning
                with st.expander("🧠 How was this answer found?"):
                    reasoning = qa_engine.explain_reasoning(
                        question, 
                        result['relevant_nodes'], 
                        result['sources']
                    )
                    st.markdown(reasoning)
                
                # Follow-up questions
                followups = qa_engine.get_followup_questions(
                    question, 
                    result['answer'], 
                    result['relevant_nodes']
                )
                
                if followups:
                    st.subheader("🤔 Follow-up Questions")
                    for fq in followups:
                        if st.button(fq, key=f"followup_{hash(fq)}"):
                            st.session_state.question = fq
                            st.rerun()
                            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

def show_graph_statistics():
    """Show graph statistics and analysis."""
    st.header("Knowledge Graph Statistics")
    
    if not st.session_state.processing_complete:
        st.warning("Please build a knowledge graph first using the sidebar.")
        return
    
    visualizer = GraphVisualizer()
    stats = visualizer.create_network_stats(st.session_state.graph)
    visualizer.create_stats_display(stats)
    
    # Document statistics
    st.subheader("📄 Document Statistics")
    
    if st.session_state.documents:
        total_chunks = len(st.session_state.documents)
        total_words = sum(doc.get('word_count', 0) for doc in st.session_state.documents)
        unique_sources = len(set(doc['source'] for doc in st.session_state.documents))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Document Chunks", total_chunks)
        with col2:
            st.metric("Total Words", f"{total_words:,}")
        with col3:
            st.metric("Unique Sources", unique_sources)
        
        # Source breakdown
        st.subheader("📊 Sources Breakdown")
        source_counts = {}
        for doc in st.session_state.documents:
            source = doc['source']
            if len(source) > 50:
                source = source[:47] + "..."
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            st.write(f"**{source}:** {count} chunks")

if __name__ == "__main__":
    main()