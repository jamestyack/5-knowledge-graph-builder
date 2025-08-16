import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import streamlit as st
import numpy as np
from typing import Dict, List, Optional

class GraphVisualizer:
    def __init__(self):
        self.color_map = {
            'entity': '#FF6B6B',      # Red
            'organization': '#4ECDC4', # Teal
            'institution': '#45B7D1',  # Blue
            'concept': '#96CEB4',      # Green
            'topic': '#FFEAA7',       # Yellow
            'unknown': '#DDA0DD'       # Plum
        }
    
    def create_interactive_plot(self, graph: nx.Graph, highlighted_nodes: List[str] = None) -> go.Figure:
        """Create interactive Plotly visualization of the knowledge graph."""
        if graph.number_of_nodes() == 0:
            return self._create_empty_plot()
        
        # Calculate layout
        pos = self._calculate_layout(graph)
        
        # Extract node and edge data
        node_trace, edge_trace = self._create_traces(graph, pos, highlighted_nodes)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title="Knowledge Graph",
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive Knowledge Graph - Click nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#666666", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def _calculate_layout(self, graph: nx.Graph) -> Dict:
        """Calculate node positions using spring layout."""
        if graph.number_of_nodes() == 1:
            node = list(graph.nodes())[0]
            return {node: (0, 0)}
        
        # Use different layout algorithms based on graph size
        if graph.number_of_nodes() < 50:
            pos = nx.spring_layout(graph, k=3, iterations=50)
        elif graph.number_of_nodes() < 200:
            pos = nx.spring_layout(graph, k=2, iterations=30)
        else:
            pos = nx.spring_layout(graph, k=1, iterations=20)
        
        return pos
    
    def _create_traces(self, graph: nx.Graph, pos: Dict, highlighted_nodes: List[str] = None) -> tuple:
        """Create node and edge traces for Plotly."""
        highlighted_nodes = highlighted_nodes or []
        
        # Edge trace
        edge_x, edge_y = [], []
        edge_weights = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(graph[edge[0]][edge[1]].get('weight', 1))
        
        # Normalize edge weights for line width
        if edge_weights:
            max_weight = max(edge_weights)
            normalized_weights = [2 + (w / max_weight) * 8 for w in edge_weights]
        else:
            normalized_weights = [2]
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Node trace
        node_x, node_y, node_colors, node_sizes, node_text, hover_text = [], [], [], [], [], []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node attributes
            node_data = graph.nodes[node]
            node_type = node_data.get('type', 'unknown')
            weight = node_data.get('weight', 1)
            sources = node_data.get('sources', 0)
            
            # Color based on type or highlight
            if node in highlighted_nodes:
                color = '#FF0000'  # Red for highlighted
            else:
                color = self.color_map.get(node_type, self.color_map['unknown'])
            node_colors.append(color)
            
            # Size based on weight/frequency
            size = min(50, max(10, weight * 3))
            node_sizes.append(size)
            
            # Text labels
            display_text = node[:20] + "..." if len(node) > 20 else node
            node_text.append(display_text)
            
            # Hover information
            hover_info = f"<b>{node}</b><br>"
            hover_info += f"Type: {node_type}<br>"
            hover_info += f"Frequency: {weight}<br>"
            hover_info += f"Sources: {sources}<br>"
            hover_info += f"Connections: {len(list(graph.neighbors(node)))}"
            hover_text.append(hover_info)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_text,
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color="white"),
                opacity=0.8
            )
        )
        
        return node_trace, edge_trace
    
    def _create_empty_plot(self) -> go.Figure:
        """Create empty plot when no graph data is available."""
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        return fig
    
    def create_network_stats(self, graph: nx.Graph) -> Dict:
        """Calculate and return network statistics."""
        if graph.number_of_nodes() == 0:
            return {}
        
        stats = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_clustering': nx.average_clustering(graph) if graph.number_of_nodes() > 2 else 0,
        }
        
        # Node type distribution
        node_types = {}
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        stats['node_types'] = node_types
        
        # Most connected nodes
        degree_centrality = nx.degree_centrality(graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        stats['top_connected'] = [(node, round(centrality, 3)) for node, centrality in top_nodes]
        
        return stats
    
    def create_stats_display(self, stats: Dict) -> None:
        """Display network statistics in Streamlit."""
        if not stats:
            st.info("No graph statistics available")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", stats['nodes'])
        with col2:
            st.metric("Edges", stats['edges'])
        with col3:
            st.metric("Density", f"{stats['density']:.3f}")
        with col4:
            st.metric("Avg Clustering", f"{stats['avg_clustering']:.3f}")
        
        # Node type distribution
        if stats.get('node_types'):
            st.subheader("Node Types")
            type_data = stats['node_types']
            fig_pie = px.pie(
                values=list(type_data.values()),
                names=list(type_data.keys()),
                color_discrete_map=self.color_map
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Top connected nodes
        if stats.get('top_connected'):
            st.subheader("Most Connected Concepts")
            for i, (node, centrality) in enumerate(stats['top_connected'], 1):
                st.write(f"{i}. **{node}** (centrality: {centrality})")
    
    def highlight_subgraph(self, graph: nx.Graph, center_nodes: List[str], max_depth: int = 2) -> List[str]:
        """Get nodes within max_depth of center_nodes for highlighting."""
        highlighted = set(center_nodes)
        
        for center in center_nodes:
            if center in graph:
                # Get nodes within max_depth
                for node in nx.single_source_shortest_path_length(graph, center, cutoff=max_depth):
                    highlighted.add(node)
        
        return list(highlighted)