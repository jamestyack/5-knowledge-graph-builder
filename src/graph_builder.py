import networkx as nx
import openai
import streamlit as st
from typing import List, Dict, Tuple
import json
import re
from collections import defaultdict, Counter

class KnowledgeGraphBuilder:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.graph = nx.Graph()
        self.node_metadata = {}
        self.concept_cache = {}
    
    def build_graph(self, documents: List[Dict]) -> nx.Graph:
        """Build knowledge graph from document chunks."""
        st.info("Extracting concepts from documents...")
        
        all_concepts = []
        concept_to_sources = defaultdict(list)
        
        # Process documents in batches
        progress_bar = st.progress(0)
        for i, doc in enumerate(documents):
            concepts = self._extract_concepts(doc['content'])
            all_concepts.extend(concepts)
            
            # Track which documents contain which concepts
            for concept in concepts:
                concept_to_sources[concept].append(doc)
            
            progress_bar.progress((i + 1) / len(documents))
        
        st.info("Building graph structure...")
        self._build_nodes(all_concepts, concept_to_sources)
        self._build_edges(all_concepts, documents)
        
        st.success(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using GPT-4."""
        # Check cache first
        cache_key = hash(text[:100])  # Use first 100 chars as cache key
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        prompt = f"""
        Extract 5-10 key concepts, entities, and topics from the following text. 
        Return ONLY a JSON list of concepts as strings. Focus on:
        - Named entities (people, places, organizations)
        - Important topics and themes
        - Technical terms and concepts
        - Key events or processes
        
        Text: {text[:1000]}...
        
        Return format: ["concept1", "concept2", "concept3"]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            # Extract JSON from response
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                concepts = json.loads(json_match.group())
                # Clean and validate concepts
                concepts = [c.strip() for c in concepts if isinstance(c, str) and len(c.strip()) > 2]
                self.concept_cache[cache_key] = concepts
                return concepts
            
        except Exception as e:
            st.warning(f"Error extracting concepts: {str(e)}")
        
        # Fallback: simple keyword extraction
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(words[:10]))
    
    def _build_nodes(self, all_concepts: List[str], concept_to_sources: Dict):
        """Add nodes to the graph with metadata."""
        concept_counts = Counter(all_concepts)
        
        for concept, count in concept_counts.items():
            if count >= 2:  # Only include concepts that appear multiple times
                node_type = self._classify_concept(concept)
                
                self.graph.add_node(
                    concept,
                    weight=count,
                    type=node_type,
                    sources=len(concept_to_sources[concept])
                )
                
                self.node_metadata[concept] = {
                    'frequency': count,
                    'type': node_type,
                    'source_documents': concept_to_sources[concept][:3]  # Keep first 3 sources
                }
    
    def _build_edges(self, all_concepts: List[str], documents: List[Dict]):
        """Add edges based on concept co-occurrence."""
        # Track concept co-occurrence within documents
        cooccurrence = defaultdict(int)
        
        for doc in documents:
            doc_concepts = [c for c in self._extract_concepts(doc['content']) if c in self.graph.nodes()]
            
            # Add edges for concepts that appear in the same document
            for i, concept1 in enumerate(doc_concepts):
                for concept2 in doc_concepts[i+1:]:
                    edge_key = tuple(sorted([concept1, concept2]))
                    cooccurrence[edge_key] += 1
        
        # Add edges with weights
        for (concept1, concept2), weight in cooccurrence.items():
            if weight >= 2:  # Only include edges with multiple co-occurrences
                self.graph.add_edge(concept1, concept2, weight=weight)
    
    def _classify_concept(self, concept: str) -> str:
        """Classify concept type based on simple heuristics."""
        if concept[0].isupper() and len(concept.split()) <= 3:
            if any(word in concept.lower() for word in ['inc', 'corp', 'ltd', 'company']):
                return 'organization'
            elif any(word in concept.lower() for word in ['university', 'institute', 'school']):
                return 'institution'
            else:
                return 'entity'
        elif concept.lower() in ['technology', 'innovation', 'research', 'development', 'science']:
            return 'concept'
        else:
            return 'topic'
    
    def get_node_details(self, node: str) -> Dict:
        """Get detailed information about a node."""
        if node not in self.graph.nodes():
            return {}
        
        node_data = self.graph.nodes[node]
        metadata = self.node_metadata.get(node, {})
        
        neighbors = list(self.graph.neighbors(node))
        edge_weights = {neighbor: self.graph[node][neighbor]['weight'] 
                       for neighbor in neighbors}
        
        return {
            'concept': node,
            'type': node_data.get('type', 'unknown'),
            'frequency': node_data.get('weight', 0),
            'sources': node_data.get('sources', 0),
            'neighbors': neighbors,
            'edge_weights': edge_weights,
            'source_documents': metadata.get('source_documents', [])
        }
    
    def find_relevant_nodes(self, query: str, top_k: int = 10) -> List[str]:
        """Find nodes most relevant to a query."""
        query_concepts = self._extract_concepts(query)
        
        # Score nodes based on query concept similarity
        node_scores = defaultdict(float)
        
        for node in self.graph.nodes():
            # Direct match
            if node.lower() in query.lower() or any(qc.lower() in node.lower() for qc in query_concepts):
                node_scores[node] += 10
            
            # Partial match
            for qc in query_concepts:
                if qc.lower() in node.lower() or node.lower() in qc.lower():
                    node_scores[node] += 5
            
            # Neighbor boost
            for neighbor in self.graph.neighbors(node):
                if neighbor.lower() in query.lower():
                    node_scores[node] += 2
        
        # Sort by score and return top k
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        return [node for node, score in sorted_nodes[:top_k] if score > 0]