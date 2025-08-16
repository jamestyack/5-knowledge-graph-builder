import networkx as nx
import openai
import streamlit as st
from typing import List, Dict, Tuple
import json
import re
from collections import defaultdict, Counter

class KnowledgeGraphBuilder:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        self.graph = nx.Graph()
        self.node_metadata = {}
        self.concept_cache = {}
        self.processing_mode = "fast"  # Default to fast mode
    
    def set_processing_mode(self, mode_string: str):
        """Set processing mode based on UI selection."""
        if "Fast" in mode_string:
            self.processing_mode = "fast"
        elif "Accurate" in mode_string:
            self.processing_mode = "accurate"
        elif "Premium" in mode_string:
            self.processing_mode = "premium"
    
    def build_graph(self, documents: List[Dict]) -> nx.Graph:
        """Build knowledge graph from document chunks."""
        st.info("Extracting concepts from documents...")
        
        all_concepts = []
        concept_to_sources = defaultdict(list)
        
        # Process documents in batches for better performance
        batch_size = 5  # Process 5 documents at once
        progress_bar = st.progress(0)
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract concepts from batch
            batch_concepts = self._extract_concepts_batch(batch)
            
            # Process results
            for j, doc in enumerate(batch):
                concepts = batch_concepts[j] if j < len(batch_concepts) else []
                all_concepts.extend(concepts)
                
                # Track which documents contain which concepts
                for concept in concepts:
                    concept_to_sources[concept].append(doc)
            
            progress_bar.progress(min(1.0, (i + batch_size) / len(documents)))
        
        st.info("Building graph structure...")
        self._build_nodes(all_concepts, concept_to_sources)
        self._build_edges(all_concepts, documents)
        
        st.success(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _extract_concepts_batch(self, documents: List[Dict]) -> List[List[str]]:
        """Extract concepts from multiple documents in a single API call."""
        # Check cache for all documents first
        results = []
        uncached_docs = []
        uncached_indices = []
        
        for i, doc in enumerate(documents):
            cache_key = hash(doc['content'][:100])
            if cache_key in self.concept_cache:
                results.append(self.concept_cache[cache_key])
            else:
                results.append(None)  # Placeholder
                uncached_docs.append(doc)
                uncached_indices.append(i)
        
        # Process uncached documents based on mode
        if uncached_docs and self.processing_mode != "fast":
            try:
                # Create batch prompt
                batch_texts = []
                for doc in uncached_docs:
                    batch_texts.append(doc['content'][:800])  # Limit text length
                
                prompt = f"""
                Extract 5-8 key concepts from each of the following {len(batch_texts)} text segments.
                Return ONLY a JSON array where each element is an array of concepts for that text.
                
                Focus on: named entities, important topics, technical terms, key events.
                
                Texts:
                {chr(10).join([f"Text {i+1}: {text}" for i, text in enumerate(batch_texts)])}
                
                Return format: [["concept1", "concept2"], ["concept3", "concept4"], ...]
                """
                
                # Choose model based on processing mode
                model = "gpt-4" if self.processing_mode == "premium" else "gpt-4o-mini"
                
                if not self.client:
                    raise Exception("No API key provided")
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800
                )
                
                content = response.choices[0].message.content.strip()
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                
                if json_match:
                    batch_concepts = json.loads(json_match.group())
                    
                    # Validate and clean results
                    for i, concepts in enumerate(batch_concepts):
                        if i < len(uncached_docs):
                            clean_concepts = [c.strip() for c in concepts if isinstance(c, str) and len(c.strip()) > 2]
                            cache_key = hash(uncached_docs[i]['content'][:100])
                            self.concept_cache[cache_key] = clean_concepts
                            results[uncached_indices[i]] = clean_concepts
                
            except Exception as e:
                st.warning(f"Batch extraction failed, using fallback: {str(e)}")
                # Fallback for uncached documents
                for i in uncached_indices:
                    if results[i] is None:
                        results[i] = self._extract_concepts_fallback(documents[i]['content'])
        
        # Ensure all results are filled
        for i, result in enumerate(results):
            if result is None:
                results[i] = self._extract_concepts_fallback(documents[i]['content'])
        
        return results
    
    def _extract_concepts_fallback(self, text: str) -> List[str]:
        """Fast fallback concept extraction using regex patterns."""
        concepts = set()
        
        # Extract capitalized phrases (likely entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.update(capitalized[:8])
        
        # Extract technical terms (words ending in -tion, -ment, -ness, etc.)
        technical = re.findall(r'\b\w+(?:tion|ment|ness|ity|ism|ogy|ing)\b', text, re.IGNORECASE)
        concepts.update([t.lower() for t in technical[:5]])
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        concepts.update([q.strip() for q in quoted[:3]])
        
        return list(concepts)[:10]
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using GPT-4 (legacy single method)."""
        # Check cache first
        cache_key = hash(text[:100])
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        # Use fallback for speed during development
        return self._extract_concepts_fallback(text)
    
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