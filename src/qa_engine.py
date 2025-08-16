import openai
import streamlit as st
import networkx as nx
from typing import List, Dict, Tuple
import re
from collections import defaultdict

class QAEngine:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.response_cache = {}
    
    def answer_question(self, question: str, graph: nx.Graph, graph_builder, documents: List[Dict]) -> Dict:
        """Answer a question using RAG approach with graph context."""
        # Check cache
        cache_key = hash(question.lower().strip())
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Find relevant nodes in the graph
        relevant_nodes = graph_builder.find_relevant_nodes(question, top_k=10)
        
        # Get relevant document chunks
        relevant_chunks = self._find_relevant_chunks(question, relevant_nodes, documents, graph_builder)
        
        # Generate answer using GPT-4
        answer_data = self._generate_answer(question, relevant_chunks, relevant_nodes)
        
        # Cache and return
        self.response_cache[cache_key] = answer_data
        return answer_data
    
    def _find_relevant_chunks(self, question: str, relevant_nodes: List[str], 
                            documents: List[Dict], graph_builder) -> List[Dict]:
        """Find document chunks relevant to the question and nodes."""
        relevant_chunks = []
        chunk_scores = defaultdict(float)
        
        # Score chunks based on relevant nodes
        for chunk in documents:
            content_lower = chunk['content'].lower()
            question_lower = question.lower()
            
            # Direct question keyword match
            question_words = set(re.findall(r'\b\w+\b', question_lower))
            content_words = set(re.findall(r'\b\w+\b', content_lower))
            keyword_overlap = len(question_words.intersection(content_words))
            
            chunk_scores[chunk['chunk_id']] += keyword_overlap * 2
            
            # Relevant node mentions
            for node in relevant_nodes:
                if node.lower() in content_lower:
                    chunk_scores[chunk['chunk_id']] += 10
                
                # Partial matches
                node_words = set(re.findall(r'\b\w+\b', node.lower()))
                overlap = len(node_words.intersection(content_words))
                chunk_scores[chunk['chunk_id']] += overlap
        
        # Sort chunks by relevance score
        sorted_chunks = sorted(
            [(chunk, chunk_scores[chunk['chunk_id']]) for chunk in documents],
            key=lambda x: x[1], reverse=True
        )
        
        # Return top 5 most relevant chunks
        return [chunk for chunk, score in sorted_chunks[:5] if score > 0]
    
    def _generate_answer(self, question: str, relevant_chunks: List[Dict], 
                        relevant_nodes: List[str]) -> Dict:
        """Generate answer using GPT-4 with context."""
        if not relevant_chunks:
            return {
                'answer': "I couldn't find relevant information in the knowledge graph to answer your question.",
                'confidence': 0.0,
                'sources': [],
                'relevant_nodes': []
            }
        
        # Prepare context
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"Source {i+1} ({chunk['source']}):\n{chunk['content'][:500]}...")
            sources.append({
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'content_preview': chunk['content'][:200] + "..."
            })
        
        context = "\n\n".join(context_parts)
        nodes_context = ", ".join(relevant_nodes[:10])
        
        prompt = f"""
        Based on the following context from a knowledge graph, answer the question as accurately as possible.
        
        Question: {question}
        
        Relevant concepts from knowledge graph: {nodes_context}
        
        Context from documents:
        {context}
        
        Instructions:
        1. Provide a clear, direct answer based on the context
        2. If the context doesn't contain enough information, say so
        3. Reference specific sources when possible
        4. Keep the answer concise but informative
        5. Indicate your confidence level (high/medium/low)
        
        Answer:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            confidence = self._estimate_confidence(answer, relevant_chunks, question)
            
            return {
                'answer': answer,
                'confidence': confidence,
                'sources': sources,
                'relevant_nodes': relevant_nodes
            }
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'sources': sources,
                'relevant_nodes': relevant_nodes
            }
    
    def _estimate_confidence(self, answer: str, chunks: List[Dict], question: str) -> float:
        """Estimate confidence based on answer content and available context."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if answer references sources
        if any(word in answer.lower() for word in ['according to', 'based on', 'source', 'document']):
            confidence += 0.2
        
        # Increase confidence if multiple chunks support the answer
        if len(chunks) >= 3:
            confidence += 0.1
        
        # Decrease confidence if answer indicates uncertainty
        uncertainty_phrases = [
            'not enough information', 'unclear', 'uncertain', 'possibly', 'might be',
            'i don\'t know', 'cannot determine', 'insufficient context'
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.3
        
        # Increase confidence if answer is detailed
        if len(answer.split()) > 50:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def get_followup_questions(self, question: str, answer: str, relevant_nodes: List[str]) -> List[str]:
        """Generate relevant follow-up questions based on the context."""
        try:
            prompt = f"""
            Based on this Q&A exchange and the related concepts, suggest 3-5 relevant follow-up questions:
            
            Original Question: {question}
            Answer: {answer}
            Related Concepts: {', '.join(relevant_nodes[:10])}
            
            Generate follow-up questions that would help explore the topic deeper.
            Return only the questions, one per line.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            questions = [q.strip() for q in content.split('\n') if q.strip() and '?' in q]
            return questions[:5]
            
        except Exception as e:
            # Fallback: generate simple follow-up questions
            return [
                f"What more can you tell me about {relevant_nodes[0]}?" if relevant_nodes else "Can you provide more details?",
                "What are the related concepts?",
                "Are there any examples mentioned?"
            ]
    
    def explain_reasoning(self, question: str, relevant_nodes: List[str], 
                         relevant_chunks: List[Dict]) -> str:
        """Explain how the answer was derived."""
        explanation = f"**How I found this answer:**\n\n"
        
        explanation += f"1. **Analyzed your question:** '{question}'\n\n"
        
        if relevant_nodes:
            explanation += f"2. **Found {len(relevant_nodes)} relevant concepts in the knowledge graph:**\n"
            for i, node in enumerate(relevant_nodes[:5], 1):
                explanation += f"   - {node}\n"
            if len(relevant_nodes) > 5:
                explanation += f"   - ... and {len(relevant_nodes) - 5} more\n"
            explanation += "\n"
        
        if relevant_chunks:
            explanation += f"3. **Retrieved {len(relevant_chunks)} relevant document sections:**\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                source = chunk['source']
                if len(source) > 50:
                    source = source[:47] + "..."
                explanation += f"   - From: {source}\n"
            explanation += "\n"
        
        explanation += "4. **Generated answer by combining information from the knowledge graph and source documents**"
        
        return explanation