import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import re
from urllib.parse import urlparse

class DocumentIngester:
    def __init__(self):
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.chunk_size = 500  # words per chunk
    
    def process_uploaded_files(self, uploaded_files) -> List[Dict]:
        """Process uploaded text files and return chunks with metadata."""
        documents = []
        
        for file in uploaded_files:
            if file.size > self.max_file_size:
                st.error(f"File {file.name} exceeds 100MB limit")
                continue
                
            try:
                content = file.read().decode('utf-8')
                chunks = self._chunk_text(content, file.name)
                documents.extend(chunks)
                st.success(f"Processed {file.name}: {len(chunks)} chunks")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        return documents
    
    def process_urls(self, urls: List[str]) -> List[Dict]:
        """Scrape content from URLs and return chunks with metadata."""
        documents = []
        
        for url in urls:
            try:
                if not self._is_valid_url(url):
                    st.error(f"Invalid URL: {url}")
                    continue
                    
                content = self._scrape_url(url)
                if content:
                    chunks = self._chunk_text(content, url)
                    documents.extend(chunks)
                    st.success(f"Scraped {url}: {len(chunks)} chunks")
                else:
                    st.error(f"No content extracted from {url}")
            except Exception as e:
                st.error(f"Error scraping {url}: {str(e)}")
        
        return documents
    
    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into chunks of approximately chunk_size words."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'content': chunk_text,
                'source': source,
                'chunk_id': f"{source}_{i//self.chunk_size}",
                'word_count': len(chunk_words)
            })
        
        return chunks
    
    def _scrape_url(self, url: str) -> str:
        """Extract text content from a URL."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from common content containers
        content_selectors = [
            'article', 'main', '.content', '#content', 
            '.post', '.entry', '.article-body'
        ]
        
        text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([elem.get_text() for elem in elements])
                break
        
        # Fallback to body text if no content containers found
        if not text:
            body = soup.find('body')
            if body:
                text = body.get_text()
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False