import os
import json
import pickle
import tempfile
import hashlib
import logging
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi  # pip install rank-bm25

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class PaperRanker:
    """
    Memory-efficient paper ranking system that uses multiple ranking methods
    and stores large data structures on disk to minimize memory usage.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the PaperRanker with optional cache directory.
        
        Args:
            cache_dir: Directory to store cached data. If None, uses a temporary directory.
        """
        self.logger = logging.getLogger("PaperRanker")
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.cache_dir = self.temp_dir.name
            
        self.logger.info(f"Initialized PaperRanker with cache at {self.cache_dir}")
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Initialize session-based cache identifiers
        self.current_session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Configure weights for different ranking factors
        self.weights = {
            'keyword': 0.2,
            'tfidf': 0.35,
            'bm25': 0.35,
            'recency': 0.1
        }

    def __del__(self):
        """Clean up any temporary directories when the object is destroyed."""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
            
    def _get_cache_path(self, key: str, prefix: str = "") -> str:
        """Get the path for a cached file using a hashed key."""
        hash_key = hashlib.md5(f"{prefix}_{key}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.pkl")
    
    def _cache_exists(self, key: str, prefix: str = "") -> bool:
        """Check if a cache file exists."""
        cache_path = self._get_cache_path(key, prefix)
        return os.path.exists(cache_path)
    
    def _save_to_cache(self, data: Any, key: str, prefix: str = "") -> None:
        """Save data to cache file."""
        cache_path = self._get_cache_path(key, prefix)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_from_cache(self, key: str, prefix: str = "") -> Any:
        """Load data from cache file."""
        cache_path = self._get_cache_path(key, prefix)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def clear_cache(self, session_only: bool = True) -> None:
        """
        Clear cached files.
        
        Args:
            session_only: If True, only clears the current session's cache.
        """
        if session_only:
            prefix = self.current_session_id
            for file in os.listdir(self.cache_dir):
                if file.startswith(hashlib.md5(f"{prefix}_".encode()).hexdigest()[:10]):
                    os.remove(os.path.join(self.cache_dir, file))
        else:
            # Clear all cache files but preserve the directory
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing, removing stopwords, and stemming.
        
        Args:
            text: The text to preprocess.
            
        Returns:
            List of processed tokens.
        """
        if not text:
            return []
            
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        
        # Stem tokens
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def extract_key_terms(self, query: str, n: int = 15) -> List[str]:
        """
        Extract key terms from a query with improved extraction.
        
        Args:
            query: The query text.
            n: The number of key terms to extract.
            
        Returns:
            List of key terms.
        """
        # Use custom key terms if they have been set
        if hasattr(self, 'custom_key_terms') and self.custom_key_terms:
            # Process the custom terms to match our stemming/preprocessing
            processed_custom_terms = []
            for term in self.custom_key_terms:
                # Preprocess each term
                processed_tokens = self.preprocess_text(term)
                processed_custom_terms.extend(processed_tokens)
            
            # Return the processed custom terms, limited to n
            return processed_custom_terms[:n]
        
        # Otherwise use the standard extraction method
        # Pre-process the query
        processed_tokens = self.preprocess_text(query)
        
        # Use a more sophisticated method that considers term frequency
        if len(processed_tokens) > 0:
            # Count token frequencies
            from collections import Counter
            token_counts = Counter(processed_tokens)
            
            # Filter out very common words that might have slipped through stopwords
            common_fillers = {'use', 'using', 'used', 'studi', 'research', 'paper', 'articl'}
            key_terms = [term for term, count in token_counts.most_common(n*2) 
                        if term not in common_fillers]
            
            # Return the most frequent terms, up to n
            return key_terms[:n]
        
        return []

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Preprocess a query for ranking.
        
        Args:
            query: The query text.
            
        Returns:
            Dict with processed query information.
        """
        processed_query = {
            'text': query,
            'tokens': self.preprocess_text(query),
            'key_terms': self.extract_key_terms(query)
        }
        return processed_query
    
    def preprocess_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Preprocess papers for ranking.
        
        Args:
            papers: List of paper dictionaries with at least 'title' and 'abstract'.
            
        Returns:
            Dict with processed paper information.
        """
        # Generate a cache key based on paper content
        cache_key = hashlib.md5(str([p.get('title', '') + p.get('abstract', '') 
                                  for p in papers]).encode()).hexdigest()
        
        # Check if we've already processed these papers
        if self._cache_exists(cache_key, self.current_session_id):
            return self._load_from_cache(cache_key, self.current_session_id)
        
        # Process papers
        processed_papers = []
        raw_docs = []
        
        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            # Combine title and abstract for processing
            full_text = f"{title} {abstract}"
            processed_tokens = self.preprocess_text(full_text)
            
            # Keep track of raw document for TF-IDF
            raw_docs.append(full_text.lower())
            
            # Extract publication year for recency scoring
            year = None
            if 'year' in paper:
                try:
                    year = int(paper['year'])
                except (ValueError, TypeError):
                    pass
            elif 'date' in paper:
                try:
                    year = int(paper['date'].split('-')[0])
                except (ValueError, TypeError, IndexError, AttributeError):
                    pass
            
            processed_paper = {
                'id': paper.get('id', str(hash(title))),
                'title': title,
                'abstract': abstract,
                'tokens': processed_tokens,
                'year': year,
                'original': paper  # Keep reference to original paper
            }
            processed_papers.append(processed_paper)
        
        # Create corpus for BM25
        corpus = [p['tokens'] for p in processed_papers]
        bm25 = BM25Okapi(corpus)
        
        # Create TF-IDF vectorizer and document vectors
        # Use sklearn's TfidfVectorizer for efficiency
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.8)
        tfidf_matrix = vectorizer.fit_transform(raw_docs)
        
        # Save vectorizer and matrix to disk
        tfidf_cache_key = f"{cache_key}_tfidf"
        matrix_cache_key = f"{cache_key}_matrix"
        
        self._save_to_cache(vectorizer, tfidf_cache_key, self.current_session_id)
        self._save_to_cache(tfidf_matrix, matrix_cache_key, self.current_session_id)
        
        # Create a BM25 index
        bm25_cache_key = f"{cache_key}_bm25"
        self._save_to_cache(bm25, bm25_cache_key, self.current_session_id)
        
        # Create the final processed data structure
        processed_data = {
            'papers': processed_papers,
            'cache_key': cache_key,
            'paper_count': len(papers)
        }
        
        # Cache the processed data
        self._save_to_cache(processed_data, cache_key, self.current_session_id)
        
        return processed_data
    
    def calculate_keyword_scores(self, processed_query: Dict[str, Any], 
                               processed_papers: Dict[str, Any]) -> List[float]:
        """
        Calculate keyword-based relevance scores.
        
        Args:
            processed_query: Processed query.
            processed_papers: Processed papers.
            
        Returns:
            List of scores for each paper.
        """
        key_terms = processed_query['key_terms']
        scores = []
        
        for paper in processed_papers['papers']:
            paper_tokens = set(paper['tokens'])
            
            # Count matching key terms
            matches = sum(1 for term in key_terms if term in paper_tokens)
            
            # Normalize by number of key terms
            score = matches / max(1, len(key_terms))
            
            scores.append(score)
            
        return scores
    
    def calculate_tfidf_scores(self, processed_query: Dict[str, Any], 
                             processed_papers: Dict[str, Any]) -> List[float]:
        """
        Calculate TF-IDF similarity scores.
        
        Args:
            processed_query: Processed query.
            processed_papers: Processed papers.
            
        Returns:
            List of scores for each paper.
        """
        cache_key = processed_papers['cache_key']
        
        # Load vectorizer and document matrix from cache
        vectorizer = self._load_from_cache(f"{cache_key}_tfidf", self.current_session_id)
        tfidf_matrix = self._load_from_cache(f"{cache_key}_matrix", self.current_session_id)
        
        if vectorizer is None or tfidf_matrix is None:
            self.logger.error("TF-IDF data not found in cache")
            return [0.0] * len(processed_papers['papers'])
        
        # Transform query to TF-IDF vector
        query_vec = vectorizer.transform([processed_query['text'].lower()])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        return similarities.tolist()
    
    def calculate_bm25_scores(self, processed_query: Dict[str, Any], 
                            processed_papers: Dict[str, Any]) -> List[float]:
        """
        Calculate BM25 scores.
        
        Args:
            processed_query: Processed query.
            processed_papers: Processed papers.
            
        Returns:
            List of scores for each paper.
        """
        cache_key = processed_papers['cache_key']
        
        # Load BM25 index from cache
        bm25 = self._load_from_cache(f"{cache_key}_bm25", self.current_session_id)
        
        if bm25 is None:
            self.logger.error("BM25 index not found in cache")
            return [0.0] * len(processed_papers['papers'])
        
        # Get scores using query tokens
        scores = bm25.get_scores(processed_query['tokens'])
        
        print(f"DEBUG BM25: scores type: {type(scores)}, shape: {scores.shape if hasattr(scores, 'shape') else 'no shape'}")
        print(f"DEBUG BM25: scores sample: {scores[:5] if len(scores) > 5 else scores}")
        
        # Normalize scores to [0,1] range
        # Fix: Check if array has elements instead of using the array in a boolean context
        if len(scores) > 0:
            max_score = np.max(scores)
            if max_score > 0:
                scores = np.array([float(score) / float(max_score) for score in scores])
        else:
            # Empty array case
            scores = np.array([])
        
        print(f"DEBUG BM25: normalized scores sample: {scores[:5] if len(scores) > 5 else scores}")
        
        # Convert to list before returning
        return scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
    
    def calculate_recency_scores(self, processed_papers: Dict[str, Any]) -> List[float]:
        """
        Calculate recency scores.
        
        Args:
            processed_papers: Processed papers.
            
        Returns:
            List of scores for each paper.
        """
        current_year = datetime.now().year
        scores = []
        
        for paper in processed_papers['papers']:
            year = paper.get('year')
            
            if year is None:
                scores.append(0.5)  # Middle value for unknown years
            else:
                # Linear decay over 20 years
                age = max(0, current_year - year)
                score = max(0, 1 - (age / 20))
                scores.append(score)
        
        return scores
    
    def generate_explanations(self, paper: Dict[str, Any], 
                            component_scores: Dict[str, float]) -> str:
        """
        Generate detailed human-readable explanations for paper scores.
        
        Args:
            paper: The processed paper.
            component_scores: Dict with scores for each component.
            
        Returns:
            Explanation string.
        """
        parts = []
        title = paper.get('title', '')
        
        # Overall relevance assessment
        overall_score = sum(score * self.weights[name] for name, score in component_scores.items())
        if overall_score > 0.7:
            parts.append(f"This paper appears highly relevant to your research question.")
        elif overall_score > 0.4:
            parts.append(f"This paper is moderately relevant to your research question.")
        else:
            parts.append(f"This paper has limited relevance to your research question.")
        
        # Explain keyword matching
        keyword_score = component_scores.get('keyword', 0)
        if keyword_score > 0.6:
            parts.append("Strong keyword matches with your research topics.")
        elif keyword_score > 0.3:
            parts.append("Some important keywords from your query appear in this paper.")
        
        # Explain content relevance
        tfidf_score = component_scores.get('tfidf', 0)
        bm25_score = component_scores.get('bm25', 0)
        
        if tfidf_score > 0.7 or bm25_score > 0.7:
            parts.append("The content is highly semantically related to your research question.")
        elif tfidf_score > 0.4 or bm25_score > 0.4:
            parts.append("The content shows moderate semantic similarity to your research question.")
        
        # Explain recency if available
        recency_score = component_scores.get('recency', 0)
        if 'year' in paper and paper['year']:
            if recency_score > 0.8:
                parts.append(f"Very recent publication ({paper['year']}), which may contain the latest findings.")
            elif recency_score > 0.5:
                parts.append(f"Published in {paper['year']}, relatively recent research.")
            else:
                parts.append(f"Published in {paper['year']}. Consider whether newer research might be available.")
        
        return " ".join(parts)
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-10 range with improved handling of outliers.
        
        Args:
            scores: List of raw scores.
            
        Returns:
            Normalized scores.
        """
        if not scores:
            return []
            
        # Debug statements
        print(f"DEBUG normalize_scores: Input type: {type(scores)}")
        print(f"DEBUG normalize_scores: First few scores: {scores[:5] if len(scores) > 5 else scores}")
        
        # Check if all scores are the same
        unique_values = set(float(s) for s in scores)
        print(f"DEBUG normalize_scores: Unique values count: {len(unique_values)}")
        if len(unique_values) <= 1:
            return [5.0] * len(scores)  # All equal, assign middle value
        
        # Handle outliers - use percentiles instead of min/max
        scores_array = np.array(scores)
        print(f"DEBUG normalize_scores: np.array type: {type(scores_array)}, shape: {scores_array.shape}, dtype: {scores_array.dtype}")
        
        min_val = np.percentile(scores_array, 5)  # 5th percentile as effective minimum
        max_val = np.percentile(scores_array, 95)  # 95th percentile as effective maximum
        
        print(f"DEBUG normalize_scores: min_val: {min_val}, type: {type(min_val)}")
        print(f"DEBUG normalize_scores: max_val: {max_val}, type: {type(max_val)}")
        
        # Ensure scalar values for comparison
        min_val_scalar = float(min_val)
        max_val_scalar = float(max_val)
        
        print(f"DEBUG normalize_scores: min_val_scalar: {min_val_scalar}, max_val_scalar: {max_val_scalar}")
        
        # If range is too small, fall back to regular min/max
        if max_val_scalar - min_val_scalar < 0.01:
            min_val_scalar = float(min(scores))
            max_val_scalar = float(max(scores))
            print(f"DEBUG normalize_scores: Range too small, using min: {min_val_scalar}, max: {max_val_scalar}")
        
        # Min-max normalization to [0, 10] range, with clamping for outliers
        normalized = []
        for score in scores:
            score_scalar = float(score)
            # Check for equal min/max
            if max_val_scalar == min_val_scalar:
                normalized.append(5.0)
            else:
                # Normalize and clamp to [0, 10]
                norm_score = ((score_scalar - min_val_scalar) / (max_val_scalar - min_val_scalar)) * 10
                normalized.append(max(0, min(10, norm_score)))
        
        print(f"DEBUG normalize_scores: First few normalized scores: {normalized[:5] if len(normalized) > 5 else normalized}")
        return normalized
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update the weights for different ranking factors.
        
        Args:
            weights: Dictionary with weights for 'keyword', 'tfidf', 'bm25', and 'recency'.
        """
        # Validate weights
        required_keys = ['keyword', 'tfidf', 'bm25', 'recency']
        if not all(key in weights for key in required_keys):
            missing = [key for key in required_keys if key not in weights]
            raise ValueError(f"Missing weights for: {', '.join(missing)}")
        
        # Normalize weights to ensure they sum to 1.0
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Total weight must be greater than zero")
        
        normalized_weights = {k: v/total for k, v in weights.items()}
        self.weights = normalized_weights
        self.logger.info(f"Updated ranking weights: {self.weights}")
    
    def set_key_terms(self, key_terms: List[str]) -> None:
        """
        Set specific key terms to prioritize in the ranking process.
        
        Args:
            key_terms: List of important terms for the current research question
        """
        self.custom_key_terms = key_terms
        self.logger.info(f"Set custom key terms for ranking: {key_terms}")
    
    def rank_papers(self, query: str, papers: List[Dict[str, Any]], 
                  include_rationales: bool = True) -> List[Dict[str, Any]]:
        """
        Rank papers by relevance to the query with improved memory efficiency.
        
        Args:
            query: The research question or hypothesis.
            papers: List of papers to rank.
            include_rationales: Whether to include explanations.
            
        Returns:
            List of dicts with paper info, score, and optionally rationale.
        """
        # ... existing implementation but process in batches for very large datasets
        if len(papers) > 1000:
            # Process in batches of 1000 for very large datasets
            batch_size = 1000
            results = []
            
            for i in range(0, len(papers), batch_size):
                batch = papers[i:i+batch_size]
                batch_results = self._rank_paper_batch(query, batch, include_rationales)
                results.extend(batch_results)
                
            return results
        else:
            # Use existing implementation for reasonable-sized datasets
            return self._rank_paper_batch(query, papers, include_rationales)
    
    def _rank_paper_batch(self, query: str, papers: List[Dict[str, Any]], 
                       include_rationales: bool = True) -> List[Dict[str, Any]]:
        """Rank a batch of papers (internal implementation)."""
        if not papers:
            return []
        
        # Implementation of the existing rank_papers method
        processed_query = self.preprocess_query(query)
        processed_papers = self.preprocess_papers(papers)
        
        # Calculate component scores
        keyword_scores = self.calculate_keyword_scores(processed_query, processed_papers)
        tfidf_scores = self.calculate_tfidf_scores(processed_query, processed_papers)
        bm25_scores = self.calculate_bm25_scores(processed_query, processed_papers)
        recency_scores = self.calculate_recency_scores(processed_papers)
        
        # Calculate composite scores
        composite_scores = []
        component_scores_list = []
        
        for i in range(len(processed_papers['papers'])):
            # Get all component scores for this paper
            components = {
                'keyword': keyword_scores[i],
                'tfidf': tfidf_scores[i],
                'bm25': bm25_scores[i],
                'recency': recency_scores[i]
            }
            component_scores_list.append(components)
            
            # Calculate weighted sum
            weighted_score = sum(score * self.weights[name] for name, score in components.items())
            composite_scores.append(weighted_score)
        
        # Normalize scores to 0-10 range
        normalized_scores = self.normalize_scores(composite_scores)
        
        # Prepare results
        results = []
        for i, paper in enumerate(processed_papers['papers']):
            result = {
                'score': float(round(normalized_scores[i], 1)),
            }
            
            # Add rationale if requested
            if include_rationales:
                result['rationale'] = self.generate_explanations(
                    paper, component_scores_list[i])
            
            results.append(result)
        
        return results
    











