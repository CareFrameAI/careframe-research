import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from literature_search.prompts import generate_process_single_abstract_prompt
import numpy as np
import spacy
import logging
from literature_search.models import Quote, SearchResult, Section
from llms.client import call_llm_async
class QuoteExtractor:
    def __init__(self, logger: logging.Logger):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.logger = logger
        self.nlp = spacy.load("en_core_sci_md")
        self.cached_patterns = None
        try:
            self.base_patterns = {}
        except Exception as e:
            self.logger.error(f"Error loading patterns.json: {e}")
            self.base_patterns = {}
            
        # Convert patterns to spaCy format
        self.matcher = spacy.matcher.Matcher(self.nlp.vocab)
        self._load_patterns()

    def _load_patterns(self, additional_patterns: List[Dict] = None):
        """Load patterns into spaCy matcher."""
        # Clear existing patterns
        self.matcher = spacy.matcher.Matcher(self.nlp.vocab)

        # Load base patterns
        for pattern_name, pattern in self.base_patterns.items():
            try:
                self.matcher.add(pattern_name, [pattern])
            except Exception as e:
                self.logger.warning(f"Error adding pattern {pattern_name}: {e}")

        # Add any additional patterns
        if additional_patterns:
            for pattern_dict in additional_patterns:
                for pattern_name, pattern in pattern_dict.items():
                    # unique_pattern_name = f"custom_{pattern_name}"
                    try:
                        self.matcher.add(pattern_name, [pattern])
                    except Exception as e:
                        self.logger.warning(f"Error adding custom pattern {pattern_name}: {e}")


    async def generate_new_patterns(self, similar_papers: List[SearchResult]) -> Dict:
        """Use Gemini to generate new patterns from similar papers."""
        import asyncio
        from typing import Dict, List

        async def process_single_abstract(paper: SearchResult, attempt: int = 0) -> List[Dict]:
            """Process a single abstract with retries."""
            if not paper.abstract:
                return []
            prompt = generate_process_single_abstract_prompt(paper)
            try:
                response = await call_llm_async(prompt)

                # Clean and validate JSON response
                json_str = response.strip()
                if not json_str.startswith('['):
                    json_str = json_str[json_str.find('['):]
                if not json_str.endswith(']'):
                    json_str = json_str[:json_str.rfind(']')+1]

                patterns = json.loads(json_str)

                # Validate patterns structure
                if not isinstance(patterns, list):
                    raise ValueError("Response must be a JSON array")

                for pattern in patterns:
                    if not isinstance(pattern, dict) or len(pattern) != 1:
                        raise ValueError("Each pattern must be a single-key dictionary")
                    pattern_name = list(pattern.keys())[0]
                    if not pattern_name.startswith("pattern_"):
                        raise ValueError(f"Pattern name '{pattern_name}' does not follow the naming convention 'pattern_<unique_identifier>'")
                    pattern_tokens = pattern[pattern_name]
                    if not isinstance(pattern_tokens, list):
                        raise ValueError(f"Pattern '{pattern_name}' must contain a token list")
                    for token in pattern_tokens:
                        if not isinstance(token, dict) or len(token) != 1:
                            raise ValueError(f"Invalid token in pattern '{pattern_name}'")

                return patterns

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < 2:  # Allow 3 total attempts (0, 1, 2)
                    self.logger.warning(f"Attempt {attempt+1} failed for paper {paper.title}: {str(e)}")
                    return await process_single_abstract(paper, attempt + 1)
                else:
                    self.logger.error(f"All attempts failed for paper {paper.title}")
                    return []
            except Exception as e:
                self.logger.error(f"Unexpected error processing paper {paper.title}: {str(e)}")
                return []

        # Create tasks for all papers with abstracts
        tasks = [
            process_single_abstract(paper)
            for paper in similar_papers
            if paper.abstract
        ]
        
        if not tasks:
            self.logger.warning("No papers with abstracts found")
            return {}
            
        # Run all tasks concurrently
        try:
            pattern_lists = await asyncio.gather(*tasks)
            
            # Merge all valid patterns
            all_patterns = []
            for patterns in pattern_lists:
                if patterns:  # Only add non-empty pattern lists
                    all_patterns.extend(patterns)
                    
            self.logger.info(f"Successfully generated {len(all_patterns)} patterns from {len(pattern_lists)} papers")
            return all_patterns
            
        except Exception as e:
            self.logger.error(f"Error gathering pattern results: {str(e)}")
            return {}

    async def extract_patterns(self, papers: List[SearchResult], top_n: int = 20) -> Dict:
        """Extract pattern matches from papers with tracking."""
        # Get top n similar papers
        papers = papers[:top_n]
        
        # Generate patterns only if not cached
        if self.cached_patterns is None:
            self.cached_patterns = await self.generate_new_patterns(papers)
            self._load_patterns(self.cached_patterns)

        results = {
            "pattern_matches": [],
            "papers_analyzed": {
                "with_abstract": [],
                "with_full_text": []
            }
        }
        
        # Process each paper
        for paper in papers:
            paper_info = {
                "doi": paper.doi,
                "title": paper.title,
                "authors": paper.authors,
                "matches": {
                    "abstract": [],
                    "full_text": []
                }
            }
            
            # Process abstract
            if paper.abstract:
                results["papers_analyzed"]["with_abstract"].append(paper.doi)
                abstract_matches = self._process_text(paper.abstract)
                if abstract_matches:
                    paper_info["matches"]["abstract"] = abstract_matches
                    
            # Process full text if available
            if paper.sections:
                results["papers_analyzed"]["with_full_text"].append(paper.doi)
                for section in paper.sections:
                    section_matches = self._process_text(section.content)
                    if section_matches:
                        paper_info["matches"]["full_text"].append({
                            "section": section.heading,
                            "matches": section_matches
                        })
                        
            results["pattern_matches"].append(paper_info)
            
        # Add pattern definitions used
        results["patterns_used"] = {
            "base_patterns": self.base_patterns,
            "generated_patterns": self.cached_patterns
        }
        
        return results

    def _process_text(self, text: str) -> List[Dict]:
        """Process text and return pattern matches with context."""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        results = []
        for match_id, start, end in matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]
            
            # Get context (sentences containing the match)
            sent = span.sent
            if sent:
                context = sent.text
            else:
                # Fallback to fixed window if sentence boundary detection fails
                context_window = 100
                start_idx = max(0, span.start_char - context_window)
                end_idx = min(len(text), span.end_char + context_window)
                context = text[start_idx:end_idx]
            
            results.append({
                "pattern": pattern_name,
                "matched_text": span.text,
                "context": context.strip(),
                "char_span": (span.start_char, span.end_char)
            })
            
        return results

    def clear_pattern_cache(self):
        """Clear the cached patterns."""
        self.cached_patterns = None

    def verify_matches(self, matches: List[Dict], source_text: str) -> List[Dict]:
        """Verify pattern matches exist in source text."""
        verified = []
        for match in matches:
            # Check if matched text exists in source
            if match["matched_text"] in source_text:
                # Verify context
                if match["context"] in source_text:
                    verified.append(match)
                else:
                    # Try to reconstruct context
                    start_idx = source_text.find(match["matched_text"])
                    if start_idx != -1:
                        context_window = 100
                        context_start = max(0, start_idx - context_window)
                        context_end = min(len(source_text), 
                                        start_idx + len(match["matched_text"]) + context_window)
                        match["context"] = source_text[context_start:context_end].strip()
                        verified.append(match)
                        
        return verified