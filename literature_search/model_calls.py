import asyncio
from datetime import datetime
import logging
from dateutil import parser
from dataclasses import asdict
from typing import Any, Dict, List
import json
from literature_search.models import Quote, SearchResult, GroundedReview, ReviewSection, Section
from literature_search.pattern_extractor import QuoteExtractor
from typing import Tuple
import time
from literature_search.prompts import generate_intro_conclusion_prompt, generate_literature_review_prompt, generate_rating_prompt, generate_revise_search_query_for_crossref_prompt, generate_revise_search_query_for_pubmed_prompt, generate_section_prompt, generate_theme_prompt
from llms.client import call_llm_async, call_llm_async_json

def parse_gemini_response(response: str) -> str:
    """
    Parses the model's response to extract the PubMed query.
    Assumes that the response contains only the query string.
    """
    # Clean the response by removing any code blocks or extra text
    query = response.strip()
    # Remove surrounding quotes or backticks if present
    query = query.strip('`"\'')
    return query

def format_papers_for_gemini(similar_papers: List[Dict[str, Any]]) -> str:
    """
    Format the top similar papers for Gemini input.
    
    Args:
        similar_papers (List[Dict[str, Any]]): The list of similar papers' metadata.
        
    Returns:
        str: The formatted string containing titles and texts of the papers.
    """
    formatted_papers = []
    for paper in similar_papers:
        title = paper.get('title', 'No Title')
        text = paper.get('abstract') or paper.get('full_text_url') or 'No Text Available'
        formatted_papers.append(f"{title}\n\n{text}")
    return "\n\n".join(formatted_papers)

async def generate_literature_review(formatted_papers: str) -> str:
    print("formatted_papers", formatted_papers)
    """
    Generate a literature review using the Gemini API based on the top similar papers.
    
    Args:
        formatted_papers (str): The formatted string of top similar papers.
        
    Returns:
        str: The generated literature review.
    """
    
    prompt = generate_literature_review_prompt(formatted_papers)
    response = await call_llm_async(prompt)
    literature_review = parse_gemini_response(response)
    return literature_review

from typing import List, Dict
import asyncio
import logging
from dataclasses import dataclass

@dataclass
class Quote:
    text: str
    section: str
    similarity_score: float
    paper_doi: str
    paper_title: str

@dataclass
class ReviewSection:
    theme: str
    content: str
    quotes: List[Quote]
    papers_cited: List[str]

@dataclass
class GroundedReview:
    title: str
    introduction: str
    sections: List[ReviewSection]
    conclusion: str
    citations: List[str]

async def generate_grounded_review(
    papers: List[SearchResult],
    query: str,
    quote_extractor: QuoteExtractor,
    logger: logging.Logger
) -> GroundedReview:
    """Generate a literature review with verified quotes and patterns."""
    
    # Extract patterns and quotes from all papers at once
    pattern_results = await quote_extractor.extract_patterns(papers)
    
    # Restructure pattern matches into quotes with source information
    pattern_quotes = process_pattern_matches(pattern_results)

    # Step 1: Generate themes from patterns
    themes = await generate_themes(pattern_results, pattern_quotes, query, logger)
    
    # Step 2: Generate sections concurrently
    sections = await process_themes_concurrently(themes, pattern_quotes, query, logger)
    
    # Step 3: Generate introduction and conclusion
    intro_conclusion = await generate_intro_conclusion(sections, query, logger)
    
    # Create citations from all sections
    citations = create_citations(pattern_quotes)
    
    return GroundedReview(
        title=f"Literature Review: {query}",
        introduction=intro_conclusion["introduction"],
        sections=sections,
        conclusion=intro_conclusion["conclusion"],
        citations=sorted(set(citations))
    )

async def process_themes_concurrently(
    themes: List[dict],
    pattern_quotes: List[dict],
    query: str,
    logger: logging.Logger
) -> List[ReviewSection]:
    """Process all themes concurrently and collect results."""
    tasks = [
        process_theme(theme, pattern_quotes, query, logger)
        for theme in themes
        if theme.get("relevant_patterns")
    ]
    
    # Wait for all theme processing to complete
    sections = await asyncio.gather(*tasks)
    
    # Filter out None results from failed theme processing
    return [section for section in sections if section]

async def generate_themes(
    pattern_results: dict,
    pattern_quotes: List[dict],
    query: str,
    logger: logging.Logger
) -> List[dict]:
    """Generate themes from patterns and quotes."""

    # First, pass all unique patterns to Gemini
    unique_patterns = {quote["pattern"] for quote in pattern_quotes}
    pattern_list = list(unique_patterns)

    # Adjust the theme_prompt accordingly
    theme_prompt = generate_theme_prompt(query, pattern_quotes)
    try:
        themes = await call_llm_async_json(theme_prompt)
        logger.debug(f"Raw Themes from Gemini: {themes}")

        # Validate themes
        valid_themes = []
        for theme in themes.get("themes", []):
            # Ensure unique and valid pattern names
            valid_patterns = [p for p in theme.get("relevant_patterns", []) if p in unique_patterns]
            
            if valid_patterns:
                theme["relevant_patterns"] = valid_patterns
                valid_themes.append(theme)
                logger.info(f"Theme '{theme['theme']}' created with patterns: {valid_patterns}")
            else:
                logger.warning(f"Skipping theme '{theme['theme']}' - no valid patterns")

        logger.debug(f"Valid Themes: {valid_themes}")
        return valid_themes

    except Exception as e:
        logger.error(f"Error generating themes: {e}")
        return []

async def process_theme(
    theme: dict,
    pattern_quotes: List[dict],
    query: str,
    logger: logging.Logger
) -> ReviewSection:
    """Process a single theme and generate its section content."""
    try:
        # Get relevant quotes for this theme
        theme_quotes = [
            q for q in pattern_quotes 
            if q['pattern'] in theme["relevant_patterns"]
        ]
        
        if not theme_quotes:
            logger.warning(f"No quotes found for theme: {theme['theme']}")
            logger.warning(f"Patterns requested: {theme['relevant_patterns']}")
            logger.warning("Available patterns in quotes: " + 
                         str({q['pattern'] for q in pattern_quotes}))
            return None

        # Log successful match
        logger.info(f"Found {len(theme_quotes)} quotes for theme '{theme['theme']}'")
        
        section_content = await generate_section_content(theme, theme_quotes, query)
        
        # Create Quote objects for this section
        section_quotes = [
            Quote(
                text=q["text"],
                section=q["section"],
                similarity_score=1.0,
                paper_doi=q["paper_doi"],
                paper_title=q["paper_title"]
            )
            for q in theme_quotes
        ]
        
        # Get unique paper DOIs for citations
        papers_cited = list(set(
            q["paper_doi"] for q in theme_quotes 
            if q["paper_doi"]
        ))
        
        logger.info(f"Generated section for '{theme['theme']}' with {len(section_quotes)} quotes and {len(papers_cited)} citations")
        
        return ReviewSection(
            theme=theme["theme"],
            content=section_content["content"],
            quotes=section_quotes,
            papers_cited=papers_cited
        )
        
    except Exception as e:
        logger.error(f"Error processing theme {theme['theme']}: {str(e)}")
        return None

async def generate_section_content(theme: dict, theme_quotes: List[dict], query: str) -> dict:
    """Generate content for a single section."""
    # Group quotes by pattern for better organization
    quotes_by_pattern = {}
    for quote in theme_quotes:
        pattern = quote["pattern"]
        if pattern not in quotes_by_pattern:
            quotes_by_pattern[pattern] = []
        quotes_by_pattern[pattern].append(quote)

    section_prompt = generate_section_prompt(query, theme, quotes_by_pattern)

    return await call_llm_async_json(section_prompt)

async def revise_search_query_for_pubmed(query: str, enhance_with_mesh: bool = True) -> str:
    """
    Revise the search query for PubMed by identifying key concepts and optionally enhancing with MeSH terms.
    
    Args:
        query (str): The original search query or research statement
        enhance_with_mesh (bool): Whether to enhance query with MeSH terms
        
    Returns:
        dict: Contains revised query and other metadata
    """
    query_prompt = generate_revise_search_query_for_pubmed_prompt(query)
    return await call_llm_async_json(query_prompt)

async def revise_search_query_for_crossref(query: str) -> str:
    """
    Revise the search query for Crossref by identifying key concepts and structuring appropriately.
    
    Args:
        query (str): The original search query or research statement
        
    Returns:
        dict: Contains revised query and other metadata
    """
    query_prompt = generate_revise_search_query_for_crossref_prompt(query)
    return await call_llm_async_json(query_prompt)

async def generate_intro_conclusion(
    sections: List[ReviewSection],
    query: str,
    logger: logging.Logger
) -> dict:
    """Generate introduction and conclusion based on all sections."""
    intro_conclusion_prompt = generate_intro_conclusion_prompt(query, sections)
    try:
        return await call_llm_async_json(intro_conclusion_prompt)
    except Exception as e:
        logger.error(f"Error generating introduction and conclusion: {e}")
        return {
            "introduction": "Error generating introduction",
            "conclusion": "Error generating conclusion"
        }

def process_pattern_matches(pattern_results: dict) -> List[dict]:
    """Process pattern matches into a flat list of quotes with metadata."""
    pattern_quotes = []
    for paper_match in pattern_results["pattern_matches"]:
        # Process abstract matches
        for match in paper_match["matches"]["abstract"]:
            pattern_quotes.append({
                "text": match["context"],
                "pattern": match["pattern"],
                "matched_text": match["matched_text"],
                "paper_doi": paper_match["doi"],
                "paper_title": paper_match["title"],
                "authors": paper_match["authors"],
                "section": "abstract"
            })
        
        # Process full text matches
        for section in paper_match["matches"]["full_text"]:
            for match in section["matches"]:
                pattern_quotes.append({
                    "text": match["context"],
                    "pattern": match["pattern"],
                    "matched_text": match["matched_text"],
                    "paper_doi": paper_match["doi"],
                    "paper_title": paper_match["title"],
                    "authors": paper_match["authors"],
                    "section": section["section"]
                })
    return pattern_quotes

def create_citations(pattern_quotes: List[dict]) -> List[str]:
    """Create formatted citations from pattern quotes."""
    citations = set()
    for quote in pattern_quotes:
        citation = f"{', '.join(quote['authors'])}. {quote['paper_title']}. DOI: {quote['paper_doi']}"
        citations.add(citation)
    return sorted(citations)

async def organize_sections(sections: List[Section], logger: logging.Logger) -> Dict[str, List[Section]]:
    """Organize sections into relevant and other categories using Gemini."""
    # First organize sections by frequency
    section_counts = {}
    for section in sections:
        normalized_heading = section.heading.lower().strip()
        if len(normalized_heading) <= 1:  # Skip very short headings
            continue
        if normalized_heading not in section_counts:
            section_counts[normalized_heading] = []
        section_counts[normalized_heading].append(section)

    # Sort by frequency and take top 20
    sorted_sections = sorted(
        section_counts.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:20]

    prompt = """
    Analyze these section headings from research papers and categorize them into:
    1. Relevant Sections - Core research sections like Methods, Results, Discussion
    2. Other Sections - Supporting or less central sections
    
    Section headings and their frequencies:
    """
    
    for heading, sections in sorted_sections:
        prompt += f"\n{heading} (appears in {len(sections)} papers)"

    prompt += "\n\nReturn the categorization in JSON format with 'relevant' and 'other' arrays."

    try:
        categories = await call_llm_async_json(prompt)
        
        organized = {
            "relevant": [],
            "other": []
        }
        
        # Map the categorized headings back to their sections
        for heading, sections in section_counts.items():
            if heading in categories["relevant"]:
                organized["relevant"].extend(sections)
            else:
                organized["other"].extend(sections)
                
        return organized
        
    except Exception as e:
        logger.error(f"Error organizing sections: {e}")
        return {"relevant": [], "other": []}
    
async def rate_papers_with_gemini(papers: List[Dict[str, Any]], query: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Rates the relevance of papers using the Gemini API in batches of 10.

    Args:
        papers (List[Dict[str, Any]]): List of papers with metadata.
        query (str): The original search query.
        logger (logging.Logger): Logger for logging information.

    Returns:
        List[Dict[str, Any]]: List containing DOI, composite score, relevance summary, journal impact factor, and formatted date.
    """
    batch_size = 10
    batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]
    tasks = [rate_paper_batch(batch, query, logger) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rated_papers = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error in rating batch: {str(result)}")
            continue
        rated_papers.extend(result)

    return rated_papers

async def rate_paper_batch(batch: List[Dict[str, Any]], query: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Rates a single batch of papers using Gemini.

    Args:
        batch (List[Dict[str, Any]]): A batch of up to 10 papers.
        query (str): The original search query.
        logger (logging.Logger): Logger for logging information.

    Returns:
        List[Dict[str, Any]]: Rated papers with DOI, composite score, relevance summary, journal impact factor, and formatted date.
    """
    prompt = generate_rating_prompt(batch, query)
    try:
        result = await call_llm_async_json(prompt)
        logger.debug(f"Rated {len(result)} papers in batch.")
        return result
    except Exception as e:
        logger.error(f"Error rating batch with Gemini: {str(e)}")
        return []




