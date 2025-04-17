import json
from typing import Any, List, Dict

from literature_search.models import SearchResult
import logging
from datetime import datetime
from dateutil import parser


def format_date(date_str: Any, doi: str = '') -> str:
    """
    Formats the date to YYYY-MM-DD. Returns a default date if input is invalid.

    Args:
        date_str (Any): Original date input, expected to be a string.
        doi (str): DOI of the paper for logging purposes.

    Returns:
        str: Formatted date string.
    """
    if not isinstance(date_str, str) or not date_str.strip():
        logging.getLogger(__name__).warning(f"Missing or invalid publication_date for DOI: {doi}. Using default date.")
        return '1970-01-01'  # Default date for invalid inputs

    try:
        # Attempt to parse the date string
        date_obj = parser.parse(date_str, default=datetime(1970, 1, 1))
        return date_obj.strftime('%Y-%m-%d')
    except (ValueError, OverflowError) as e:
        logging.getLogger(__name__).warning(f"Unrecognized date format '{date_str}' for DOI: {doi}. Using default date.")
        return '1970-01-01'  # Default date if parsing fails

def generate_literature_review_prompt(formatted_papers: str) -> str:
   prompt = f"""
You are an expert researcher tasked with writing a comprehensive literature review. Below are summaries of 10 similar research papers:

{formatted_papers}

**Instructions:**
1. **Structure:** The literature review should include the following sections:
- **Introduction:** Provide an overview of the topic, its significance, and the purpose of the review.
- **Thematic Sections:** Create distinct thematic sections, each focusing on a unique aspect of the research. Each section must have:
  * A clear, specific title reflecting its unique focus
  * Content that is completely distinct from other sections
  * Analysis specific to that theme's particular aspect
  * No repetition of content from other sections
- **Conclusion:** Synthesize key insights and suggest future research directions.
2. **Format:** Use clear and concise language. Ensure that each section flows logically to the next. Do not summarize papers separately.
3. **Details to Include:** Highlight salient findings, methodologies, and gaps in the research. Compare and contrast the studies where relevant.
4. **Details to Exclude:** Do not include irrelevant information, personal opinions, or unverifiable claims.
5. **Citations:** Do not include citations; focus on summarizing the content.
6. **Tone:** Maintain an objective and analytical tone throughout.

2. **Format:** 
For each thematic section:
- Focus on one specific aspect of the research
- Include only information relevant to that specific theme
- Ensure content is unique to that section
- Cross-reference other sections when needed instead of repeating content

**Format the output in JSON format.**
```json
{{
    "title": "Literature Review Title",
    "introduction": "Introduction text...",
    "thematic_sections": [
        {{
            "title": "First Theme Title",
            "content": "Unique content specific to this theme..."
        }},
        {{
            "title": "Second Theme Title",
            "content": "Different content specific to this second theme..."
        }},
        {{
            "title": "Third Theme Title",
            "content": "Different content specific to this third theme..."
        }},
        {{
            "title": "Fourth Theme Title",
            "content": "Different content specific to this fourth theme..."
        }},
        {{
            "title": "Fifth Theme Title",
            "content": "Different content specific to this fifth theme..."
        }},
        // Additional sections
    ],
    "conclusion": "Conclusion text..."
}}
```
**Length:** 
Maximum 20 sections.

**Generate the literature review based on the above summaries and instructions in json.**
"""
   return prompt


def generate_theme_prompt(query: str, pattern_quotes: List[dict]) -> str:
    theme_prompt = f"""
    You are a biomedical research expert analyzing scientific papers on: {query}

    Reference phrases and corresponding extraction patterns:
    {json.dumps(pattern_quotes, indent=2)}

    Task: Generate 10-20 discrete scientific themes from the reference material. Each theme should incorporate up to 10 relevant patterns. Use EXACT pattern names provided.

Example Response Format:
{{
    "themes": [
        {{
            "theme": "JAK-STAT Signaling in Disease Progression",
            "relevant_patterns": [
                "jak_stat_activation_pattern",
                "cytokine_signaling_pattern",
                "pathway_inhibition_pattern",
                "cellular_response_pattern",
                "phosphorylation_pattern"
            ]
        }},
        {{
            "theme": "Immune Cell Dysfunction Mechanisms",
            "relevant_patterns": [
                "tcell_response_pattern",
                "cytokine_production_pattern",
                "immune_regulation_pattern",
                "inflammation_pattern"
            ]
        }},
        ...
    ]
}}

Theme Development Guidelines:

1. Scientific Depth Categories:
   - Molecular Mechanisms (e.g., signaling cascades, protein interactions)
   - Cellular Processes (e.g., cell death, proliferation)
   - Genetic Regulation (e.g., transcription, epigenetics)
   - Immune Responses (e.g., inflammation, adaptive immunity)
   - Disease Mechanisms (e.g., pathogenesis, progression)
   - Therapeutic Approaches (e.g., drug mechanisms, resistance)
   - Clinical Manifestations (e.g., symptoms, complications)
   - Treatment Outcomes (e.g., efficacy, adverse effects)
   - Biomarker Studies (e.g., diagnostic, prognostic)
   - Drug Development (e.g., screening, optimization)
   - Therapeutic Resistance (e.g., mechanisms, overcome strategies)
   - Patient Response Patterns (e.g., subgroup analysis)
   - Technology Applications (e.g., new methods, tools)
   - Systems Biology Approaches (e.g., network analysis)

2. Theme Construction Rules:
   - Each theme should focus on a specific scientific aspect
   - Include up to 10 most relevant patterns per theme
   - Themes should be scientifically distinct
   - Use precise biomedical terminology
   - Title should reflect specific research focus
   - Description should capture theme's scope
   - Cover different research aspects (basic/translational/clinical)

3. Pattern Distribution Requirements:
   - Use EXACT pattern name from provided list corresponding to the theme
   - Patterns should be scientifically coherent within theme

Return a JSON object with themes array exactly matching the format above. 
Generate as many relevant themes as the patterns support (5-20 themes),
ensuring each theme represents a distinct and significant aspect of the research.
"""
    return theme_prompt


def generate_section_prompt(query: str, theme: dict, quotes_by_pattern: dict) -> str:
    section_prompt = f"""
Analyzing research on: {query}

Theme: {theme["theme"]}
Description: {theme.get("description", "")}

Evidence organized by pattern:
{json.dumps(quotes_by_pattern, indent=2)}

Task: Write a coherent section that synthesizes these findings.

Requirements:
1. Use evidence from ALL provided patterns
2. Cite specific findings from the quotes
3. Compare and contrast different findings
4. Use academic, objective language
5. Length: 200-300 words
6. Make explicit connections between different patterns

Return format:
{{
    "content": "Section text here..."
}}
"""
    return section_prompt


def generate_revise_search_query_for_pubmed_prompt(query: str) -> str:
    query_prompt = f"""
Analyzing the search query/statement: {query}

Task: Transform this into an effective PubMed search strategy by:
1. Identifying the key concepts and research intent
2. Determining appropriate search fields for each concept (e.g. [Title/Abstract], [MeSH Terms])
3. Structuring with proper Boolean operators and wildcards where appropriate
4. Adding relevant MeSH terms to enhance retrieval and exploring hierarchical terms when beneficial
5. Including relevant subheadings and qualifiers
6. Considering both specificity and sensitivity

Example inputs and outputs:

Input: "Effectiveness of SGLT2 inhibitors in reducing cardiovascular events in type 2 diabetes patients"
Output:
```json
{{
    "query": "(sodium-glucose transporter 2 inhibitors[MeSH Terms] OR SGLT2 inhibitor*[Title/Abstract] OR sodium glucose cotransporter 2 inhibitor*[Title/Abstract]) AND (cardiovascular diseases[MeSH Terms] OR cardiovascular[Title/Abstract] OR cardiac[Title/Abstract]) AND (diabetes mellitus, type 2[MeSH Terms] OR type 2 diabetes[Title/Abstract] OR T2DM[Title/Abstract]) AND (treatment outcome[MeSH Terms] OR effectiveness[Title/Abstract] OR efficacy[Title/Abstract])",
    "key_concepts": [
        "SGLT2 inhibitors",
        "cardiovascular events",
        "type 2 diabetes",
        "treatment effectiveness"
    ],
    "research_intent": "To evaluate cardiovascular outcomes in T2DM patients treated with SGLT2 inhibitors",
    "suggested_fields": [
        "Title/Abstract",
        "MeSH Terms"
    ],
    "mesh_terms_added": [
        "sodium-glucose transporter 2 inhibitors",
        "cardiovascular diseases",
        "diabetes mellitus, type 2",
        "treatment outcome"
    ]
}}
```

Input: "Impact of immunotherapy combined with chemotherapy on survival in advanced non-small cell lung cancer"
Output:
```json
{{
    "query": "(immunotherapy[MeSH Terms] OR immune checkpoint inhibitors[MeSH Terms] OR immunotherap*[Title/Abstract]) AND (drug therapy, combination[MeSH Terms] OR combined modalit*[Title/Abstract]) AND (carcinoma, non-small-cell lung[MeSH Terms] OR NSCLC[Title/Abstract]) AND (survival[MeSH Terms] OR survival rate[MeSH Terms] OR survival analysis[MeSH Terms] OR survival[Title/Abstract] OR prognos*[Title/Abstract])",
    "key_concepts": [
        "immunotherapy",
        "combination therapy",
        "advanced NSCLC",
        "survival outcomes"
    ],
    "research_intent": "To assess survival outcomes of combined immunotherapy-chemotherapy in advanced NSCLC",
    "suggested_fields": [
        "Title/Abstract", 
        "MeSH Terms"
    ],
    "mesh_terms_added": [
        "immunotherapy",
        "immune checkpoint inhibitors", 
        "drug therapy, combination",
        "carcinoma, non-small-cell lung",
        "survival",
        "survival rate"
    ]
}}
```

Analyze the input query and return a similar JSON structure with a comprehensive PubMed search strategy that maximizes both precision and recall.
"""
    return query_prompt

def generate_revise_search_query_for_crossref_prompt(query: str) -> str:
    query_prompt = f"""
Analyzing the search query/statement: {query}

Task: Transform this into an effective Crossref search strategy by:
1. Identifying the key concepts and research intent
2. Structuring with proper Boolean operators (AND, OR)
3. Including relevant synonyms and alternative terms
4. Balancing specificity and sensitivity
5. Considering both title and abstract searching
6. Formatting for optimal retrieval in academic literature

Example inputs and outputs:

Input: "Effectiveness of SGLT2 inhibitors in reducing cardiovascular events in type 2 diabetes patients"
```json
{{
    "query": "(SGLT2 inhibitors OR sodium glucose cotransporter 2 inhibitors OR dapagliflozin OR empagliflozin OR canagliflozin) AND (cardiovascular OR cardiac OR heart) AND (type 2 diabetes OR T2DM OR diabetes mellitus type 2) AND (effectiveness OR efficacy OR outcomes)",
    "key_concepts": [
        "SGLT2 inhibitors",
        "cardiovascular events",
        "type 2 diabetes",
        "treatment effectiveness"
    ],
    "research_intent": "To evaluate cardiovascular outcomes in T2DM patients treated with SGLT2 inhibitors",
    "search_fields": [
        "title",
        "abstract"
    ],
    "alternative_terms": {{
        "SGLT2 inhibitors": ["dapagliflozin", "empagliflozin", "canagliflozin"],
        "cardiovascular": ["cardiac", "heart"],
        "type 2 diabetes": ["T2DM", "diabetes mellitus type 2"]
    }}
}}
```

Input: "Impact of immunotherapy combined with chemotherapy on survival in advanced non-small cell lung cancer"
```json
{{
    "query": "(immunotherapy OR immune checkpoint inhibitors OR immunotherapeutic) AND (chemotherapy OR drug therapy OR combined therapy) AND (non-small cell lung cancer OR NSCLC OR lung carcinoma) AND (survival OR prognosis OR mortality)",
    "key_concepts": [
        "immunotherapy",
        "combination therapy",
        "advanced NSCLC",
        "survival outcomes"
    ],
    "research_intent": "To assess survival outcomes of combined immunotherapy-chemotherapy in advanced NSCLC",
    "search_fields": [
        "title",
        "abstract"
    ],
    "alternative_terms": {{
        "immunotherapy": ["immune checkpoint inhibitors", "immunotherapeutic"],
        "chemotherapy": ["drug therapy", "combined therapy"],
        "NSCLC": ["non-small cell lung cancer", "lung carcinoma"],
        "survival": ["prognosis", "mortality"]
    }}
}}
```

Analyze the input query and return a similar JSON structure with a comprehensive Crossref search strategy that maximizes both precision and recall.
"""
    return query_prompt

def generate_intro_conclusion_prompt(query: str, sections: List[dict]) -> str:
    intro_conclusion_prompt = f"""
Topic: {query}

Generated sections:
{json.dumps([{
    "theme": s.theme,
    "content": s.content,
    "num_papers_cited": len(s.papers_cited)
} for s in sections], indent=2)}

Task: Generate an introduction and conclusion for this literature review.

Example Response:
{{
    "introduction": "This literature review examines... The analysis covers themes including...",
    "conclusion": "The reviewed literature reveals... Future research should..."
}}

Requirements:
1. Introduction (150-200 words):
   - Introduce the topic and its significance
   - Preview the main themes ({len(sections)} identified)
   - State the purpose of the review
2. Conclusion (150-200 words):
   - Synthesize key findings across themes
   - Identify research gaps
   - Suggest future research directions

Return a JSON object with introduction and conclusion fields.
"""
    return intro_conclusion_prompt


def generate_rating_prompt(batch: List[Dict[str, Any]], query: str) -> str:
    """
    Generates the prompt for Gemini to rate a batch of papers.

    Args:
        batch (List[Dict[str, Any]]): A batch of papers.
        query (str): The original search query.

    Returns:
        str: The formatted prompt.
    """
    formatted_papers = []
    for paper in batch:
        doi = paper.get('doi', 'N/A')
        pub_date = paper.get('publication_date', '')
        formatted_date = format_date(pub_date, doi) if pub_date else '1970-01-01'
        formatted_papers.append({
            "doi": doi,
            "title": paper.get('title', 'No Title'),
            "abstract": paper.get('abstract', 'No Abstract'),
            "journal": paper.get('journal', 'No Journal'),
            "publication_date": formatted_date
        })

    prompt = f"""
You are an expert bibliometrician tasked with evaluating the relevance of scientific papers to a specific research query. Below are details of up to 10 papers. For each paper, provide a composite score assessing its relevance based on the following criteria:

1. **Publication Date:** More recent publications are generally more relevant.
2. **Journal Impact Factor:** Dynamically assess the journal's impact factor to indicate its quality and relevance.
3. **Abstract Relevance:** Direct relevance based on the abstract content.
4. **Additional Factors:** Consider other relevant aspects that may affect the paper's relevance.

**Instructions:**

- For each paper, assign a **Composite Score** between 0 and 100, where 100 indicates the highest relevance.
- Provide a brief **Summary of Relevance** explaining the reasoning behind the score. Do not mention internal journal impact factors in the summary.
- Assign a **Journal Impact Factor Score** between 0 and 30, where higher scores indicate higher impact factors.
- Ensure the **Publication Date** is formatted as YYYY-MM-DD.

**Return the results in the following JSON format:**

```json
[
    {{
        "doi": "10.1000/exampledoi1",
        "composite_score": 85,
        "relevance_summary": "Highly relevant due to recent publication and strong alignment with the research query.",
        "journal_impact_factor": 25,
        "formatted_date": "2023-05-20"
    }},
    {{
        "doi": "10.1000/exampledoi2",
        "composite_score": 70,
        "relevance_summary": "Moderately relevant with good journal standing but less direct alignment with the research query.",
        "journal_impact_factor": 18,
        "formatted_date": "2022-11-15"
    }}
    // Additional papers...
]
```

**Papers:**
{json.dumps(formatted_papers, indent=2)}
    """
    return prompt

def generate_process_single_abstract_prompt(paper: SearchResult) -> str:
    prompt = f"""
            You are an expert in natural language processing and research paper analysis.

            Analyze this research paper abstract and identify recurring linguistic patterns 
            that could indicate important research concepts, methods, or findings.

            Abstract to analyze:
            Title: {paper.title}
            Abstract: {paper.abstract}

            Instructions:
            1. Identify all key linguistic patterns that appear in the provided abstract
            2. Each pattern should capture meaningful research concepts, methods or findings
            3. Format each pattern as a sequence of tokens with their required properties
            4. Focus on novel patterns not covered by common research terminology

            **Important:** Ensure that each `pattern_name` is unique and descriptive. Use a naming convention such as `pattern_<unique_identifier>`.

            Return ONLY a valid JSON array of pattern objects in this exact format:
            [
                {{
                    "pattern_unique_identifier_1": [
                        {{"LOWER": "required_lowercase_word"}},
                        {{"LEMMA": "required_lemma_form"}},
                        {{"POS": "required_part_of_speech"}}
                    ]
                }},
                {{
                    "pattern_unique_identifier_2": [
                        {{"LOWER": "another_word"}},
                        {{"LEMMA": "another_lemma"}},
                        {{"POS": "ADJ"}}
                    ]
                }},
                ...
            ]

            The JSON must be parseable and each pattern must follow spaCy matcher syntax.
            Do not include any explanation text, only return the JSON array.
            """
    return prompt

