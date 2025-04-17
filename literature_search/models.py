
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Section:
    heading: str
    content: str

# Updated SearchResult dataclass with sections
@dataclass
class SearchResult:
    title: str
    authors: List[str]
    abstract: Optional[str]
    doi: Optional[str]
    publication_date: Optional[str]  # Keep as string, parse when needed
    journal: str
    source: str
    pmcid: Optional[str]
    pmid: Optional[str]
    full_url: Optional[str] = None
    sections: Optional[List[Section]] = field(default=None)

    def get_publication_date(self) -> Optional[datetime]:
        """Helper method to parse publication date."""
        if not self.publication_date:
            return None
            
        date_formats = [
            '%Y-%m-%d',
            '%Y-%m',
            '%Y-%b',
            '%Y %b',
            '%b %Y',
            '%Y'
        ]
        
        for date_format in date_formats:
            try:
                date = datetime.strptime(self.publication_date, date_format)
                if date_format in ['%Y-%m', '%Y-%b', '%Y %b', '%b %Y']:
                    date = date.replace(day=1)
                return date
            except ValueError:
                continue
                
        return None

@dataclass
class ClusterInfo:
    id: int
    keywords: List[str]
    papers: List[str]  # DOIs of papers in cluster
    centroid: List[float]

@dataclass
class TimelinePoint:
    date: datetime
    papers: List[str]  # DOIs
    cluster_id: int

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

@dataclass
class Quote:
    text: str
    section: str
    similarity_score: float
    paper_doi: str
    paper_title: str