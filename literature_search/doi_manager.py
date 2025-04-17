import asyncio
import re
from dataclasses import asdict
from typing import List, Dict, Optional
from habanero import Crossref
from literature_search.model_calls import revise_search_query_for_crossref
from literature_search.models import SearchResult

class DOIManager:
    def __init__(self):
        self.crossref = Crossref()
        self.doi_list: List[Dict] = []

    async def search_crossref_async(self, query: str, max_results: int = 100, search_field: str = 'all') -> 'DOIManager':
        """
        Asynchronously search CrossRef for DOIs based on a query.
        """
        results = await self._search_crossref_sync(
            query,
            max_results,
            search_field
        )
        self.doi_list.extend([asdict(result) for result in results])
        return self

    async def _search_crossref_sync(self, query: str, max_results: int, search_field: str) -> List[SearchResult]:
        """
        Search CrossRef for DOIs based on a query.
        """
        try:
            enhanced_query = await revise_search_query_for_crossref(query)
        except:
            print("\n\nError revising query for Crossref\n\n")
            # Create a dictionary with the original query when revision fails
            enhanced_query = {"query": query}
        try:
            if search_field == 'title':
                res = self.crossref.works(query_title=enhanced_query['query'], limit=max_results)
            else:
                res = self.crossref.works(query=enhanced_query['query'], limit=max_results)
        except Exception as e:
            return []

        results = []
        for item in res['message']['items']:
            authors = []
            if 'author' in item:
                for author in item['author']:
                    given = author.get('given', '')
                    family = author.get('family', '')
                    name = ' '.join([given, family]).strip()
                    authors.append(name)
            # Extract publication date
            pub_date = None
            if 'published-print' in item and 'date-parts' in item['published-print']:
                date_parts = item['published-print']['date-parts'][0]
                pub_date = '-'.join(map(str, date_parts))
            elif 'published-online' in item and 'date-parts' in item['published-online']:
                date_parts = item['published-online']['date-parts'][0]
                pub_date = '-'.join(map(str, date_parts))
            # Extract full URL if available
            full_url = item.get('URL', None)
            search_result = SearchResult(
                title=item.get('title', [''])[0].strip(),
                authors=authors,
                abstract=self._clean_html(item.get('abstract', '')) if 'abstract' in item else None,
                doi=item.get('DOI', '').lower(),
                publication_date=pub_date,
                journal=item.get('container-title', [''])[0].strip() if item.get('container-title') else '',
                full_url=full_url,
                source='CrossRef',  
                pmcid=None,         
                pmid=None           
            )
            results.append(search_result)
        return results

    def _clean_html(self, html_text: str) -> str:
        """Utility method to remove HTML tags from abstract."""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', html_text).strip()

    def get_doi_list(self) -> List[Dict]:
        """Returns the list of DOIs with metadata."""
        return self.doi_list

    def to_dict(self) -> List[Dict]:
        """Alias for get_doi_list to facilitate method chaining."""
        return self.get_doi_list()

    def display_doi_list(self) -> 'DOIManager':
        """Prints the DOI list with metadata."""
        for idx, item in enumerate(self.doi_list, start=1):
            print(f"Result {idx}:")
            print(f"  Title: {item.get('title', '')}")
            print(f"  Authors: {', '.join(item.get('authors', []))}")
            print(f"  Journal: {item.get('journal', '')}")
            print(f"  Publication Date: {item.get('publication_date', '')}")
            print(f"  DOI: {item.get('doi', '')}")
            print(f"  Full URL: {item.get('full_url', '')}")
            print(f"  Abstract: {item.get('abstract', '')}\n")
        return self  # Allow chaining


class AdditionalMetadataAdder:
    def __init__(self, doi_list: List[Dict]):
        self.doi_list = doi_list

    def add_custom_field(self, field_name: str, default_value: Optional[str] = None) -> 'AdditionalMetadataAdder':
        for item in self.doi_list:
            item[field_name] = default_value
        return self

    def get_updated_list(self) -> List[Dict]:
        return self.doi_list