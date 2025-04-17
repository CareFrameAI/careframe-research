from lxml import etree
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from Bio import Entrez
from Bio.Entrez.Parser import DictionaryElement
import aiohttp
from literature_search.model_calls import revise_search_query_for_pubmed
from literature_search.models import SearchResult, Section

class PubMedClient:
    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
    ):
        """
        Initialize the PubMedClient with user email and optional API key.

        Args:
            email (str): User email for Entrez.
            api_key (Optional[str], optional): Entrez API key. Defaults to None.
            logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
            max_retries (int, optional): Maximum number of retries for API calls. Defaults to 5.
            backoff_factor (float, optional): Factor for exponential backoff. Defaults to 1.0.
        """
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    async def search_pubmed(
        self,
        query: str,
        max_results: int = 10,
        search_field: str = 'all',
        **additional_params
    ) -> List[SearchResult]:
        """
        Asynchronously search PubMed with the given query and parameters.

        Args:
            query (str): The search query.
            max_results (int, optional): Maximum number of results to retrieve. Defaults to 10.
            search_field (str, optional): Field to search in ('all', 'title', 'abstract', etc.). Defaults to 'all'.
            **additional_params: Additional search parameters supported by PubMed.

        Returns:
            List[SearchResult]: A list of structured search results.
        """
        return await self._search_pubmed_sync(
            query,
            max_results,
            search_field,
            additional_params
        )

    async def _search_pubmed_sync(
        self,
        query: str,
        max_results: int,
        search_field: str,
        additional_params: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        Synchronously perform PubMed search and parse results.

        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to retrieve.
            search_field (str): Field to search in.
            additional_params (Dict[str, Any]): Additional search parameters.

        Returns:
            List[SearchResult]: A list of structured search results.
        """
        # Construct the search term based on the search_field
        if search_field.lower() == 'title':
            term = f'{query}[Title]'
        elif search_field.lower() == 'abstract':
            term = f'{query}[Abstract]'
        else:
            term = query  # Default to all fields

        # Incorporate additional search parameters
        for key, value in additional_params.items():
            term += f' AND {value}[{key}]'

        results = []
        attempt = 0

        try:
            loop = asyncio.get_event_loop()
            revised_query = await revise_search_query_for_pubmed(term)
        except:
            print("\n\nError revising query for PubMed\n\n")
            # Create a dictionary with the original term when revision fails
            revised_query = {"query": term}
        while attempt <= self.max_retries:
            try:
                self.logger.debug(f"Attempt {attempt+1} for PubMed search with term: {term}")
                handle = Entrez.esearch(
                    db='pubmed',
                    term=revised_query['query'],
                    retmax=max_results,
                    retmode='xml'
                )
                record = Entrez.read(handle)
                handle.close()
                id_list = record.get('IdList', [])
                if not id_list:
                    self.logger.info("No results found for the given query.")
                    return results
                self.logger.debug(f"Found {len(id_list)} articles.")
                # Fetch detailed records
                fetch_results = self._fetch_details_sync(id_list)
                results.extend(fetch_results)
                break  # Exit loop on successful fetch
            except Exception as e:
                self.logger.error(f"Error during PubMed search: {e}")
                attempt += 1
                if attempt > self.max_retries:
                    self.logger.error("Max retries exceeded for PubMed search.")
                    break
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                self.logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
        return results

    def _fetch_details_sync(self, id_list: List[str]) -> List[SearchResult]:
        """
        Synchronously fetch detailed information for a list of PubMed IDs.

        Args:
            id_list (List[str]): List of PubMed IDs.

        Returns:
            List[SearchResult]: A list of structured search results.
        """
        results = []
        batch_size = 100  # Entrez can handle up to 100 IDs per request
        for start in range(0, len(id_list), batch_size):
            batch_ids = id_list[start:start + batch_size]
            ids = ','.join(batch_ids)
            attempt = 0
            while attempt <= self.max_retries:
                try:
                    self.logger.debug(f"Fetching details for IDs: {ids}")
                    handle = Entrez.efetch(
                        db='pubmed',
                        id=ids,
                        rettype='xml',
                        retmode='xml'
                    )
                    records = Entrez.read(handle)
                    handle.close()
                    for article in records['PubmedArticle']:
                        search_result = self._parse_article(article)
                        results.append(search_result)
                    break  # Exit retry loop on success
                except Exception as e:
                    self.logger.error(f"Error fetching details: {e}")
                    attempt += 1
                    if attempt > self.max_retries:
                        self.logger.error("Max retries exceeded while fetching details.")
                        break
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        return results

    def _parse_article(self, article: DictionaryElement) -> SearchResult:
        """
        Parse a single PubMed article record into a SearchResult object.

        Args:
            article (DictionaryElement): Parsed XML article.

        Returns:
            SearchResult: Structured search result.
        """
        article_data = article['MedlineCitation']['Article']
        title = article_data.get('ArticleTitle', '').strip()
        abstract = None
        if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
            abstract_parts = article_data['Abstract']['AbstractText']
            if isinstance(abstract_parts, list):
                abstract = ' '.join(part.strip() for part in abstract_parts)
            else:
                abstract = str(abstract_parts).strip()
        authors = []
        if 'AuthorList' in article_data:
            for author in article_data['AuthorList']:
                if 'ForeName' in author and 'LastName' in author:
                    name = f"{author['ForeName']} {author['LastName']}".strip()
                    authors.append(name)
                elif 'CollectiveName' in author:
                    authors.append(author['CollectiveName'].strip())
        doi = None
        if 'ELocationID' in article_data:
            for eid in article_data['ELocationID']:
                if eid.attributes.get('EIdType') == 'doi':
                    doi = str(eid).lower()
                    break
        publication_date = None
        if 'Journal' in article_data and 'JournalIssue' in article_data['Journal']:
            journal_issue = article_data['Journal']['JournalIssue']
            pub_date_parts = []
            if 'PubDate' in journal_issue:
                pub_date = journal_issue['PubDate']
                if 'Year' in pub_date:
                    pub_date_parts.append(pub_date['Year'])
                if 'Month' in pub_date:
                    pub_date_parts.append(pub_date['Month'])
                if 'Day' in pub_date:
                    pub_date_parts.append(pub_date['Day'])
                publication_date = '-'.join(pub_date_parts)
        journal = article_data['Journal']['Title'].strip() if 'Journal' in article_data and 'Title' in article_data['Journal'] else ''
        pmcid = None
        pmid = article['MedlineCitation']['PMID']
        if 'PubmedData' in article and 'ArticleIdList' in article['PubmedData']:
            for article_id in article['PubmedData']['ArticleIdList']:
                if article_id.attributes.get('IdType') == 'pmc':
                    pmcid = str(article_id)
                    break
        return SearchResult(
            title=title,
            authors=authors,
            abstract=abstract if abstract else None,
            doi=doi,
            publication_date=publication_date if publication_date else None,
            journal=journal,
            source='PubMed',
            pmcid=pmcid,
            pmid=pmid
        )

    async def fetch_full_text(self, pmcid: str) -> Optional[str]:
        """
        Asynchronously fetch the full text XML from Europe PMC using PMCID.

        Args:
            pmcid (str): PubMed Central ID.

        Returns:
            Optional[str]: XML content of the full text if successful, else None.
        """
        url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
        attempt = 0
        while attempt <= self.max_retries:
            try:
                self.logger.debug(f"Fetching full text for PMCID: {pmcid}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            xml_content = await response.text()
                            return xml_content
                        elif response.status in {429, 500, 502, 503, 504}:
                            self.logger.warning(f"Received {response.status} status. Retrying...")
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                        else:
                            self.logger.error(f"Failed to fetch full text for PMCID: {pmcid}. Status: {response.status}")
                            return None
            except Exception as e:
                self.logger.error(f"Error fetching full text for PMCID {pmcid}: {e}")
                attempt += 1
                if attempt > self.max_retries:
                    self.logger.error("Max retries exceeded while fetching full text.")
                    return None
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                self.logger.info(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
        return None

    async def fetch_pubmed_abstracts(
        self,
        id_list: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Optional[str]]]:
        """
        Asynchronously fetch abstracts and metadata for a list of PubMed IDs.

        Args:
            id_list (List[str]): List of PubMed IDs.

        Returns:
            Tuple containing:
                - List of abstracts.
                - List of metadata dictionaries.
                - List of PMCIDs.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._fetch_pubmed_abstracts_sync,
            id_list
        )

    def _fetch_pubmed_abstracts_sync(
        self,
        id_list: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Optional[str]]]:
        """
        Synchronously fetch abstracts and metadata for a list of PubMed IDs.

        Args:
            id_list (List[str]): List of PubMed IDs.

        Returns:
            Tuple containing:
                - List of abstracts.
                - List of metadata dictionaries.
                - List of PMCIDs.
        """
        abstracts = []
        metadata_list = []
        pmcid_list = []
        batch_size = 100
        for start in range(0, len(id_list), batch_size):
            batch_ids = id_list[start:start + batch_size]
            ids = ','.join(batch_ids)
            attempt = 0
            while attempt <= self.max_retries:
                try:
                    self.logger.debug(f"Fetching abstracts for IDs: {ids}")
                    handle = Entrez.efetch(
                        db='pubmed',
                        id=ids,
                        rettype='xml',
                        retmode='xml'
                    )
                    records = Entrez.read(handle)
                    handle.close()
                    for record in records['PubmedArticle']:
                        article = record['MedlineCitation']['Article']
                        abstract_text = ''
                        if 'Abstract' in article and 'AbstractText' in article['Abstract']:
                            abstract_parts = article['Abstract']['AbstractText']
                            if isinstance(abstract_parts, list):
                                abstract_text = ' '.join(part for part in abstract_parts if isinstance(part, str))
                            else:
                                abstract_text = str(abstract_parts)
                        # Extract metadata
                        title = article.get('ArticleTitle', '')
                        authors = []
                        if 'AuthorList' in article:
                            for author in article['AuthorList']:
                                if 'LastName' in author and 'ForeName' in author:
                                    authors.append(f"{author['LastName']}, {author['ForeName']}")
                        journal = article['Journal']['Title'] if 'Journal' in article and 'Title' in article['Journal'] else ''
                        pub_date = {}
                        if 'Journal' in article and 'JournalIssue' in article['Journal']:
                            journal_issue = article['Journal']['JournalIssue']
                            if 'PubDate' in journal_issue:
                                pub_date = journal_issue['PubDate']
                        # Get PMCID if available
                        pmcid = None
                        if 'PubmedData' in record and 'ArticleIdList' in record['PubmedData']:
                            for article_id in record['PubmedData']['ArticleIdList']:
                                if article_id.attributes.get('IdType') == 'pmc':
                                    pmcid = str(article_id)
                                    break
                        pmcid_list.append(pmcid)
                        # Include pmcid in metadata
                        metadata = {
                            'title': title,
                            'authors': authors,
                            'journal': journal,
                            'pub_date': pub_date,
                            'impact_factor': None,  # Placeholder, can be integrated if data is available
                            'pmcid': pmcid
                        }
                        abstracts.append(abstract_text)
                        metadata_list.append(metadata)
                    break  # Exit retry loop on success
                except Exception as e:
                    self.logger.error(f"Error fetching abstracts: {e}")
                    attempt += 1
                    if attempt > self.max_retries:
                        self.logger.error("Max retries exceeded while fetching abstracts.")
                        break
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        self.logger.info(f"Fetched {len(abstracts)} abstracts.")
        return abstracts, metadata_list, pmcid_list

    async def fetch_full_text_sections(self, pmcid: str) -> Optional[List[Section]]:
        """
        Asynchronously fetch and parse full-text sections from Europe PMC.

        Args:
            pmcid (str): PubMed Central ID.

        Returns:
            Optional[List[Section]]: List of sections with headings and content if successful, else None.
        """
        xml_content = await self.fetch_full_text(pmcid)
        if not xml_content:
            self.logger.error(f"No XML content fetched for PMCID: {pmcid}")
            return None
        try:
            sections = self._parse_full_text_xml(xml_content)
            return sections
        except Exception as e:
            self.logger.error(f"Error parsing full text XML for PMCID {pmcid}: {e}")
            return None

    def _parse_full_text_xml(self, xml_content: str) -> List[Section]:
        """
        Parse the full-text XML and extract sections.

        Args:
            xml_content (str): Full-text XML content.

        Returns:
            List[Section]: List of sections with headings and content.
        """
        parser = etree.XMLParser(ns_clean=True, recover=True)
        try:
            root = etree.fromstring(xml_content.encode('utf-8'), parser=parser)
        except etree.XMLSyntaxError as e:
            self.logger.error(f"XML Syntax Error: {e}")
            return []

        # Attempt to extract namespaces dynamically
        namespaces = root.nsmap.copy()
        # Remove None key if present
        namespaces = {k if k is not None else 'default': v for k, v in namespaces.items()}

        # Initialize list to hold sections
        sections = []

        # Attempt to find 'sec' elements, considering possible namespaces
        # Commonly, JATS uses 'jats', but namespaces may vary
        # We'll search for any 'sec' element regardless of namespace
        sec_elements = root.findall('.//sec')
        if not sec_elements:
            # Try with wildcard namespace
            sec_elements = root.findall('.//{*}sec')

        for sec in sec_elements:
            # Extract heading
            title_elem = sec.find('title')
            if title_elem is None:
                # Try with wildcard namespace
                title_elem = sec.find('.//{*}title')
            if title_elem is not None and title_elem.text:
                heading = title_elem.text.strip()
            else:
                heading = 'No Title'

            # Extract content: concatenate all paragraphs within the section
            paragraphs = sec.findall('.//p')
            if not paragraphs:
                # Try with wildcard namespace
                paragraphs = sec.findall('.//{*}p')
            content = ' '.join(p.text.strip() for p in paragraphs if p.text)

            sections.append(Section(heading=heading, content=content))

        self.logger.debug(f"Extracted {len(sections)} sections from full-text XML.")
        return sections