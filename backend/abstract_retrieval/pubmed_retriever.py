from typing import List, Tuple
from metapub import PubMedFetcher
from backend.data_repository.models import ScientificAbstract
from backend.abstract_retrieval.interface import AbstractRetriever
from backend.abstract_retrieval.pubmed_query_simplification import simplify_pubmed_query
from config.logging_config import get_logger
import time
import re

class PubMedAbstractRetriever(AbstractRetriever):
    def __init__(self, pubmed_fetch_object: PubMedFetcher):
        self.pubmed_fetch_object = pubmed_fetch_object
        self.logger = get_logger(__name__)

    @staticmethod
    def _simplify_pubmed_query(
            query: str, input_is_medical_notes: bool,
            simplification_function: callable = simplify_pubmed_query) -> str:
        return simplification_function(query, input_is_medical_notes)

    def _fetch_article_with_retry(self, pmid, retries=3, delay=2):
        attempt = 0
        while attempt < retries:
            try:
                article = self.pubmed_fetch_object.article_by_pmid(pmid)
                return article  # Return the article if fetched successfully
            except Exception as e:
                self.logger.warning(f"Error fetching article with PMID {pmid}: {e}")
                attempt += 1
                if attempt < retries:
                    self.logger.warning(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    self.logger.warning(f"Failed to fetch article with PMID {pmid} after {retries} retries.")
                    return None

    def _get_abstract_list(
            self, query: str,
            input_is_medical_notes: bool,
            simplify_query: bool = True) -> Tuple[List[str], str]:
        """ Fetch a list of PubMed IDs for the given query. """
        query_simplified = None
        if simplify_query:
            self.logger.info(f'Trying to simplify scientist query {query}')
            query_simplified = self._simplify_pubmed_query(query, input_is_medical_notes)

            if query_simplified != query:
                self.logger.info(f'Initial query simplified to: {query_simplified}')
                query = query_simplified
            else:
                self.logger.info('Initial query is simple enough and does not need simplification.')

        self.logger.info(f'Searching abstracts for query: {query}')
        return self.pubmed_fetch_object.pmids_for_query(
                                                        query,
                                                        retmax=30,
                                                        sort='relevance',
                                                        since='2011',
                                                        until='2016'
                                                    ), query_simplified

    def _get_abstracts(self, pubmed_ids: List[str]) -> List[ScientificAbstract]:
        """ Fetch PubMed abstracts  """
        self.logger.info(f'Fetching abstract data for following pubmed_ids: {pubmed_ids}')
        scientific_abstracts = []

        for pubmed_id in pubmed_ids:
            try:
                # abstract = self.pubmed_fetch_object.article_by_pmid(pubmed_id)
                abstract = self._fetch_article_with_retry(pubmed_id)
                if abstract.abstract is None:
                    continue
                abstract_formatted = ScientificAbstract(
                    doi=abstract.doi,
                    title=abstract.title,
                    authors=', '.join(abstract.authors),
                    year=abstract.year,
                    abstract_content=abstract.abstract,
                    pmid=pubmed_id
                )
                scientific_abstracts.append(abstract_formatted)
            except Exception as e:
                self.logger.warning(f"Error fetching article with PubMed ID {pubmed_id}: {e}")

        self.logger.info(f'Total of {len(scientific_abstracts)} abstracts retrieved.')

        return scientific_abstracts

    def get_abstract_data(
            self, scientist_question: str, input_is_medical_notes: bool, simplify_query: bool = True) -> (
            Tuple)[List[ScientificAbstract], str]:
        """  Retrieve abstract list for scientist query. """
        pmids, query_simplified = self._get_abstract_list(scientist_question, simplify_query)
        abstracts = self._get_abstracts(pmids)
        return abstracts, query_simplified
