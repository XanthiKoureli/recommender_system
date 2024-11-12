from abc import ABC, abstractmethod
from typing import List
from backend.data_repository.models import ScientificAbstract


class AbstractRetriever(ABC):

    @abstractmethod
    def get_abstract_data(
            self,
            scientist_question: str,
            input_is_medical_notes: str) -> List[ScientificAbstract]:
        """ Retrieve a list of scientific abstracts based on a given query. """
        raise NotImplementedError
