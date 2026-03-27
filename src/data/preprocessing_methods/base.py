from abc import ABC, abstractmethod
import pandas as pd

class BaseProcessor(ABC):

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass