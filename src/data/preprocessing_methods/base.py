from abc import ABC, abstractmethod
import pandas as pd

class BaseProcessor(ABC):

    @abstractmethod
    def process(df: pd.DataFrame, columns: str):
        pass