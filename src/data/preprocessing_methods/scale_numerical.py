import pandas as pd
from typing import List
import logging
from sklearn.preprocessing import StandardScaler
from data.preprocessing_methods.base import BaseProcessor

logger = logging.getLogger(__name__)
class StandardScalerProcessor(BaseProcessor):
    def process(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        logger.info(f"Scaled columns: {columns}")
        return df