import pandas as pd
from typing import List
import logging
from sklearn.preprocessing import StandardScaler
from data.preprocessing_methods.base import BaseProcessor

logger = logging.getLogger(__name__)

class StandardScalerProcessor(BaseProcessor):
    def __init__(self, columns: List[str] = None):
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_scale = self.columns
        if cols_to_scale is None:
            # Default to numerical columns
            cols_to_scale = df.select_dtypes(include=['number', 'float64', 'int64']).columns.tolist()
            
        if not cols_to_scale:
            logger.warning("No columns to scale found.")
            return df
            
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        logger.info(f"Scaled columns: {cols_to_scale}")
        return df