import pandas as pd
import logging

from data.preprocessing_methods.base import BaseProcessor

logger = logging.getLogger(__name__)

class RemoveDuplicatesProcessor(BaseProcessor):
    def process(df: pd.DataFrame, columns: str):
        df = df.drop_duplicates()
        logger.info(f"drop_duplicates applied, new shape: {df.shape}")
        return df