import pandas as pd
import logging
from data.preprocessing_methods.base import BaseProcessor

logger = logging.getLogger(__name__)

class RemoveNullDateProcessor(BaseProcessor):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        logger.info(f"drop_nulls applied, new shape: {df.shape}")
        return df