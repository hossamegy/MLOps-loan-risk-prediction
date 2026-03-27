import joblib
import pandas as pd
from typing import List
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class CategoryEncoder:
    def __init__(self, columns: List[str] = None):
        self.columns = columns
        self.encoder = LabelEncoder()

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        categoricl_columns = self.columns
        if categoricl_columns is None:
            categoricl_columns = df.select_dtypes(include=['object']).columns.tolist()
            
        if not categoricl_columns:
            logger.warning("No columns to encode found.")
            return df
        df[categoricl_columns] = self.encoder.fit_transform(df[categoricl_columns])
        logger.info(f"Encoded columns: {categoricl_columns}")
        return df
    
    def save(self, file_path):
        joblib.dump(self.encoder, file_path)

    def load(self, file_path):
        self.encoder = joblib.load(file_path)