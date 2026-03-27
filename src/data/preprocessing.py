import logging
import os
import pandas as pd

from data.load_data import CsvLoader
from data.preprocessing_methods.base import BaseProcessor
from utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipeLineProcessor:
    def __init__(self, preprocess_dict: dict[str, BaseProcessor]):
        self.preprocess_dict = preprocess_dict

    def run(self) -> pd.DataFrame:
        config = load_config('config/params.yaml')
        loader = CsvLoader(config['data']['raw_data_path'])
        df = loader.load_data()
        for name, processor in self.preprocess_dict.items():
            df = processor.process(df)
            logger.info(f"Applying {name} preprocessor")
            logger.info(f"Final shape: {df.shape}")
        
        saved_path = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
        loader.save_data(df, saved_path)
        logger.info(f"Saved processed data to {saved_path}")
        return df