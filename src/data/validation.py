import logging

import pandas as pd

from utils import load_config

logger = logging.getLogger(__name__)

class DataValidator:
    def validate_raw_data(df: pd.DataFrame):
        config = load_config("config/params.yaml")

        required_columns = [
            "Age",	
            "Income",	
            "LoanAmount",	
            "CreditScore",	
            "YearsExperience",	
            "Gender",	
            "Education",	
            "City",	
            "EmploymentType",	
            "LoanApproved",
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            if config['data']['target'] in df.columns and 'label' not in df.columns:
                logger.warning(f"Found '{config['data']['target']}' but missing 'label'. Renaming '{config['data']['target']}' to 'label'.")
                df.rename(columns={config['data']['target']: "label"}, inplace=True)
            else:
                logger.warning(f"Missing required columns: {missing}")
                raise ValueError(f"Missing required columns: {missing}")
            
        if df.isnull().sum():
            logger.warning("Found null values in columns. These will be handled by the pipeline.")
        
        logger.info("Raw data validation passed.")

        return df


    def validate_processed_data(df: pd.DataFrame):
        if df.empty:
            raise ValueError("Processed dataframe is empty!")
            
        logger.info("Processed data validation passed.")
        return df

