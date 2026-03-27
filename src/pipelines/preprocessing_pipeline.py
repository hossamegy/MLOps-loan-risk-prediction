import logging

from data.preprocessing import PipeLineProcessor
from data.preprocessing_methods.remove_duplicates import RemoveDuplicatesProcessor
from data.preprocessing_methods.remove_nulls import RemoveNullDateProcessor
from data.preprocessing_methods.scale_numerical import StandardScalerProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Starting Preprocessing Pipeline")

    preprocessors = {
        "DropNulls": RemoveNullDateProcessor(),
        "DropDuplicates": RemoveDuplicatesProcessor(),
        "RemoveRepeatedWords": StandardScalerProcessor()
    }
    
    preprocessor = PipeLineProcessor(preprocessors)
    preprocessor.run()

if __name__ == "__main__":
    main()