from sklearn.model_selection import train_test_split
import logging
from utils import load_config

logger = logging.getLogger(__name__)

class PrepareData:
    def __init__(self, df, train_size):
        self.df = df
        self.train_size = train_size

    def prepare(self):
        config = load_config('config/params.yaml')

        x = self.df.drop(config['data']['target'], axis=1)
        y = self.df[config['data']['target']]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.train_size, random_state=42, stratify=y
        )
        logger.info(f"X_train shape: {x_train.shape}")
        logger.info(f"X_test shape: {x_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_test shape: {y_test.shape}")
        logger.info(f"y_train value counts: {y_train.value_counts()}")
        logger.info(f"y_test value counts: {y_test.value_counts()}")

        return x_train, x_test, y_train, y_test
        
        
        