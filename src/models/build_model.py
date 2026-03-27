import pickle
import logging
from sklearn.base import BaseEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, X_train, y_train, models: dict[str, BaseEstimator]):
        self.X_train = X_train
        self.y_train = y_train
        self.models = models

    def train(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            logger.info(f"Training {name} model")
            self.save(model, name)

    def save(self, model, name):
        with open(f'models/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved {name} model")

