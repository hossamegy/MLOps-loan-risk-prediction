from sklearn.gaussian_process.kernels import Hyperparameter
import logging
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from src.data.load_data import CsvLoader
from src.data.validation import DataValidator
from src.features.category_encoder import CategoryEncoder
from src.models.prepare_data import PrepareData
from src.models.train import ModelTrainer
from utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "GaussianNB": GaussianNB,
    "SVC": SVC
}

class TrainPipeline():
    logger.info("Starting Training Pipeline")
    def __init__(self):
        self.config = load_config('config/params.yaml')
        self.df = CsvLoader(self.config['data']['processed_data_path']).load_data()
        self.df = DataValidator.validate_processed_data(self.df)
        self.loader = CsvLoader(self.config['data']['processed_data_path'])
        self.encoder = CategoryEncoder()
        self.df = self.encoder.encode(self.df)

        self.encoder.save(self.config['models']['encoder_path'])

        prepare_data =  PrepareData(self.df, self.config['data']['train_size'])
        self.X_train, self.X_test, self.y_train, self.y_test = prepare_data.prepare()

        self.models = {}
        for model_name, params in self.config['model'].items():
            if model_name in MODEL_MAPPING:
                model_class = MODEL_MAPPING[model_name]
                self.models[model_name] = model_class(**params)
    
    def run(self):
        trainer = ModelTrainer( self.X_train, self.y_train, self.models)

        trainer.train()