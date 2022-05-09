import os

import pandas as pd
import torch
from functional import seq
from tensorflow.python.keras.models import load_model
from transformers import T5ForConditionalGeneration, T5Tokenizer

from core.util.config import load_config
from core.util.singleton import Singleton
from ml.core.data.data_processing import DataProcessing
from ml.core.data.dataset import Dataset
from ml.core.model.model import Model

current_file_path = os.path.dirname(__file__)


class ClassificationModel:
    __metaclass__ = Singleton

    def __init__(self):
        self.config = load_config()
        self.model = load_model(
            os.path.join(current_file_path, '../../ml/results/models', self.config['runtime']['model'], 'model.h5'),
            custom_objects={'F1Score': Model.f1_score()})

    def process_data(self, data):
        dataset = Dataset.from_config(self.config['datasets'][self.config['runtime']['dataset']])
        data = pd.DataFrame(seq(data).map(lambda d: d.text), columns=[dataset.data_column])
        dataset.load(data)
        data_processor = DataProcessing(dataset, is_runtime=True)
        data_processor.proceed()
        return dataset.X_train

    def compute(self, data):
        processed_data = self.process_data(data)

        return self.model.predict(processed_data)


class SummarizationModel:
    __metaclass__ = Singleton

    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.device = torch.device('cpu')

    def process_data(self, data):
        preprocess_text = '</s>'.join(seq(data).map(lambda d: d.text)).strip().replace("\n", "")
        t5_prepared_text = "summarize: " + preprocess_text

        return self.tokenizer.encode(t5_prepared_text, return_tensors="pt").to(self.device)

    def compute(self, data):
        processed_data = self.process_data(data)

        summary_ids = self.model.generate(processed_data, num_beams=4, no_repeat_ngram_size=2, min_length=30,
                                          max_length=100, early_stopping=True)

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
