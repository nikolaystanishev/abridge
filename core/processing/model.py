import os
from abc import ABC
from typing import List

import pandas as pd
import torch
from functional import seq
from tensorflow.python.keras.models import load_model
from transformers import T5ForConditionalGeneration, T5Tokenizer

from core.platform.data import Label, Analysis, DataObject
from core.util.config import load_config
from core.util.singleton import Singleton
from ml.core.data.data_processing import DataProcessing
from ml.core.data.dataset import Dataset
from ml.core.model.model import Model

current_file_path = os.path.dirname(__file__)


class TrainedModel(ABC):

    def compute(self, data: DataObject):
        raise NotImplementedError()


class ClassificationModel(TrainedModel):
    __metaclass__ = Singleton

    def __init__(self):
        self.__config = load_config()
        self.__model = load_model(
            os.path.join(current_file_path, '../../ml/results/models', self.__config['runtime']['model'], 'model.h5'),
            custom_objects={'F1Score': Model.f1_score()})

    def compute(self, data: DataObject):
        processed_data = self.__process_data(data)

        return self.__model.predict(processed_data)

    def __process_data(self, data: DataObject):
        dataset = Dataset.from_config(self.__config['datasets'][self.__config['runtime']['dataset']])
        data = pd.DataFrame(seq(data).map(lambda d: d.text), columns=[dataset.data_column])
        dataset.load(data)
        data_processor = DataProcessing(dataset, is_runtime=True)
        data_processor.proceed()
        return dataset.X_train


class SummarizationModel(TrainedModel):
    __metaclass__ = Singleton

    def __init__(self):
        self.__model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.__tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.__device = torch.device('cpu')

    def compute(self, data: DataObject):
        processed_data = self.__process_data(data)

        summary_ids = self.__model.generate(processed_data, num_beams=4, no_repeat_ngram_size=2, min_length=30,
                                            max_length=100, early_stopping=True)

        return self.__tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def __process_data(self, data: DataObject) -> List[int]:
        preprocess_text = '</s>'.join(seq(data).map(lambda d: d.text)).strip().replace("\n", "")
        t5_prepared_text = "summarize: " + preprocess_text

        return self.__tokenizer.encode(t5_prepared_text, return_tensors="pt").to(self.__device)


class Models:

    def __init__(self, analysis: Analysis):
        self.__analysis: Analysis = analysis

    def process(self):
        self.__classify(self.__analysis.data)
        self.__summarize(self.__analysis.data)

    def __classify(self, data: DataObject):
        labels = ClassificationModel().compute(data)

        for d, l in zip(self.__analysis.data, labels):
            d.label = Label(1 if l >= 0.5 else 0)

    def __summarize(self, data: DataObject):
        self.__analysis.summary = SummarizationModel().compute(data)
