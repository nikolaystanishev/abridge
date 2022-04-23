import pandas as pd
from functional import seq

from core.platform.data import Label
from core.processing.model import ClassificationModel
from ml.core.data.data_processing import DataProcessing

classification_model = ClassificationModel()


class DataProcessor:

    def __init__(self, analysis):
        self.analysis = analysis

    def process(self):
        processed_data = DataProcessing("data").process(
            pd.DataFrame(seq(self.analysis.data).map(lambda d: d.text), columns=["data"]))

        self._classify(processed_data)
        self._summarize(processed_data)

    def _classify(self, data):
        labels = classification_model.compute(data)

        for d, l in zip(self.analysis.data, labels):
            d.label = Label(l)

    def _summarize(self, data):
        self.analysis.summary = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor " \
                                "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud " \
                                "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute " \
                                "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla " \
                                "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia " \
                                "deserunt mollit anim id est laborum. "
