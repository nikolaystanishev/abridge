from core.platform.data import Label
from core.processing.model import ClassificationModel, SummarizationModel

classification_model = ClassificationModel()
summarization_model = SummarizationModel()


class Models:

    def __init__(self, analysis):
        self.analysis = analysis

    def process(self):
        self._classify(self.analysis.data)
        self._summarize(self.analysis.data)

    def _classify(self, data):
        labels = classification_model.compute(data)

        for d, l in zip(self.analysis.data, labels):
            d.label = Label(1 if l >= 0.5 else 0)

    def _summarize(self, data):
        self.analysis.summary = summarization_model.compute(data)
