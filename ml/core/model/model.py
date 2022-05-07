import os
import uuid

import pandas as pd
from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score

from ml.core.data.dataset import Dataset
from ml.core.model.architecture import ArchitectureFactory
from ml.core.model.optimizer import Optimizer
from ml.tensorflow.ext.time_history_callback import TimeHistory

current_file_path = os.path.dirname(__file__)


class Model:
    '''
    ex.
    model = Model.from_config(config)
    model.proceed()
    '''

    def __init__(self, ID, dataset, architecture, input_shape, loss, optimizer, batch_size, epochs):
        self.id = ID
        self.UUId = str(uuid.uuid4())
        self.dataset = dataset
        self.dataset.load_processed()
        self.architecture = architecture
        self.input_shape = input_shape
        self.model = ArchitectureFactory.create(architecture, input_shape, self.dataset.embedding)
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

    @staticmethod
    def from_config(config):
        return Model(
            config['id'],
            Dataset.from_config(config['dataset']),
            config['architecture'],
            config['input_shape'],
            config['loss'],
            Optimizer(config['optimizer'], config['learning_rate'], config['decay']),
            config['batch_size'],
            config['epochs']
        )

    def proceed(self):
        self.summary()
        self.compile()
        self.fit()
        self.save()

    def summary(self):
        self.model.summary()

        with open(os.path.join(current_file_path, '../../results/model-summary/model-' + self.UUId), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

    def compile(self):
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer.get(),
            metrics=['accuracy', Precision(), Recall(), F1Score(num_classes=1, threshold=0.5)]
        )

    def fit(self):
        self.model_dir = os.path.join(current_file_path, '../../results/models/' + self.UUId)
        os.mkdir(self.model_dir)

        filepath = os.path.join(self.model_dir, "saved-model-{epoch:02d}-{val_accuracy:.2f}.hdf5")

        self.model.fit(
            self.dataset.X_train, self.dataset.Y_train,
            batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2,
            callbacks=[
                ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max'),
                CSVLogger(os.path.join(current_file_path, '../../results/training/model-' + self.UUId + '.csv')),
                TimeHistory(
                    os.path.join(current_file_path, '../../results/training/model-times-' + self.UUId + '.json'))
            ]
        )

    def evaluate(self):
        return self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

    def save(self):
        self.model.save(self.model_dir + '/model.h5')
        results = pd.read_csv(os.path.join(current_file_path, '../../results/results.tsv'), sep='\t', header=0)

        evaluation = self.evaluate()

        result = {
            'UUID': self.UUId,
            'model_id': self.id,
            'dataset': self.dataset.id,
            'architecture': self.architecture,
            'input_shape': self.input_shape,
            'loss': self.loss,
            'optimizer': self.optimizer.optimizer,
            'learning_rate': self.optimizer.learning_rate,
            'decay': self.optimizer.decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'final_loss': evaluation[0],
            'accuracy': evaluation[1],
            'precision': evaluation[2],
            'recall': evaluation[3],
            'f1': evaluation[4]
        }

        print(result)

        results = results.append(result, ignore_index=True)

        results.to_csv(os.path.join(current_file_path, '../../results/results.tsv'), sep='\t', index=False)
