import csv
import io
import os

import numpy as np
import pandas as pd

current_file_path = os.path.dirname(__file__)


class Dataset:
    '''
    ex.
    dataset = Dataset.from_config(config)
    dataset.load()
    DataProcessing(dataset).process()
    '''

    def __init__(self, ID, dataset_path, dataset_file, columns, data_column, label_column, test_ratio,
                 replace_character, max_length, sequence_padding, processing, runtime_processing):
        self.dataset_df = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        self.id = ID
        self.dataset_path = dataset_path
        self.dataset_file = dataset_file
        self.columns = columns
        self.data_column = data_column
        self.label_column = label_column
        self.test_ratio = test_ratio
        self.replace_character = replace_character
        self.max_length = max_length
        self.sequence_padding = sequence_padding
        self.processing = processing
        self.runtime_processing = runtime_processing

        self.visualization = 0

    @staticmethod
    def from_config(config):
        return Dataset(
            config['id'],
            config['dataset_path'],
            config['dataset_file'],
            config['columns'],
            config['data_column'],
            config['label_column'],
            config['test_ratio'],
            config['replace_character'],
            config['max_length'],
            config['sequence_padding'],
            config['processing'],
            config['runtime_processing'],
        )

    def load(self, dataset_df=None):
        if dataset_df is None:
            dataset_df = pd.read_csv(os.path.join(current_file_path, '../..', self.dataset_path, self.dataset_file),
                                     names=self.columns, encoding="latin-1", quoting=csv.QUOTE_NONE,
                                     sep=',' if self.dataset_file.endswith('.csv') else '\t')
        self.dataset_df = dataset_df

    def load_processed(self):
        self.X_train = np.load(
            os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'X_train.npy'))
        self.Y_train = np.load(
            os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'Y_train.npy'))
        self.X_test = np.load(
            os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'X_test.npy'))
        self.Y_test = np.load(
            os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'Y_test.npy'))

    def save(self):
        if not os.path.exists(os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id)):
            os.makedirs(os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id))
        np.save(os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'X_train.npy'),
                self.X_train)
        np.save(os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'Y_train.npy'),
                self.Y_train)
        np.save(os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'X_test.npy'),
                self.X_test)
        np.save(os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'Y_test.npy'),
                self.Y_test)

    def save_visualization(self, wordcloud, label_distribution):
        self.visualization += 1
        if not os.path.exists(
                os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'visualization',
                             str(self.visualization))):
            os.makedirs(
                os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'visualization',
                             str(self.visualization)))

        wordcloud.savefig(
            os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'visualization',
                         str(self.visualization), 'wordcloud.png'))

        label_distribution.savefig(
            os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'visualization',
                         str(self.visualization), 'label_distribution.png'))

        buff = io.StringIO()
        self.dataset_df.info(buf=buff)

        with open(os.path.join(current_file_path, '../..', self.dataset_path, 'processed', self.id, 'visualization',
                               str(self.visualization), 'dataset_df'), 'w') as f:
            f.write(buff.getvalue())
            f.write('\n')
            f.write(self.dataset_df.head().to_string())
