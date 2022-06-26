import argparse
import os
from typing import List

from core.platform.filter import Filter
from core.platform.platform_facade import PlatformFacade
from core.util.config import load_config
from core.util.serializable import from_json
from core.util.sh import register_postactions, sh, start_web
from ml.core.data.combine import combine_datasets
from ml.core.data.data_processing import DataProcessing
from ml.core.data.dataset import Dataset
from ml.core.model.model import Model

current_file_path = os.path.dirname(__file__)


def setup():
    parser = argparse.ArgumentParser(description='''
    abridge:
        User control manager.
    ''')
    parser._action_groups.pop()

    optional = parser.add_argument_group('optional arguments')

    optional.add_argument('-start-web', help='Start server.', action='store_true')
    optional.add_argument('-fetch', help='Fetch data API.', metavar='<QUERY>')
    optional.add_argument('-save-env', help='Save conda environment to file.', metavar='<PATH>')
    optional.add_argument('-process-data', help='Process dataset defined in ./ml/config.json.', metavar='<DATASET_ID>')
    optional.add_argument('-train', help='Train model defined in ./ml/config.json.', metavar='<MODEL_ID>')
    optional.add_argument('-combine-data', help='Combine datasets.', metavar='<LEFT_DATASET_ID,RIGHT_DATASET_ID>')

    return parser


def fetch(platform):
    fetcher = PlatformFacade().create_fetcher(platform)

    print("Query: ", end='')
    request = from_json(input(), List[Filter])

    print(fetcher.fetch(request))


def save_env(platform):
    sh(['conda', 'env', 'export', '--no-builds', '--file', 'environment-' + platform + '.yml'])


def process_data(dataset_ids):
    config = load_config()

    for dataset_id in dataset_ids.split(','):
        dataset = Dataset.from_config(config['datasets'][dataset_id])
        dataset.load()
        DataProcessing(dataset).proceed()
        dataset.save()


def train(model_ids):
    config = load_config()

    for model_id in model_ids.split(','):
        config['models'][model_id]['dataset'] = config['datasets'][config['models'][model_id]['dataset']]
        model = Model.from_config(config['models'][model_id])
        model.proceed()


def combine_data(dataset_ids):
    config = load_config()
    dataset_ids = dataset_ids.split(',')

    combine_datasets(config['datasets'][dataset_ids[0]], config['datasets'][dataset_ids[1]],
                     config['datasets'][dataset_ids[2]])


if __name__ == '__main__':
    parser = setup()
    register_postactions()
    args = parser.parse_args()

    if args.start_web:
        start_web(True)
    elif args.fetch:
        fetch(args.fetch)
    elif args.save_env:
        save_env(args.save_env)
    elif args.process_data:
        process_data(args.process_data)
    elif args.train:
        train(args.train)
    elif args.combine_data:
        combine_data(args.combine_data)
