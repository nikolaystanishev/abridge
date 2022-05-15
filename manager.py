import argparse
import os

from core.platform.twitter.twitter_data_fetcher import TwitterDataFetcher
from core.util.config import load_config
from core.util.sh import bootstrap, register_postactions, sh, start_web
from ml.core.data.combine import combine_datasets
from ml.core.data.data_processing import DataProcessing
from ml.core.data.dataset import Dataset
from ml.core.model.model import Model

current_file_path = os.path.dirname(__file__)


def setup():
    parser = argparse.ArgumentParser(description='''
    abridge:
        Control every possible operation for this project.
    ''')
    parser._action_groups.pop()

    # required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument('-bootstrap', help='Bootstrap applications.', action='store_true')
    optional.add_argument('-start-web', help='Start server.', action='store_true')
    optional.add_argument('-fetch', help='Fetch data API.', metavar='PLATFORM', choices=['twitter'])
    optional.add_argument('-save-env', help='Save conda environment to file.')
    optional.add_argument('-process-data', help='Process dataset defined in ./ml/config.json.')
    optional.add_argument('-train', help='Train model defined in ./ml/config.json.')
    optional.add_argument('-combine-data', help='Combine datasets.')

    return parser


def fetch(platform):
    if platform == 'twitter':
        fetcher = TwitterDataFetcher()

    request = fetcher.get_empty_request()
    for k in request.keys():
        print('Insert ' + k.literal + ': ', end='')
        value = input()
        if value != '':
            request[k].extend(value.split('|'))
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

    if args.bootstrap:
        bootstrap()
    elif args.start_web:
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
