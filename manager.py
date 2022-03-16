import argparse

from core.fetch.twitter_fetcher import TwitterFetcher
from core.util.sh import bootstrap, register_postactions, sh, start_web


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
    optional.add_argument('-save-env', help='Save conda environment to file.', action='store_true')

    return parser


def fetch(platform):
    if platform == 'twitter':
        fetcher = TwitterFetcher()

    request = fetcher.get_empty_request()
    for k in request.keys():
        print('Insert ' + k.literal + ': ', end='')
        value = input()
        if value != '':
            request[k].extend(value.split('|'))
    print(fetcher.fetch(request))


def save_env():
    sh(['conda',  'env', 'export', '>', 'environment.yml'])


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
        save_env()