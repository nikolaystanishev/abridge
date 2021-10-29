import argparse
from abridge.fetch.twitter_fetcher import TwitterFetcher

from abridge.util.sh import register_postactions, sh


def setup():
    parser = argparse.ArgumentParser(description='''
    abridge:
        Control every possible operation for this project.
    ''')
    parser._action_groups.pop()

    # required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument('-bootstrap', help='Migrate Django', action='store_true')
    optional.add_argument('-start-web', help='Start backend and UI servers.', action='store_true')
    optional.add_argument('-fetch', help='Fetch data API.', metavar='PLATFORM', choices=['twitter'])

    return parser


def bootstrap():
    sh('python3 web/backend/manage.py migrate')


def start_web():
    sh('python3 web/backend/manage.py runserver &')
    sh('yarn --cwd web/ui start')


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


if __name__ == '__main__':
    parser = setup()
    register_postactions()
    args = parser.parse_args()

    if args.bootstrap:
        bootstrap()
    elif args.start_web:
        start_web()
    elif args.fetch:
        fetch(args.fetch)
