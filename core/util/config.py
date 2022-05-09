import os
import json

current_file_path = os.path.dirname(__file__)


def load_config():
    with open(os.path.join(current_file_path, '../../ml/config.json')) as f:
        config = json.load(f)

    return config
