import json
import ruamel.yaml


def parse_yaml(file='cfgs/default.yaml'):
    with open(file) as f:
        return ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)


def format_config(config, indent=2):
    return json.dumps(config, indent=indent)
