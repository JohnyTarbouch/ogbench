import ml_collections

from agents.language_byol import get_config as get_base_config


def get_config():
    config = get_base_config()
    config['alignment'] = 0.0
    config['language_variant_seed'] = ml_collections.config_dict.placeholder(int)
    return config
