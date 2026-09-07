from agents.language_byol import get_config as get_language_byol_config


def get_config():
    config = get_language_byol_config()
    config.atomic_transition_reuse_policy = 'forbid'
    config.atomic_sampling_mode = 'uniform_transition'
    config.atomic_sampling_family_ids = ()
    config.atomic_sampling_family_probabilities = ()
    config.atomic_require_goal_language_coupling = False
    return config
