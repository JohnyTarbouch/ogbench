"""
composite multi-horizon GoalImageBYOL runs.
"""

from agents.goal_image_byol import get_config as get_goal_image_byol_config


def get_config():
    config = get_goal_image_byol_config()
    config.atomic_transition_reuse_policy = 'forbid'
    config.atomic_sampling_mode = 'uniform_transition'
    config.atomic_sampling_family_ids = ()
    config.atomic_sampling_family_probabilities = ()
    return config
