"""
LCBC BYOL with an initialization matched exactly to LCBC
"""

from __future__ import annotations

import copy

import flax
import jax

from agents.language_bc import LanguageBCAgent
from agents.language_byol import LanguageBYOLAgent, get_config as get_byol_config
from utils.flax_utils import TrainState


INITIALIZATION_CONTRACT = 'ordinary_lcbc_parameter_transplant_v1'


def _copy_compatible_parameters(destination, source, name):
    if jax.tree_util.tree_structure(destination) != jax.tree_util.tree_structure(source):
        raise ValueError(f'Matched LCBC initialization has incompatible {name} structure.')
    destination_leaves = jax.tree_util.tree_leaves(destination)
    source_leaves = jax.tree_util.tree_leaves(source)
    for destination_leaf, source_leaf in zip(destination_leaves, source_leaves):
        if destination_leaf.shape != source_leaf.shape or destination_leaf.dtype != source_leaf.dtype:
            raise ValueError(f'Matched LCBC initialization has incompatible {name} shape or dtype.')
    return copy.deepcopy(source)


def transplant_ordinary_lcbc_parameters(byol_params, ordinary_actor_params):
    ordinary_actor_params = flax.core.unfreeze(ordinary_actor_params)
    required_modules = {
        'modules_encoder', 'modules_target_encoder', 'modules_actor',
        'modules_value', 'modules_target_value',
    }
    if set(byol_params) != required_modules:
        raise ValueError('Unexpected LanguageBYOL parameter modules for the declared transplant.')
    if 'gc_encoder' not in ordinary_actor_params:
        raise ValueError('Ordinary LCBC actor is missing its GCEncoder parameters.')
    ordinary_encoders = ordinary_actor_params['gc_encoder']
    if set(ordinary_encoders) != {'state_encoder', 'goal_encoder'}:
        raise ValueError('Ordinary LCBC must contain one observation encoder and one language encoder.')
    if set(byol_params['modules_actor']) != {'language_encoder', 'actor'}:
        raise ValueError('LanguageBYOL actor does not have the expected split language/action modules.')

    was_frozen = isinstance(byol_params, flax.core.FrozenDict)
    params = flax.core.unfreeze(byol_params) if was_frozen else copy.deepcopy(byol_params)
    ordinary_visual = ordinary_encoders['state_encoder']
    ordinary_language = ordinary_encoders['goal_encoder']
    ordinary_action_head = {
        key: value for key, value in ordinary_actor_params.items() if key != 'gc_encoder'
    }
    params['modules_encoder'] = _copy_compatible_parameters(
        params['modules_encoder'], ordinary_visual, 'visual encoder',
    )
    params['modules_target_encoder'] = _copy_compatible_parameters(
        params['modules_target_encoder'], params['modules_encoder'], 'target visual encoder',
    )
    params['modules_actor']['language_encoder'] = _copy_compatible_parameters(
        params['modules_actor']['language_encoder'], ordinary_language, 'language projection',
    )
    params['modules_actor']['actor'] = _copy_compatible_parameters(
        params['modules_actor']['actor'], ordinary_action_head, 'action head',
    )
    # modules_value 
    return flax.core.freeze(params) if was_frozen else params


class LanguageBYOLMatchedBCAgent(LanguageBYOLAgent):
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        contract = config.get('bc_initialization_contract', INITIALIZATION_CONTRACT)
        if contract != INITIALIZATION_CONTRACT:
            raise ValueError(f'Unsupported matched-LCBC initialization contract: {contract!r}.')
        if config.get('policy_conditioning') != 'language':
            raise ValueError('Matched LCBC BYOL requires language-only actor conditioning.')
        config['bc_initialization_contract'] = INITIALIZATION_CONTRACT
        agent = super().create(seed, ex_observations, ex_actions, config)
        ordinary = LanguageBCAgent.create(seed, ex_observations, ex_actions, config)
        params = transplant_ordinary_lcbc_parameters(
            agent.network.params, ordinary.network.params['modules_actor'],
        )
        
        network = TrainState.create(
            agent.network.model_def, params, tx=agent.network.tx,
        )
        return agent.replace(network=network, rng=ordinary.rng)


def get_config():

    config = get_byol_config()
    config.agent_name = 'language_byol_matched_bc'
    config.num_language_tasks = 135
    config.bc_initialization_contract = INITIALIZATION_CONTRACT
    return config
