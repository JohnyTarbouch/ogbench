"""
GCIQL with independent observation and goal visual encoders.
Priviosly (early-fusion) we concatenated the observation and goal and fed them through a single encoder
observation-> state_encoder -> state_embedding
goal-> goal_encoder  -> goal_embedding

concat(state_embedding, goal_embedding) -> Q/V/actor network
"""

from __future__ import annotations

import copy

import flax
import jax
import ml_collections
import optax

from agents.gciql import GCIQLAgent, get_config as get_base_config
from agents.representation_experiments.factored_utils import make_factored_encoder
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue


class FactoredGCIQLAgent(GCIQLAgent):
    """GCIQL branches each encode observations and goals separately"""

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = ex_actions.max() + 1 if config['discrete'] else ex_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            encoders = {
                name: make_factored_encoder(config['encoder'])
                for name in ('value', 'critic', 'actor')
            }

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )
        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
            )
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
            )
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = {
            'value': (value_def, (ex_observations, ex_goals)),
            'critic': (critic_def, (ex_observations, ex_goals, ex_actions)),
            'target_critic': (copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            'actor': (actor_def, (ex_observations, ex_goals)),
        }
        network_def = ModuleDict(
            {key: value[0] 
                for key, value in network_info.items()
            }
        )
        network_args = {key: value[1] for key, value in network_info.items()}
        network_params = network_def.init(init_rng, **network_args)['params']
        params = flax.core.unfreeze(network_params)
        params['modules_target_critic'] = copy.deepcopy(params['modules_critic'])
        network = TrainState.create(
            network_def,
            params,
            tx=optax.adam(learning_rate=config['lr']),
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = get_base_config()
    config.update(
        ml_collections.ConfigDict(
            dict(
                agent_name='gciql_factored',
                representation_type='factored_observation_goal',
                representation_source='gciql_critic',
            )
        )
    )
    return config
