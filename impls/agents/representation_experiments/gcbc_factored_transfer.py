"""
Factored GCBC with random, frozen, or warm-started GCIQL encoders
"""

from __future__ import annotations

import flax
import jax
import ml_collections
import optax

from agents.gcbc import GCBCAgent, get_config as get_base_config
from agents.representation_experiments.factored_utils import (
    load_factored_encoder_params,
    make_factored_encoder,
    resolve_seed_path,
    transfer_factored_encoder_params,
)
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCDiscreteActor


class FactoredTransferGCBCAgent(GCBCAgent):
    """GCBC policy over separate observation and goal embeddings"""

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        transfer_mode = config['encoder_transfer_mode']
        if transfer_mode not in ('random', 'frozen', 'finetune'):
            raise ValueError(
                f'encoder_transfer_mode must be random, frozen, or finetune; got {transfer_mode!r}'
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.max() + 1 if config['discrete'] else ex_actions.shape[-1]

        encoder = None
        if config['encoder'] is not None:
            encoder = make_factored_encoder(
                config['encoder'],
                frozen=transfer_mode == 'frozen',
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoder,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoder,
            )
        # Random:
        # - random state encoder
        # - random goal encoder
        # - random BC head
        network_def = ModuleDict({'actor': actor_def})
        network_params = network_def.init(
            init_rng,
            actor=(ex_observations, ex_goals),
        )['params']

        if transfer_mode != 'random':
            # Frozen:
            # GCIQL critic state encoder -> frozen
            # GCIQL critic goal encoder -: frozen
            # new GCBC policy head -> trained
            
            # finetune:
            # GCIQL critic state encoder -> trained with BC loss
            # GCIQL critic goal encoder -> trained with BC
            # new GCBC policy head -> trained with BC
            source_pattern = str(config['source_checkpoint'])
            if not source_pattern:
                raise ValueError(
                    'Set source_checkpoint to a factored GCIQL params_*.pkl path '
                    'or glob pattern for frozen/finetune transfer.'
                )
            checkpoint = resolve_seed_path(source_pattern, seed)
            config['resolved_source_checkpoint'] = checkpoint
            state_params, goal_params = load_factored_encoder_params(
                checkpoint,
                module_name=config['source_module'],
            )
            network_params = transfer_factored_encoder_params(
                network_params,
                state_params,
                goal_params,
            )
            print(
                f'Loaded {config["source_module"]} state/goal encoders from '
                f'{checkpoint}; transfer_mode={transfer_mode}'
            )
        else:
            config['resolved_source_checkpoint'] = ''
            print('Using randomly initialized factored GCBC encoders.')

        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(learning_rate=config['lr']),
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = get_base_config()
    config.update(
        ml_collections.ConfigDict(
            dict(
                agent_name='gcbc_factored_transfer',
                representation_type='factored_observation_goal',
                encoder_transfer_mode='random',
                source_checkpoint='',
                resolved_source_checkpoint='',
                source_module='critic',
            )
        )
    )
    return config
