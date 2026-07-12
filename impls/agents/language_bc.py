"""
Behavioral cloning conditioned on frozen language embedding
"""

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.gcbc import GCBCAgent, get_config as get_base_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCDiscreteActor, MLP


class LanguageBCAgent(GCBCAgent):
    """
    extends the GCBCAgent class and overrides the actor_loss method to compute the loss based on 
    the log probability of the actions given the observation and language embeddings
    """
    
    def actor_loss(self, batch, grad_params, rng=None):
        # compute the log probability of the actions given the observation and language embedding
        dist = self.network.select('actor')(
            batch['observations'],
            batch['language_embeddings'],
            params=grad_params,
        )
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -log_prob.mean()
        actor_info = {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            # compute the mean squared error and standard deviation of the actions
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )
        return actor_loss, actor_info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        if config['encoder'] is None:
            raise ValueError('LanguageBCAgent requires a visual observation encoder.')

        if ex_observations.ndim >= 4:
            batch_shape = ex_observations.shape[:-3]
        else:
            batch_shape = ex_observations.shape[:-1]
        ex_language = jnp.zeros((*batch_shape, config['language_embedding_dim']), dtype=jnp.float32)
        action_dim = ex_actions.max() + 1 if config['discrete'] else ex_actions.shape[-1]

        encoder_module = encoder_modules[config['encoder']]
        language_dims = (*config['language_hidden_dims'], config['language_latent_dim'])
        actor_encoder = GCEncoder(
            state_encoder=encoder_module(),
            goal_encoder=MLP(language_dims),
        )
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=actor_encoder,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=actor_encoder,
            )

        network_def = ModuleDict({'actor': actor_def})
        network_params = network_def.init(
            init_rng,
            actor={'observations': ex_observations, 'goals': ex_language},
        )['params']
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
                agent_name='language_bc',
                dataset_class='AtomicLanguageDataset',
                policy_conditioning='language',
                language_dataset_mode='atomic_movement',
                num_language_tasks=ml_collections.config_dict.placeholder(int),
                atomic_train_manifest_path='',
                atomic_val_manifest_path='',
                future_language_train_labels_path='',
                future_language_val_labels_path='',
                language_embedding_path='',
                language_embedding_dim=512,
                language_hidden_dims=(256, 256),
                language_latent_dim=256,
                language_train_variant='canonical',
                language_eval_variants=('canonical', 'heldout'),
            )
        )
    )
    return config
