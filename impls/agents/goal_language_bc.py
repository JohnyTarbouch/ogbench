"""
Behavioral cloning conditioned on both a goal image and language
"""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.gcbc import GCBCAgent
from agents.language_bc import get_config as get_language_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCDiscreteActor, MLP


class GoalLanguageActor(nn.Module):
    # fuse a gcbc visual representation with language vector

    visual_encoder: nn.Module
    language_encoder: nn.Module
    actor: nn.Module

    @nn.compact
    def __call__(
        self,
        observations,
        goals,
        language_embeddings,
        temperature=1.0,
    ):
        visual = self.visual_encoder(observations, goals)
        language = self.language_encoder(language_embeddings.astype(jnp.float32))
        inputs = jnp.concatenate([visual, language], axis=-1)
        return self.actor(inputs, temperature=temperature)


class GoalLanguageBCAgent(GCBCAgent):
    # gc actor augmented with a frozen language condition

    def actor_loss(self, batch, grad_params, rng=None):
        dist = self.network.select('actor')(
            batch['observations'],
            batch['actor_goals'],
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
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )
        return actor_loss, actor_info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals,
        language_embeddings,
        seed=None,
        temperature=1.0,
    ):
        dist = self.network.select('actor')(
            observations,
            goals,
            language_embeddings,
            temperature=temperature,
        )
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        if config['encoder'] is None:
            raise ValueError('GoalLanguageBCAgent requires a visual encoder.')

        if ex_observations.ndim >= 4:
            batch_shape = ex_observations.shape[:-3]
        else:
            batch_shape = ex_observations.shape[:-1]
        ex_goals = ex_observations
        ex_language = jnp.zeros((*batch_shape, config['language_embedding_dim']), dtype=jnp.float32)
        action_dim = ex_actions.max() + 1 if config['discrete'] else ex_actions.shape[-1]

        encoder_module = encoder_modules[config['encoder']]
        visual_encoder = GCEncoder(concat_encoder=encoder_module())
        language_dims = (*config['language_hidden_dims'], config['language_latent_dim'])
        language_encoder = MLP(language_dims)
        if config['discrete']:
            actor = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
            )
        else:
            actor = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
            )
        actor_def = GoalLanguageActor(
            visual_encoder=visual_encoder,
            language_encoder=language_encoder,
            actor=actor,
        )

        network_def = ModuleDict({'actor': actor_def})
        network_params = network_def.init(
            init_rng,
            actor={
                'observations': ex_observations,
                'goals': ex_goals,
                'language_embeddings': ex_language,
            },
        )['params']
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(learning_rate=config['lr']),
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = get_language_config()
    config.update(
        dict(
            agent_name='goal_language_bc',
            dataset_class='FutureGoalImageLanguageDataset',
            policy_conditioning='goal_language',
            language_dataset_mode='standard_future_goal',
        )
    )
    return config
