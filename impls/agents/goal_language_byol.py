"""Goalimage-and-lang BC with a training-only BYOL-gamma auxiliary.
"""

from __future__ import annotations

import copy

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.goal_image_byol import GCPredValue, GoalImageBYOLAgent, PolicyRepr
from agents.goal_image_byol import get_config as get_goal_image_byol_config
from agents.goal_language_bc import GoalLanguageBCAgent
from agents.goal_language_bc import get_config as get_goal_language_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCDiscreteActor, MLP


class GoalLanguageLatentActor(nn.Module):
    """Fuse the shared visual pair feature with frozen-language features."""

    language_encoder: nn.Module
    actor: nn.Module

    @nn.compact
    def __call__(
        self,
        visual_representations,
        language_embeddings,
        temperature=1.0,
    ):
        language = self.language_encoder(
            language_embeddings.astype(jnp.float32)
        )
        inputs = jnp.concatenate([visual_representations, language], axis=-1)
        return self.actor(inputs, temperature=temperature)


class GoalLanguageBYOLAgent(GoalImageBYOLAgent):
    """Image+Language BC whose visual pair encoder also receives BYOL gradients."""

    def actor_loss(self, batch, grad_params, rng=None):
        del rng
        visual = self._pair_representation(
            batch['observations'],
            batch['actor_goals'],
            params=grad_params,
        )
        dist = self.network.select('actor')(
            visual,
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
        visual = self._pair_representation(observations, goals)
        visual = jax.lax.stop_gradient(visual)
        dist = self.network.select('actor')(
            visual,
            language_embeddings,
            temperature=temperature,
        )
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a split BYOL stack with the exact ordinary actor initializer."""
        if config['encoder'] is None:
            raise ValueError('GoalLanguageBYOLAgent requires a visual encoder.')
        if config['policy_repr'] != 'phi__phi':
            raise ValueError(
                "GoalLanguageBYOLAgent retains policy_repr='phi__phi' only "
                'for the pinned BYOL checkpoint schema; the behavior actor '
                'uses the direct early-fusion visual feature plus language.'
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        policy_repr = PolicyRepr[config['policy_repr']]

        shared_encoder = encoder_modules[config['encoder']]()
        ex_pair = jnp.concatenate([ex_observations, ex_observations], axis=-1)
        shape_variables = shared_encoder.init(jax.random.PRNGKey(0), ex_pair)
        ex_features = shared_encoder.apply(shape_variables, ex_pair)
        feature_dim = ex_features.shape[-1]
        latent_dim = (
            feature_dim
            if config['use_obs_latent_dim']
            else config['value_latent_dim']
        )

        if ex_observations.ndim >= 4:
            batch_shape = ex_observations.shape[:-3]
        else:
            batch_shape = ex_observations.shape[:-1]
        ex_language = jnp.zeros(
            (*batch_shape, config['language_embedding_dim']),
            dtype=jnp.float32,
        )
        action_dim = (
            ex_actions.max() + 1
            if config['discrete']
            else ex_actions.shape[-1]
        )

        value_def = GCPredValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=latent_dim,
            layer_norm=config['layer_norm'],
            ensemble_size=config['ensemble_size'],
            state_encoder=None,
            goal_encoder=None,
            pred_both=config.get('pred_both', False),
            normalize_phi=config.get('normalize_phi', False),
            action_forward=config.get('action_forward', False),
        )
        language_encoder = MLP(
            (*config['language_hidden_dims'], config['language_latent_dim'])
        )
        if config['discrete']:
            base_actor = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
            )
        else:
            base_actor = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
            )
        actor_def = GoalLanguageLatentActor(
            language_encoder=language_encoder,
            actor=base_actor,
        )

        network_def = ModuleDict(
            {
                'encoder': shared_encoder,
                'target_encoder': copy.deepcopy(shared_encoder),
                'value': value_def,
                'target_value': copy.deepcopy(value_def),
                'actor': actor_def,
            }
        )
        value_args = (
            ex_features,
            ex_features,
            False,
            config['pred_both'],
            False,
            ex_actions,
        )
        network_params = network_def.init(
            init_rng,
            encoder=(ex_pair,),
            target_encoder=(ex_pair,),
            value=value_args,
            target_value=value_args,
            actor=(ex_features, ex_language),
        )['params']
        
        ordinary_config = get_goal_language_config()
        for key in (
            'lr',
            'batch_size',
            'actor_hidden_dims',
            'const_std',
            'discrete',
            'encoder',
            'frame_stack',
            'language_embedding_dim',
            'language_hidden_dims',
            'language_latent_dim',
        ):
            ordinary_config[key] = config[key]
        ordinary = GoalLanguageBCAgent.create(
            seed,
            ex_observations,
            ex_actions,
            ordinary_config,
        )
        ordinary_actor = ordinary.network.params['modules_actor']
        ordinary_encoder = ordinary_actor['visual_encoder']['concat_encoder']
        ordinary_actor_without_encoder = {
            key: value
            for key, value in ordinary_actor.items()
            if key != 'visual_encoder'
        }
        if jax.tree_util.tree_structure(
            network_params['modules_encoder']
        ) != jax.tree_util.tree_structure(ordinary_encoder):
            raise ValueError(
                'Goal+Language BYOL encoder does not match ordinary Goal+Language BC.'
            )
        if jax.tree_util.tree_structure(
            network_params['modules_actor']
        ) != jax.tree_util.tree_structure(ordinary_actor_without_encoder):
            raise ValueError(
                'Goal+Language BYOL policy head does not match ordinary Goal+Language BC.'
            )
        network_params['modules_encoder'] = copy.deepcopy(ordinary_encoder)
        network_params['modules_actor'] = copy.deepcopy(
            ordinary_actor_without_encoder
        )
        network_params['modules_target_encoder'] = copy.deepcopy(
            network_params['modules_encoder']
        )
        network_params['modules_target_value'] = copy.deepcopy(
            network_params['modules_value']
        )

        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(learning_rate=config['lr']),
        )
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            ex_actions=ex_actions,
            policy_repr=policy_repr,
        )


def get_config():
    """Return corrected BYOL settings plus the Goal+Language contract."""
    config = get_goal_image_byol_config()
    language_config = get_goal_language_config()
    language_keys = (
        'composite_eval_order',
        'endpoint_dataset_mode',
        'endpoint_sampling',
        'endpoint_train_manifest_path',
        'endpoint_val_manifest_path',
        'atomic_train_manifest_path',
        'atomic_val_manifest_path',
        'atomic_train_manifest_sha256',
        'atomic_val_manifest_sha256',
        'atomic_goal_stack_mode',
        'atomic_require_source_fingerprint',
        'atomic_transition_reuse_policy',
        'atomic_sampling_mode',
        'atomic_sampling_family_ids',
        'atomic_sampling_family_probabilities',
        'atomic_require_language_contract',
        'atomic_outside_grid_policy',
        'atomic_require_goal_language_coupling',
        'future_language_train_labels_path',
        'future_language_val_labels_path',
        'language_embedding_path',
        'language_embedding_model',
        'language_embedding_sha256',
        'language_task_spec_sha256',
        'language_embedding_dim',
        'language_min_train_retrieval_top1',
        'language_min_heldout_retrieval_top1',
        'language_hidden_dims',
        'language_latent_dim',
        'language_train_variant',
        'language_train_control',
        'language_eval_variants',
        'language_final_eval_variants',
    )
    config.update({key: language_config[key] for key in language_keys})
    config.update(
        ml_collections.ConfigDict(
            dict(
                agent_name='goal_language_byol_gamma',
                dataset_class='AtomicGoalLanguageBYOLDataset',
                policy_conditioning='goal_language',
                language_dataset_mode='atomic_movement',
                num_language_tasks=ml_collections.config_dict.placeholder(int),
            )
        )
    )
    return config
