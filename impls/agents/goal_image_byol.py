"""
Goal-image BC with a training-only BYOL-gamma visual auxiliary looss.

The actor preserves ordinary visual GCBC early fusion: 
then encod the current observation and the fixed endpoint goal after concatenating them on
the channel axis.
"""

from __future__ import annotations

import copy

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.byol import BYOLAgent, _REFERENCE_BYOL, get_config as get_byol_config
from agents.gcbc import GCBCAgent, get_config as get_gcbc_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCDiscreteActor
from utils.byol_late import sample_weights, weight_metrics, weighted_bdino


GCPredValue = _REFERENCE_BYOL.GCPredValue
PolicyRepr = _REFERENCE_BYOL.PolicyRepr


class GoalImageBYOLAgent(BYOLAgent):
    """
    GCBC early-fusion encoder also receives BYOL gradient"""

    def _pair_representation(self, observations, actor_goals, params=None, target=False):
        if actor_goals is None:
            raise ValueError('GoalImageBYOLAgent requires a goal image.')
        inputs = jnp.concatenate([observations, actor_goals], axis=-1)
        module_name = 'target_encoder' if target else 'encoder'
        return self.network.select(module_name)(inputs, params=params)

    def pred_loss(
        self,
        observations,
        future_observations,
        actor_goals,
        actions,
        grad_params,
        module_name='value',
        use_backwards=False,
        sample_weights=None,
    ):
        """Predict a future pair representation with the endpoint goal fixed."""
        target_features = self._pair_representation(
            future_observations,
            actor_goals,
            params=grad_params,
        )
        current_features = self._pair_representation(
            observations,
            actor_goals,
            params=grad_params,
        )
        phi, psi, _ = self.network.select(module_name)(
            target_features,
            current_features,
            params=grad_params,
            use_backwards=use_backwards,
            actions=actions,
        )

        if self.config['target']:
            target_features = self._pair_representation(
                future_observations,
                actor_goals,
                params=grad_params,
                target=True,
            )
            current_features = self._pair_representation(
                observations,
                actor_goals,
                params=grad_params,
                target=True,
            )
            phi, _, _ = self.network.select('target_value')(
                target_features,
                current_features,
                params=grad_params,
                use_backwards=use_backwards,
            )

        phi = jax.lax.stop_gradient(phi)
        if phi.ndim == 2:
            phi = phi[None, ...]
            psi = psi[None, ...]
        if sample_weights is None:
            pred_loss, pred_stats = self.compute_pred_loss(
                psi, phi, loss_type=self.config['pred_loss_type'],
            )
        else:
            if self.config['pred_loss_type'] != 'bdino':
                raise ValueError('Late BYOL currently supports bdino only.')
            pred_loss = weighted_bdino(psi, phi, sample_weights)
            pred_stats = {}
        return pred_loss, {'pred_loss': pred_loss}, pred_stats

    def target_update(self, network, module_name):
        """Update both the auxiliary projector and shared pair encoder target"""
        super().target_update(network, module_name)
        if module_name == 'value':
            super().target_update(network, 'encoder')

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute ordinary goal-image BC through the shared pair encoder."""
        representations = self._pair_representation(
            batch['observations'],
            batch['actor_goals'],
            params=grad_params,
        )
        dist = self.network.select('actor')(representations, params=grad_params)
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
    def total_loss(self, batch, grad_params, rng=None):
        """Combine matched GCBC and forward/backward BYOL loss"""
        del rng
        info = {}
        weights = sample_weights(batch, self.config)
        if weights is not None:
            info.update(weight_metrics(weights))
        if self.config['pred_both']:
            forward_loss, forward_info, _ = self.pred_loss(
                batch['observations'],
                batch['value_goals'],
                batch['actor_goals'],
                batch['actions'],
                grad_params,
                sample_weights=weights,
            )
            backward_loss, backward_info, _ = self.pred_loss(
                batch['value_goals'],
                batch['observations'],
                batch['actor_goals'],
                batch['actions'],
                grad_params,
                use_backwards=True,
                sample_weights=weights,
            )
            info.update({f'pred_f/{key}': value for key, value in forward_info.items()})
            info.update({f'pred_b/{key}': value for key, value in backward_info.items()})
            prediction_loss = forward_loss + backward_loss
        elif self.config['pred_backwards']:
            prediction_loss, prediction_info, _ = self.pred_loss(
                batch['value_goals'],
                batch['observations'],
                batch['actor_goals'],
                batch['actions'],
                grad_params,
                use_backwards=True,
                sample_weights=weights,
            )
            info.update({f'pred_b/{key}': value for key, value in prediction_info.items()})
        else:
            prediction_loss, prediction_info, _ = self.pred_loss(
                batch['observations'],
                batch['value_goals'],
                batch['actor_goals'],
                batch['actions'],
                grad_params,
                sample_weights=weights,
            )
            info.update({f'pred_f/{key}': value for key, value in prediction_info.items()})

        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        info.update({f'actor/{key}': value for key, value in actor_info.items()})
        info['stats'] = {}
        total = (
            self.config['bc_weight'] * actor_loss
            + self.config['alignment'] * prediction_loss
        )
        return total, info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample from the goal-image actor and no future image is accepted."""
        representations = self._pair_representation(observations, goals)
        representations = jax.lax.stop_gradient(representations)
        dist = self.network.select('actor')(
            representations,
            temperature=temperature,
        )
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create the early fusion GCBC actor and auxiliary representation."""
        if config['encoder'] is None:
            raise ValueError('GoalImageBYOLAgent requires a visual encoder.')
        if config['policy_repr'] != 'phi__phi':
            raise ValueError(
                "GoalImageBYOLAgent retains policy_repr='phi__phi' only for "
                'the pinned BYOL checkpoint schema; the actor uses the direct '
                'early-fusion encoder feature.'
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
            feature_dim if config['use_obs_latent_dim'] else config['value_latent_dim']
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
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
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
            actor=(ex_features,),
        )['params']

        # init the behavior-cloning path exactly like GCBC
        gcbc_config = get_gcbc_config()
        for key in (
            'lr',
            'batch_size',
            'actor_hidden_dims',
            'const_std',
            'discrete',
            'encoder',
            'frame_stack',
        ):
            gcbc_config[key] = config[key]
        ordinary_gcbc = GCBCAgent.create(
            seed,
            ex_observations,
            ex_actions,
            gcbc_config,
        )
        ordinary_actor = ordinary_gcbc.network.params['modules_actor']
        ordinary_encoder = ordinary_actor['gc_encoder']['concat_encoder']
        ordinary_actor_without_encoder = {
            key: value
            for key, value in ordinary_actor.items()
            if key != 'gc_encoder'
        }
        if jax.tree_util.tree_structure(
            network_params['modules_encoder']
        ) != jax.tree_util.tree_structure(ordinary_encoder):
            raise ValueError(
                'Goal-image BYOL encoder does not match ordinary GCBC.'
            )
        if jax.tree_util.tree_structure(
            network_params['modules_actor']
        ) != jax.tree_util.tree_structure(ordinary_actor_without_encoder):
            raise ValueError(
                'Goal-image BYOL actor does not match ordinary GCBC.'
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
    """Return the pinned BYOL settings with goal-image conditioning metadata."""
    config = get_byol_config()
    config.atomic_train_manifest_sha256 = ''
    config.atomic_val_manifest_sha256 = ''
    config.update(
        ml_collections.ConfigDict(
            dict(
                agent_name='goal_image_byol_gamma',
                dataset_class='AtomicGoalImageBYOLDataset',
                policy_conditioning='goal_image',
                byol_late_fraction=0.0,
            )
        )
    )
    return config
