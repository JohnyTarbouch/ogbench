"""
Language-conditioned behavioral cloning with a byol-gamma auxiliary loss.

The policy receives only the current observasion and a language embedding. 
Future obs is used only by the byol-gamma prediction objective during training.
policy and prediction losses share the online visual encoder.  
byol projector and predictor remain auxiliary-only, so an
alignment-zero run has the same policy architecture as ordinary LCBC.
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
from agents.language_bc import get_config as get_language_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCDiscreteActor, MLP
from utils.byol_late import sample_weights, weight_metrics, weighted_bdino


# Keep the prediction architecture and policy-representation enum identical to
# the pinned self-pred-bc implementation loaded by agents.byol.
GCPredValue = _REFERENCE_BYOL.GCPredValue
PolicyRepr = _REFERENCE_BYOL.PolicyRepr


class LanguageLatentActor(nn.Module):
    """Fuse a shared BYOL observation representation with frozen language."""

    language_encoder: nn.Module
    actor: nn.Module

    @nn.compact
    def __call__(
        self,
        observation_representations,
        language_embeddings,
        temperature=1.0,
    ):
        language = self.language_encoder(language_embeddings.astype(jnp.float32))
        return self.actor(
            observation_representations,
            language,
            goal_encoded=True,
            temperature=temperature,
        )


class LanguageBYOLAgent(BYOLAgent):
    def _observation_representation(self, observations, params=None):
        return self.network.select('encoder')(observations, params=params)

    def pred_loss(
        self,
        observations,
        goals,
        actions,
        grad_params,
        module_name='value',
        use_backwards=False,
        sample_weights=None,
    ):
        """Apply the pinned Byol objective to features from the shared encoder."""
        target_features = self.network.select('encoder')(
            goals,
            params=grad_params,
        )
        current_features = self.network.select('encoder')(
            observations,
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
            target_features = self.network.select('target_encoder')(
                goals,
                params=grad_params,
            )
            current_target_features = self.network.select('target_encoder')(
                observations,
                params=grad_params,
            )
            phi, _, _ = self.network.select('target_value')(
                target_features,
                current_target_features,
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        weights = sample_weights(batch, self.config)
        if weights is None:
            return super().total_loss(batch, grad_params, rng)
        info = weight_metrics(weights)
        prediction_loss = 0.0
        directions = (
            (False, True) if self.config['pred_both']
            else (bool(self.config['pred_backwards']),)
        )
        for backwards in directions:
            first, second = ('value_goals', 'observations') if backwards else (
                'observations', 'value_goals'
            )
            loss, diagnostics, _ = self.pred_loss(
                batch[first], batch[second], batch['actions'], grad_params,
                
                use_backwards=backwards and self.config['pred_both'],
                sample_weights=weights,
            )
            prefix = 'pred_b' if backwards else 'pred_f'
            info.update({f'{prefix}/{key}': value for key, value in diagnostics.items()})
            prediction_loss = prediction_loss + loss
        rng = rng if rng is not None else self.rng
        _, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info.update({f'actor/{key}': value for key, value in actor_info.items()})
        info['stats'] = {}
        return (
            self.config['bc_weight'] * actor_loss
            + self.config['alignment'] * prediction_loss
        ), info

    def target_update(self, network, module_name):
        """EMA-update both parts of the split target representation stack."""
        super().target_update(network, module_name)
        if module_name == 'value':
            super().target_update(network, 'encoder')

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute language bc while updating the shared online encoder."""
        observations = self._observation_representation(
            batch['observations'],
            params=grad_params,
        )
        dist = self.network.select('actor')(
            observations,
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
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample from the language policy; ``goals`` is the language vector."""
        if goals is None:
            raise ValueError('LanguageBYOLAgent requires a language embedding.')
        observation_representations = self._observation_representation(observations)
        observation_representations = jax.lax.stop_gradient(
            observation_representations
        )
        dist = self.network.select('actor')(
            observation_representations,
            goals,
            temperature=temperature,
        )
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create the shared BYOL representation and language-conditioned actor."""
        if config['encoder'] is None:
            raise ValueError('LanguageBYOLAgent requires a visual observation encoder.')
        if config['policy_repr'] != 'phi__phi':
            raise ValueError(
                "LanguageBYOLAgent keeps policy_repr='phi__phi' for the pinned "
                'BYOL checkpoint schema; its LCBC actor uses the direct visual '
                'encoder feature and language.'
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        policy_repr = PolicyRepr[config['policy_repr']]

        # Instantiate the observation encoder exactly as LanguageBCAgent does.
        # BYOL's pinned GCPredValue is applied after this shared LCBC feature.
        encoder_module = encoder_modules[config['encoder']]
        shared_encoder = encoder_module()

        if ex_observations.ndim >= 4:
            batch_shape = ex_observations.shape[:-3]
        else:
            batch_shape = ex_observations.shape[:-1]

        # Shape inference only.  The real encoder parameters are initialized as
        # part of network_def below, using the experiment's init_rng.
        shape_variables = shared_encoder.init(
            jax.random.PRNGKey(0),
            ex_observations,
        )
        ex_obs_features = shared_encoder.apply(shape_variables, ex_observations)
        feature_dim = ex_obs_features.shape[-1]
        latent_dim = (
            feature_dim if config['use_obs_latent_dim'] else config['value_latent_dim']
        )
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
        actor_def = LanguageLatentActor(
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
            ex_obs_features,
            ex_obs_features,
            False,
            config['pred_both'],
            False,
            ex_actions,
        )
        network_params = network_def.init(
            init_rng,
            encoder=(ex_observations,),
            target_encoder=(ex_observations,),
            value=value_args,
            target_value=value_args,
            actor=(ex_obs_features, ex_language),
        )['params']
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
    """Return BYOL-gamma defaults plus the established LCBC contract."""
    config = get_byol_config()
    language_config = get_language_config()
    language_keys = (
        'composite_eval_order',
        'endpoint_dataset_mode',
        'endpoint_sampling',
        'endpoint_train_manifest_path',
        'endpoint_val_manifest_path',
        'atomic_train_manifest_sha256',
        'atomic_val_manifest_sha256',
        'atomic_require_language_contract',
        'atomic_outside_grid_policy',
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
                agent_name='language_byol_gamma',
                dataset_class='AtomicLanguageBYOLDataset',
                policy_conditioning='language',
                byol_late_fraction=0.0,
                language_dataset_mode='atomic_movement',
                # Preserve the typed placeholder used by absl/ml_collections;
                # copying its resolved value from LanguageBCAgent would turn it
                # into plain None and make CLI integer overrides fail.
                num_language_tasks=ml_collections.config_dict.placeholder(int),
            )
        )
    )
    return config
