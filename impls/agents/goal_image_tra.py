"""
Fixed endpoint GCBC with a training TRA auxiliary loss.

The actor is exactly the ordinary early-fusion GCBC actor: every action is
conditioned on the stable atomic endpoint image. 
"""

from __future__ import annotations

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.gcbc import GCBCAgent, get_config as get_gcbc_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor


def symmetric_tra_loss(phi, psi):
    """REturn the released TRA symmetric InfoNCE reduction and diagnostics."""
    if phi.ndim == 2:
        phi = phi[None, ...]
        psi = psi[None, ...]
    if phi.shape != psi.shape:
        raise ValueError(f'TRA representation shapes differ: {phi.shape} != {psi.shape}.')

    batch_size = phi.shape[1]
    logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
    identity = jnp.eye(batch_size)
    loss = -(
        jax.nn.log_softmax(logits, axis=0) * identity[..., None]
        + jax.nn.log_softmax(logits, axis=1) * identity[..., None]
    )
    
    loss = jnp.mean(loss)

    mean_logits = jnp.mean(logits, axis=-1)
    predicted = jnp.argmax(mean_logits, axis=1)
    positive = jnp.sum(mean_logits * identity) / jnp.sum(identity)
    negative_denominator = jnp.maximum(jnp.sum(1.0 - identity), 1.0)
    negative = jnp.sum(mean_logits * (1.0 - identity)) / negative_denominator
    return loss, {
        'categorical_accuracy': jnp.mean(predicted == jnp.arange(batch_size)),
        'logits_pos': positive,
        'logits_neg': negative,
        'logit_margin': positive - negative,
        'logits': mean_logits.mean(),
    }


class GoalImageTRAAgent(flax.struct.PyTreeNode):
    """Endpoint GCBC whose shared encoder also receives TRA gradients."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _pair_representation(self, observations, actor_goals, params=None):
        if actor_goals is None:
            raise ValueError('GoalImageTRAAgent requires a stable endpoint goal image.')
        pair = jnp.concatenate([observations, actor_goals], axis=-1)
        return self.network.select('encoder')(pair, params=params)

    def _slot_representation(self, images, slot, params=None):
        """
        Encode one image in the state or goal slot of GCBC's shared encoder.
        """
        neutral = jnp.full(images.shape, 127.5, dtype=jnp.float32)
        images = images.astype(jnp.float32)
        if slot == 'state':
            pair = jnp.concatenate([images, neutral], axis=-1)
        elif slot == 'goal':
            pair = jnp.concatenate([neutral, images], axis=-1)
        else:
            raise ValueError(f'Unknown TRA encoder slot: {slot!r}.')
        return self.network.select('encoder')(pair, params=params)

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute ordinary fixed-endpoint GCBC through the shared encoder."""
        del rng
        representations = self._pair_representation(
            batch['observations'],
            batch['actor_goals'],
            params=grad_params,
        )
        dist = self.network.select('actor')(representations, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        loss = -log_prob.mean()
        info = {'actor_loss': loss, 'bc_log_prob': log_prob.mean()}
        if not self.config['discrete']:
            info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )
        return loss, info

    def temporal_alignment_loss(self, batch, grad_params):
        """Align current-state and sampled-future features only.
        """
        current_features = self._slot_representation(
            batch['observations'], 'state', params=grad_params
        )
        future_features = self._slot_representation(
            batch['value_goals'], 'goal', params=grad_params
        )
        value, phi, psi = self.network.select('value')(
            current_features,
            future_features,
            info=True,
            params=grad_params,
        )
        loss, info = symmetric_tra_loss(phi, psi)
        regularizer = jnp.mean(phi**2) + jnp.mean(psi**2)
        loss = loss + self.config['repr_reg'] * regularizer
        info.update(
            {
                'loss': loss,
                'repr_reg': regularizer,
                'v_mean': value.mean(),
                'v_max': value.max(),
                'v_min': value.min(),
            }
        )
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        del rng
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        if self.config['alignment'] == 0.0:
            tra_loss = jnp.asarray(0.0, dtype=actor_loss.dtype)
            tra_info = {'loss': tra_loss}
        else:
            tra_loss, tra_info = self.temporal_alignment_loss(batch, grad_params)
        total = (
            self.config['bc_weight'] * actor_loss
            + self.config['alignment'] * tra_loss
        )
        info = {
            'loss': total,
            'weighted_bc': self.config['bc_weight'] * actor_loss,
            'weighted_tra': self.config['alignment'] * tra_loss,
        }
        info.update({f'actor/{key}': value for key, value in actor_info.items()})
        info.update({f'tra/{key}': value for key, value in tra_info.items()})
        return total, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
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
        """GCBC-identical actor plus independent TRA projection heads."""
        if config['encoder'] != 'drq':
            raise ValueError(
                "GoalImageTRAAgent's neutral-slot contract currently requires "
                "encoder='drq'."
            )
        if config['tra_aux_input'] != 'state_slot_to_future_goal_slot':
            raise ValueError(
                'Unsupported TRA auxiliary input contract: '
                f"{config['tra_aux_input']!r}."
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        shared_encoder = encoder_modules[config['encoder']]()
        ex_pair = jnp.concatenate([ex_observations, ex_observations], axis=-1)
        shape_variables = shared_encoder.init(jax.random.PRNGKey(0), ex_pair)
        ex_features = shared_encoder.apply(shape_variables, ex_pair)
        action_dim = (
            ex_actions.max() + 1
            if config['discrete']
            else ex_actions.shape[-1]
        )

        value_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['value_latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=True,
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
                'value': value_def,
                'actor': actor_def,
            }
        )
        network_params = network_def.init(
            init_rng,
            encoder=(ex_pair,),
            value=(ex_features, ex_features),
            actor=(ex_features,),
        )['params']
        
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
        if jax.tree_util.tree_structure(network_params['modules_encoder']) != (
            jax.tree_util.tree_structure(ordinary_encoder)
        ):
            raise ValueError('TRA encoder does not match ordinary GCBC.')
        if jax.tree_util.tree_structure(network_params['modules_actor']) != (
            jax.tree_util.tree_structure(ordinary_actor_without_encoder)
        ):
            raise ValueError('TRA actor does not match ordinary GCBC.')
        network_params['modules_encoder'] = copy.deepcopy(ordinary_encoder)
        network_params['modules_actor'] = copy.deepcopy(ordinary_actor_without_encoder)

        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(learning_rate=config['lr']),
        )
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    """Return fixed-endpoint GCBC plus the released TRA auxiliary defaults."""
    config = get_gcbc_config()
    config.update(
        ml_collections.ConfigDict(
            dict(
                agent_name='goal_image_tra',
                dataset_class='AtomicGoalImageBYOLDataset',
                policy_conditioning='goal_image',
                atomic_train_manifest_path='',
                atomic_val_manifest_path='',
                atomic_train_manifest_sha256='',
                atomic_val_manifest_sha256='',
                atomic_goal_stack_mode='repeat_endpoint',
                atomic_require_source_fingerprint=False,
                atomic_transition_reuse_policy='forbid',
                atomic_sampling_mode='uniform_transition',
                atomic_sampling_family_ids=(),
                atomic_sampling_family_probabilities=(),
                value_hidden_dims=(64, 64, 64),
                value_latent_dim=64,
                layer_norm=True,
                discount=0.99,
                alignment=20.0,
                bc_weight=1.0,
                repr_reg=1e-6,
                value_p_curgoal=0.0,
                value_p_trajgoal=1.0,
                value_p_randomgoal=0.0,
                value_geom_sample=True,
                gc_negative=False,
                tra_nce_reduction='released_b2_mean',
                tra_aux_input='state_slot_to_future_goal_slot',
            )
        )
    )
    return config
