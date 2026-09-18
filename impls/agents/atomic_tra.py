"""
Separate fixed-endpoint GCBC or LCBC-15 with both TRA alignment losses.
TRA loss:
    1. align current and future visual features (temporal)
    2. align last visual features with language features (task)
"""

from __future__ import annotations

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.gcbc import GCBCAgent
from agents.goal_image_tra import symmetric_tra_loss
from agents.language_bc import LanguageBCAgent, get_config as get_language_config
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, MLP


def symmetric_task_alignment_loss(language, endpoints, temperature):
    if endpoints.ndim == 2:
        endpoints = endpoints[None, ...]
    if language.ndim != 2 or endpoints.ndim != 3 or language.shape != endpoints.shape[1:]:
        raise ValueError(
            'Task alignment requires matching [tasks, latent_dim] features; '
            f'got {language.shape} and {endpoints.shape}.'
        )
    language_norm = jnp.linalg.norm(language, axis=-1, keepdims=True)
    endpoint_norm = jnp.linalg.norm(endpoints, axis=-1, keepdims=True)

    language_scale = jnp.sqrt(jnp.maximum(jnp.sum(language ** 2, axis=-1, keepdims=True), 1e-16))
    endpoint_scale = jnp.sqrt(jnp.maximum(jnp.sum(endpoints ** 2, axis=-1, keepdims=True), 1e-16))
    language_unit = language / language_scale
    endpoint_unit = endpoints / endpoint_scale
    logits = jnp.einsum('eik,jk->ije', endpoint_unit, language_unit) / temperature
    identity = jnp.eye(logits.shape[0])
    denominator = logits.shape[0] * logits.shape[-1]
    forward = -jnp.sum(jax.nn.log_softmax(logits, axis=1) * identity[..., None]) / denominator
    backward = -jnp.sum(jax.nn.log_softmax(logits, axis=0) * identity[..., None]) / denominator
    loss = 0.5 * (forward + backward)
    logits = jnp.mean(logits, axis=-1)
    labels = jnp.arange(logits.shape[0])
    negative_count = jnp.maximum(jnp.sum(1.0 - identity), 1.0)
    positive = jnp.diag(logits).mean()
    negative = jnp.sum(logits * (1.0 - identity)) / negative_count
    return loss, {
        'loss': loss,
        'language_to_goal_accuracy': jnp.mean(jnp.argmax(logits, axis=0) == labels),
        'goal_to_language_accuracy': jnp.mean(jnp.argmax(logits, axis=1) == labels),
        'logits_pos': positive,
        'logits_neg': negative,
        'logit_margin': positive - negative,
        'language_norm': language_norm.mean(),
        'goal_norm': endpoint_norm.mean(),
    }


def _copy_matching_parameters(destination, source, name):
    if jax.tree_util.tree_structure(destination) != jax.tree_util.tree_structure(source):
        raise ValueError(f'Atomic TRA {name} does not match ordinary BC structure.')
    destination_leaves = jax.tree_util.tree_leaves(destination)
    source_leaves = jax.tree_util.tree_leaves(source)
    if any(a.shape != b.shape for a, b in zip(destination_leaves, source_leaves)):
        raise ValueError(f'Atomic TRA {name} does not match ordinary BC parameter shapes.')
    return copy.deepcopy(source)


class AtomicTRAAgent(flax.struct.PyTreeNode):
    """An unchanged image-only or language-only BC actor with shared features"""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _visual_representation(self, images, slot='state', params=None):
        if self.config['policy_conditioning'] == 'goal_image':
            neutral = jnp.full(images.shape, 127.5, dtype=jnp.float32)
            images = images.astype(jnp.float32)
            if slot == 'state':
                images = jnp.concatenate([images, neutral], axis=-1)
            elif slot == 'goal':
                images = jnp.concatenate([neutral, images], axis=-1)
            else:
                raise ValueError(f'Unknown atomic TRA image slot: {slot!r}.')
        return self.network.select('encoder')(images, params=params)

    def _language_representation(self, language, params=None):
        return self.network.select('language_encoder')(
            language.astype(jnp.float32), params=params,
        )

    def _actor_distribution(self, observations, condition, params=None, temperature=1.0):
        if condition is None:
            raise ValueError('Atomic TRA requires the configured BC conditioning input.')
        if self.config['policy_conditioning'] == 'goal_image':
            images = jnp.concatenate([observations, condition], axis=-1)
            features = self.network.select('encoder')(images, params=params)
            return self.network.select('actor')(
                features, params=params, temperature=temperature,
            )
        observation_features = self._visual_representation(observations, params=params)
        language_features = self._language_representation(condition, params=params)
        return self.network.select('actor')(
            observation_features, language_features, params=params,
            temperature=temperature,
        )

    def actor_loss(self, batch, grad_params, rng=None):
        del rng
        condition_key = (
            'actor_goals' if self.config['policy_conditioning'] == 'goal_image'
            else 'language_embeddings'
        )
        distribution = self._actor_distribution(
            batch['observations'], batch[condition_key], params=grad_params,
        )
        log_prob = distribution.log_prob(batch['actions'])
        loss = -log_prob.mean()
        info = {'actor_loss': loss, 'bc_log_prob': log_prob.mean()}
        if not self.config['discrete']:
            info.update({
                'mse': jnp.mean((distribution.mode() - batch['actions']) ** 2),
                'std': jnp.mean(distribution.scale_diag),
            })
        return loss, info

    def temporal_alignment_loss(self, batch, grad_params):
        current = self._visual_representation(batch['observations'], 'state', grad_params)
        future = self._visual_representation(batch['value_goals'], 'goal', grad_params)
        values, phi, psi = self.network.select('value')(
            current, future, info=True, params=grad_params,
        )
        nce, info = symmetric_tra_loss(phi, psi)
        regularizer = jnp.mean(phi ** 2) + jnp.mean(psi ** 2)
        loss = nce + self.config['repr_reg'] * regularizer
        info.update({
            'loss': loss,
            'nce': nce,
            'repr_reg': regularizer,
            'phi_norm': jnp.linalg.norm(phi, axis=-1).mean(),
            'psi_norm': jnp.linalg.norm(psi, axis=-1).mean(),
            'v_mean': values.mean(),
            'v_max': values.max(),
            'v_min': values.min(),
        })
        return loss, info

    def task_alignment_loss(self, batch, grad_params):
        goals = batch['task_alignment_goals']
        language = batch['task_alignment_language_embeddings']
        task_ids = batch['task_alignment_task_ids']
        expected = self.config['num_language_tasks']
        if goals.shape[0] != expected or language.shape[0] != expected or task_ids.shape != (expected,):
            raise ValueError('Atomic TRA requires one alignment endpoint per language task.')
        endpoint_features = self._visual_representation(goals, 'goal', grad_params)

        _, _, psi = self.network.select('value')(
            endpoint_features, endpoint_features, info=True, params=grad_params,
        )
        language_features = self._language_representation(language, grad_params)
        xi = self.network.select('task_language')(
            language_features, params=grad_params,
        )
        loss, info = symmetric_task_alignment_loss(
            xi, psi, self.config['task_temperature'],
        )
        regularizer = jnp.mean(psi ** 2) + jnp.mean(xi ** 2)
        info['nce'] = loss
        info['repr_reg'] = regularizer
        loss = loss + self.config['repr_reg'] * regularizer
        
        valid_ids = jnp.all(jnp.sort(task_ids) == jnp.arange(1, expected + 1))
        loss = jnp.where(valid_ids, loss, jnp.asarray(jnp.nan, dtype=loss.dtype))
        info.update({'loss': loss, 'unique_task_bank_valid': valid_ids.astype(jnp.float32)})
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        del rng
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        zero = jnp.asarray(0.0, dtype=actor_loss.dtype)
        if self.config['alignment'] != 0.0:
            temporal_loss, temporal_info = self.temporal_alignment_loss(batch, grad_params)
        else:
            temporal_loss, temporal_info = zero, {'loss': zero}
        if self.config['task_alignment'] != 0.0:
            task_loss, task_info = self.task_alignment_loss(batch, grad_params)
        else:
            task_loss, task_info = zero, {'loss': zero}
        weighted_bc = self.config['bc_weight'] * actor_loss
        weighted_temporal = self.config['alignment'] * temporal_loss
        weighted_task = self.config['task_alignment'] * task_loss
        loss = weighted_bc + weighted_temporal + weighted_task
        info = {
            'loss': loss,
            'weighted_bc': weighted_bc,
            'weighted_tra': weighted_temporal,
            'weighted_task_alignment': weighted_task,
        }
        info.update({f'actor/{key}': value for key, value in actor_info.items()})
        info.update({f'tra/{key}': value for key, value in temporal_info.items()})
        info.update({f'task_alignment/{key}': value for key, value in task_info.items()})
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        distribution = self._actor_distribution(observations, goals, temperature=temperature)
        actions = distribution.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        conditioning = config['policy_conditioning']
        if conditioning not in {'goal_image', 'language'}:
            raise ValueError('Atomic TRA supports separate goal_image and language actors only.')
        if config['encoder'] != 'drq':
            raise ValueError('Atomic TRA currently requires the established DrQ encoder.')
        if int(config['num_language_tasks']) != 15:
            raise ValueError('This Atomic TRA experiment is restricted to the 15 coarse instructions.')
        if config['task_temperature'] <= 0.0:
            raise ValueError('Task alignment temperature must be positive.')
        if config['alignment'] < 0.0 or config['task_alignment'] < 0.0:
            raise ValueError('TRA alignment weights must be nonnegative.')
        if config['tra_nce_reduction'] != 'released_b2_mean':
            raise ValueError('Atomic TRA must retain the released temporal loss reduction.')
        if config['tra_task_nce_reduction'] != 'mean_bidirectional_ce':
            raise ValueError('Atomic TRA task alignment requires the declared CLIP reduction.')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        encoder = encoder_modules[config['encoder']]()
        ex_visual = (
            jnp.concatenate([ex_observations, ex_observations], axis=-1)
            if conditioning == 'goal_image' else ex_observations
        )
        shape_variables = encoder.init(jax.random.PRNGKey(0), ex_visual)
        ex_features = encoder.apply(shape_variables, ex_visual)
        batch_shape = ex_observations.shape[:-3]
        ex_language = jnp.zeros(
            (*batch_shape, config['language_embedding_dim']), dtype=jnp.float32,
        )
        ex_language_features = jnp.zeros(
            (*batch_shape, config['language_latent_dim']), dtype=jnp.float32,
        )
        language_encoder = MLP((*config['language_hidden_dims'], config['language_latent_dim']))
        value = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'], latent_dim=config['value_latent_dim'],
            layer_norm=config['layer_norm'], ensemble=True, value_exp=False,
        )
        task_language = MLP(
            (*config['value_hidden_dims'], config['value_latent_dim']),
            layer_norm=config['layer_norm'],
        )
        action_dim = ex_actions.max() + 1 if config['discrete'] else ex_actions.shape[-1]
        if config['discrete']:
            actor = GCDiscreteActor(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim)
        else:
            actor = GCActor(
                hidden_dims=config['actor_hidden_dims'], action_dim=action_dim,
                state_dependent_std=False, const_std=config['const_std'],
            )
        network_def = ModuleDict({
            'encoder': encoder,
            'language_encoder': language_encoder,
            'value': value,
            'task_language': task_language,
            'actor': actor,
        })
        actor_args = (ex_features,) if conditioning == 'goal_image' else (ex_features, ex_language_features)
        params = network_def.init(
            init_rng, encoder=(ex_visual,), language_encoder=(ex_language,),
            value=(ex_features, ex_features), task_language=(ex_language_features,),
            actor=actor_args,
        )['params']
        reference_cls = GCBCAgent if conditioning == 'goal_image' else LanguageBCAgent
        reference = reference_cls.create(seed, ex_observations, ex_actions, config)
        reference_actor = reference.network.params['modules_actor']
        reference_encoder = reference_actor['gc_encoder']
        source_visual = reference_encoder[
            'concat_encoder' if conditioning == 'goal_image' else 'state_encoder'
        ]
        source_actor = {key: value for key, value in reference_actor.items() if key != 'gc_encoder'}
        params['modules_encoder'] = _copy_matching_parameters(
            params['modules_encoder'], source_visual, 'visual encoder',
        )
        params['modules_actor'] = _copy_matching_parameters(
            params['modules_actor'], source_actor, 'actor',
        )
        if conditioning == 'language':
            params['modules_language_encoder'] = _copy_matching_parameters(
                params['modules_language_encoder'], reference_encoder['goal_encoder'],
                'language encoder',
            )
        network = TrainState.create(network_def, params, tx=optax.adam(learning_rate=config['lr']))
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the complete launcher schema for either separate policy."""
    config = get_language_config()
    config.update(ml_collections.ConfigDict(dict(
        agent_name='atomic_tra',
        dataset_class='AtomicTRADataset',
        policy_conditioning='goal_image',
        encoder='drq',
        frame_stack=3,
        num_language_tasks=15,
        value_hidden_dims=(64, 64, 64),
        value_latent_dim=64,
        layer_norm=True,
        discount=0.99,
        alignment=20.0,
        task_alignment=1.0,
        task_temperature=0.1,
        bc_weight=1.0,
        repr_reg=1e-6,
        value_p_curgoal=0.0,
        value_p_trajgoal=1.0,
        value_p_randomgoal=0.0,
        value_geom_sample=True,
        gc_negative=False,
        tra_nce_reduction='released_b2_mean',
        tra_task_nce_reduction='mean_bidirectional_ce',
        tra_task_bank='one_endpoint_per_coarse_task',
        tra_aux_input='unconditioned_current_to_future',
        tra_actor_contract='ordinary_bc_with_shared_encoder_auxiliaries',
    )))
    return config
