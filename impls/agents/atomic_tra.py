"""
Fixed-endpoint GCBC, LCBC-15 or Image+Language with TRA alignment losses.
TRA loss:
    1. align current and future visual features (temporal)
    2. align last visual features with language features (task)

An opt-in BYOL-gamma objective replaces only (1); the actor and (2) retain
their existing parameterization and initialization.
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
from agents.goal_language_bc import GoalLanguageBCAgent
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
    """BC with optional actor consumption of the aligned task feature."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _visual_representation(self, images, slot='state', params=None, target=False):
        if self.config['policy_conditioning'] in {'goal_image', 'goal_language'}:
            neutral = jnp.full(images.shape, 127.5, dtype=jnp.float32)
            images = images.astype(jnp.float32)
            if slot == 'state':
                images = jnp.concatenate([images, neutral], axis=-1)
            elif slot == 'goal':
                images = jnp.concatenate([neutral, images], axis=-1)
            else:
                raise ValueError(f'Unknown atomic TRA image slot: {slot!r}.')
        module = 'byol_target_encoder' if target else 'encoder'
        return self.network.select(module)(images, params=params)

    def _language_representation(self, language, params=None):
        return self.network.select('language_encoder')(
            language.astype(jnp.float32), params=params,
        )

    def _task_language_representation(self, language, params=None):
        """The same xi feeds task alignment and the opt-in projected actor."""
        language_features = self._language_representation(language, params=params)
        return self.network.select('task_language')(language_features, params=params)

    def _actor_distribution(
        self, observations, condition, params=None, temperature=1.0,
        language_embeddings=None,
    ):
        if condition is None:
            raise ValueError('Atomic TRA requires the configured BC conditioning input.')
        conditioning = self.config['policy_conditioning']
        if conditioning in {'goal_image', 'goal_language'}:
            images = jnp.concatenate([observations, condition], axis=-1)
            observation_features = self.network.select('encoder')(images, params=params)
        else:
            observation_features = self._visual_representation(observations, params=params)
        if conditioning == 'goal_image':
            return self.network.select('actor')(
                observation_features, params=params, temperature=temperature,
            )
        language = language_embeddings if conditioning == 'goal_language' else condition
        if language is None:
            raise ValueError('Atomic TRA goal_language requires language_embeddings as well as goals.')
        if self.config.get('actor_task_input', 'language_features') == 'task_language':
            language_features = self._task_language_representation(language, params=params)
        else:
            language_features = self._language_representation(language, params=params)
        return self.network.select('actor')(
            observation_features, language_features, params=params,
            temperature=temperature,
        )

    def actor_loss(self, batch, grad_params, rng=None):
        del rng
        condition_key = (
            'actor_goals' if self.config['policy_conditioning'] in {'goal_image', 'goal_language'}
            else 'language_embeddings'
        )
        distribution = self._actor_distribution(
            batch['observations'], batch[condition_key], params=grad_params,
            language_embeddings=(
                batch['language_embeddings']
                if self.config['policy_conditioning'] == 'goal_language' else None
            ),
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

    def byol_prediction_loss(self, batch, grad_params, use_backwards=False):
        """Pinned bdino prediction with temporal images in their original slots.

        Reverse prediction swaps source and target, including their slots: a
        future image is still encoded in the goal slot. The fixed actor endpoint
        and language never enter either direction of this auxiliary.
        """
        from agents.byol import _REFERENCE_BYOL

        source_key, source_slot = 'observations', 'state'
        target_key, target_slot = 'value_goals', 'goal'
        if use_backwards:
            source_key, target_key = target_key, source_key
            source_slot, target_slot = target_slot, source_slot
        source = self._visual_representation(batch[source_key], source_slot, grad_params)
        target = self._visual_representation(batch[target_key], target_slot, grad_params)
        _, prediction, _ = self.network.select('byol_value')(
            target, source, params=grad_params, use_backwards=use_backwards,
            actions=batch['actions'],
        )
        target = self._visual_representation(
            batch[target_key], target_slot, grad_params, target=True,
        )
        # With no actions the pinned module returns only its projector output.
        # The second input therefore cannot enter the target or prediction.
        target, _, _ = self.network.select('byol_target_value')(
            target, target, params=grad_params, use_backwards=use_backwards,
        )
        target = jax.lax.stop_gradient(target)
        loss, _ = _REFERENCE_BYOL.BYOLAgent.compute_pred_loss(
            self, prediction, target, loss_type='bdino',
        )
        return loss

    def byol_temporal_loss(self, batch, grad_params):
        forward = self.byol_prediction_loss(batch, grad_params)
        backward = self.byol_prediction_loss(batch, grad_params, use_backwards=True)
        loss = forward + backward
        return loss, {'loss': loss, 'pred_f': forward, 'pred_b': backward}

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
        xi = self._task_language_representation(language, params=grad_params)
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
        if self.config.get('temporal_objective', 'tra') == 'byol':
            if self.config['byol_alignment'] != 0.0:
                byol_loss, byol_info = self.byol_temporal_loss(batch, grad_params)
            else:
                byol_loss, byol_info = zero, {'loss': zero, 'pred_f': zero, 'pred_b': zero}
            weighted_byol = self.config['byol_alignment'] * byol_loss
            loss = loss + weighted_byol
            info.update({'loss': loss, 'weighted_byol': weighted_byol})
            info.update({f'byol/{key}': value for key, value in byol_info.items()})
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if self.config.get('temporal_objective', 'tra') == 'byol':
            # Match the pinned implementation: tau weights the PRE-update online
            # parameters, not the old target, and is deliberately 0.99.
            params = dict(network.params)
            tau = self.config['byol_target_tau']
            for online, target in (
                ('encoder', 'byol_target_encoder'),
                ('byol_value', 'byol_target_value'),
            ):
                params[f'modules_{target}'] = jax.tree_util.tree_map(
                    lambda p, tp: tau * p + (1.0 - tau) * tp,
                    self.network.params[f'modules_{online}'],
                    self.network.params[f'modules_{target}'],
                )
            network = network.replace(params=params)
        return self.replace(network=network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self, observations, goals=None, seed=None, temperature=1.0,
        language_embeddings=None,
    ):
        distribution = self._actor_distribution(
            observations, goals, temperature=temperature,
            language_embeddings=language_embeddings,
        )
        actions = distribution.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        temporal_objective = config.get('temporal_objective', 'tra')
        if temporal_objective not in {'tra', 'byol'}:
            raise ValueError("temporal_objective must be 'tra' or 'byol'.")
        if temporal_objective == 'byol':
            if config['alignment'] != 0.0:
                raise ValueError('BYOL replacement requires TRA alignment=0.')
            if not config['byol_alignment'] >= 0.0:
                raise ValueError('byol_alignment must be nonnegative.')
            if not 0.0 <= config['byol_target_tau'] <= 1.0:
                raise ValueError('byol_target_tau must be in [0, 1].')
        conditioning = config['policy_conditioning']
        if conditioning not in {'goal_image', 'language', 'goal_language'}:
            raise ValueError('Atomic TRA supports goal_image, language and goal_language actors only.')
        actor_task_input = config.get('actor_task_input', 'language_features')
        if actor_task_input not in {'language_features', 'task_language'}:
            raise ValueError("actor_task_input must be 'language_features' or 'task_language'.")
        if actor_task_input == 'task_language':
            if conditioning not in {'language', 'goal_language'}:
                raise ValueError('The projected task input requires language conditioning.')
            if config['language_latent_dim'] != 256 or config['value_latent_dim'] != 256:
                raise ValueError(
                    'The projected actor requires language_latent_dim=value_latent_dim=256 '
                    'to preserve the ordinary BC actor parameter shapes.'
                )
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
            if conditioning in {'goal_image', 'goal_language'} else ex_observations
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
        # Keep every module and its initialization identical across task-input
        # modes. The 256-dimensional guard lets both actors copy ordinary BC's
        # parameters; only the forward input selects language features or xi.
        actor_args = (ex_features,) if conditioning == 'goal_image' else (ex_features, ex_language_features)
        params = network_def.init(
            init_rng, encoder=(ex_visual,), language_encoder=(ex_language,),
            value=(ex_features, ex_features), task_language=(ex_language_features,),
            actor=actor_args,
        )['params']
        reference_cls = {
            'goal_image': GCBCAgent,
            'language': LanguageBCAgent,
            'goal_language': GoalLanguageBCAgent,
        }[conditioning]
        reference = reference_cls.create(seed, ex_observations, ex_actions, config)
        reference_actor = reference.network.params['modules_actor']
        if conditioning == 'goal_language':
            source_visual = reference_actor['visual_encoder']['concat_encoder']
            source_actor = reference_actor['actor']
            source_language = reference_actor['language_encoder']
        else:
            reference_encoder = reference_actor['gc_encoder']
            source_visual = reference_encoder[
                'concat_encoder' if conditioning == 'goal_image' else 'state_encoder'
            ]
            source_actor = {key: value for key, value in reference_actor.items() if key != 'gc_encoder'}
            if conditioning == 'language':
                source_language = reference_encoder['goal_encoder']
        params['modules_encoder'] = _copy_matching_parameters(
            params['modules_encoder'], source_visual, 'visual encoder',
        )
        params['modules_actor'] = _copy_matching_parameters(
            params['modules_actor'], source_actor, 'actor',
        )
        if conditioning in {'language', 'goal_language'}:
            params['modules_language_encoder'] = _copy_matching_parameters(
                params['modules_language_encoder'], source_language,
                'language encoder',
            )
        if temporal_objective == 'byol':
            from agents.byol import _REFERENCE_BYOL

            byol_value = _REFERENCE_BYOL.GCPredValue(
                hidden_dims=config['byol_value_hidden_dims'],
                latent_dim=config['byol_value_latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble_size=config['byol_ensemble_size'],
                state_encoder=None, goal_encoder=None,
                pred_both=True, normalize_phi=False, action_forward=True,
            )
            byol_modules = {
                'byol_value': byol_value,
                'byol_target_value': copy.deepcopy(byol_value),
                'byol_target_encoder': copy.deepcopy(encoder),
            }
            # Initialize auxiliary modules separately so adding BYOL cannot
            # change any existing actor, task-head, or encoder initialization.
            value_args = (ex_features, ex_features, False, True, False, ex_actions)
            byol_params = ModuleDict(byol_modules).init(
                init_rng, byol_value=value_args, byol_target_value=value_args,
                byol_target_encoder=(ex_visual,),
            )['params']
            params.update(byol_params)
            params['modules_byol_target_encoder'] = copy.deepcopy(params['modules_encoder'])
            params['modules_byol_target_value'] = copy.deepcopy(params['modules_byol_value'])
            network_def = ModuleDict(dict(network_def.modules, **byol_modules))
        network = TrainState.create(network_def, params, tx=optax.adam(learning_rate=config['lr']))
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the complete launcher schema for all three policy inputs."""
    config = get_language_config()
    config.update(ml_collections.ConfigDict(dict(
        agent_name='atomic_tra',
        dataset_class='AtomicTRADataset',
        policy_conditioning='goal_image',
        actor_task_input='language_features',
        tra_allow_train_paraphrases=False,
        language_variant_seed=ml_collections.config_dict.placeholder(int),
        encoder='drq',
        frame_stack=3,
        num_language_tasks=15,
        value_hidden_dims=(64, 64, 64),
        value_latent_dim=64,
        layer_norm=True,
        discount=0.99,
        temporal_objective='tra',
        alignment=20.0,
        byol_alignment=10.0,
        byol_target_tau=0.99,
        byol_value_hidden_dims=(64, 64, 64),
        byol_value_latent_dim=64,
        byol_ensemble_size=2,
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
