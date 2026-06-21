"""
To separate observation/goal encoder transfer experiments
"""

from __future__ import annotations

import glob
import pickle

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.encoders import GCEncoder, encoder_modules


def format_seed_tokens(path: str, seed: int) -> str:
    path = str(path)
    path = path.replace('%SEED3%', f'{seed:03d}')
    path = path.replace('%SEED%', str(seed))
    if '{seed' in path:
        path = path.format(seed=seed)
    return path


class FrozenFactoredGCEncoder(nn.Module):
    """Factored encoder whose representations are constants for backpropagation"""

    state_encoder: nn.Module
    goal_encoder: nn.Module

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        # GCBC cannot modify the transferred encoders
        state_rep = jax.lax.stop_gradient(self.state_encoder(observations))
        reps = [state_rep]
        if goals is not None:
            if goal_encoded:
                reps.append(goals)
            else:
                # Aply stop gradient to both representations to make the entire encoder frozen
                reps.append(jax.lax.stop_gradient(self.goal_encoder(goals)))
        return jnp.concatenate(reps, axis=-1)


def make_factored_encoder(encoder_name: str, frozen: bool = False):
    """Build independent observation and goal encoders with stable parameter names."""
    # will create:
    # - state_encoder(observation)
    # - goal_encoder(goal)
    # then concatenate both representations
    encoder_module = encoder_modules[encoder_name]
    if frozen:
        return FrozenFactoredGCEncoder(
            state_encoder=encoder_module(),
            goal_encoder=encoder_module(),
        )
    return GCEncoder(
        state_encoder=encoder_module(),
        goal_encoder=encoder_module(),
    )


def resolve_seed_path(pattern: str, seed: int) -> str:
    """Resolve a seed-formatted checkpoint pattern to exactly one file."""

    formatted = format_seed_tokens(pattern, seed)
    candidates = sorted(glob.glob(formatted))
    if len(candidates) != 1:
        raise ValueError(
            f'Expected exactly one source checkpoint for seed {seed}, found '
            f'{len(candidates)} from pattern {formatted!r}: {candidates}'
        )
    return candidates[0]


def load_factored_encoder_params(checkpoint: str, module_name: str = 'critic'):
    """Load state/goal encoder parameters from a factored GCIQL checkpoint"""
    # GCIQL critic
    # - gc_encoder
    # -- state_encoder
    # -- goal_encoder
    with open(checkpoint, 'rb') as file:
        payload = pickle.load(file)
    params = payload['agent']['network']['params']
    module_key = f'modules_{module_name}'
    try:
        encoder_params = params[module_key]['gc_encoder']
        state_params = encoder_params['state_encoder']
        goal_params = encoder_params['goal_encoder']
    except KeyError as exc:
        raise KeyError(
            f'{checkpoint} does not contain factored {module_name!r} encoder '
            f'parameters at {module_key}/gc_encoder/{{state_encoder,goal_encoder}}. '
            'Use a checkpoint trained with gciql_factored.py.'
        ) from exc
    return flax.core.unfreeze(state_params), flax.core.unfreeze(goal_params)


def transfer_factored_encoder_params(target_params, state_params, goal_params):
    """
    Replace only the GCBC actor encoders, leaving its policy head random init"""

    params = flax.core.unfreeze(target_params)
    actor_encoder = params['modules_actor']['gc_encoder']

    expected_state = jax.tree_util.tree_map(
        lambda value: value.shape, actor_encoder['state_encoder']
    )
    expected_goal = jax.tree_util.tree_map(
        lambda value: value.shape, actor_encoder['goal_encoder']
    )
    source_state = jax.tree_util.tree_map(
        lambda value: value.shape, state_params
    )
    source_goal = jax.tree_util.tree_map(
        lambda value: value.shape, goal_params
    )
    if expected_state != source_state or expected_goal != source_goal:
        raise ValueError(
            'Source and target factored encoder shapes do not match. '
            f'expected_state={expected_state}, source_state={source_state}, '
            f'expected_goal={expected_goal}, source_goal={source_goal}'
        )

    actor_encoder['state_encoder'] = state_params
    actor_encoder['goal_encoder'] = goal_params
    return flax.core.freeze(params)
