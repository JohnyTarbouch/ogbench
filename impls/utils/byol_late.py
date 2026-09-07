"""
Late movement BYOL gating, the BC batch and inference inputs stay intact
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np


def late_window_starts(starts, endpoints, fraction):
    """
    select the final (fraction * N) action rows of each [start, end].
    """
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError('BYOL late fraction must be finite and in (0, 1].')
    starts = np.asarray(starts, dtype=np.int64)
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if starts.shape != endpoints.shape or np.any(endpoints <= starts):
        raise ValueError('BYOL late windows require matching, nonempty segment ranges.')
    ratio = Fraction(str(float(fraction)))
    lengths = endpoints - starts
    counts = (lengths * ratio.numerator + ratio.denominator - 1) // ratio.denominator
    return endpoints - counts


def sample_weights(batch, config):
    if float(config.get('byol_late_fraction', 0.0)) <= 0.0:
        return None
    if config['pred_loss_type'] != 'bdino':
        raise ValueError('Late BYOL currently supports the pinned bdino objective only.')
    if 'byol_sample_weights' not in batch:
        raise ValueError('Late BYOL requires byol_sample_weights from the atomic sampler.')
    weights = jnp.asarray(batch['byol_sample_weights'], dtype=jnp.float32)
    if weights.shape != (batch['observations'].shape[0],):
        raise ValueError('BYOL weights must have one entry per sampled action row.')
    return weights


def weighted_bdino(z_pred, z_target, weights):
    probabilities = jax.nn.softmax(z_pred, axis=-1)
    targets = jax.nn.softmax(z_target, axis=-1)
    losses = -jnp.sum(targets * jnp.log(probabilities + 1e-6), axis=-1)
    if losses.ndim == 2:
        losses = jnp.mean(losses, axis=0)
    elif losses.ndim != 1:
        raise ValueError('Expected BYOL logits with shape (B,K) or (E,B,K).')
    if weights.shape != losses.shape:
        raise ValueError('BYOL weights and per-example losses must match.')
    return jnp.sum(jnp.where(weights > 0, losses * weights, 0.0)) / jnp.maximum(
        jnp.sum(weights), 1.0
    )


def weight_metrics(weights):
    return {
        'byol/active_samples': jnp.sum(weights),
        'byol/active_fraction': jnp.mean(weights),
    }
