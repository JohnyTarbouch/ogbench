import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out

#################################################################
# DrQ encoder
class DrQEncoder(nn.Module):
    """
    DrQ-v2 RGB encoder.

    DrQ-v2 style:
        image -> conv stack -> projection -> layer norm -> tanh

    Input:
        x: uint8 RGB image (H, W, C)

    Output:
        feature vector (feature_dim)

    It normalizes internally:
        x / 255.0 - 0.5

    In GCBC/GCIQL, this is usually used through GCEncoder(concat_encoder=...).
    That means the encoder receives state and goal images concatenated on the
    channel axis, e.g. RGB state + RGB goal -> 6 channels. Flax Conv supports
    this directly, so the same module works for goal-conditioned image inputs.
    """

    feature_dim: int = 256

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        init = nn.initializers.xavier_uniform()

        x = x.astype(jnp.float32) / 255.0 - 0.5

        x = nn.relu(
            nn.Conv(
                32,
                kernel_size=(3, 3),
                strides=2,
                padding='VALID',
                kernel_init=init,
                name='conv0',
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                32,
                kernel_size=(3, 3),
                strides=1,
                padding='VALID',
                kernel_init=init,
                name='conv1',
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                32,
                kernel_size=(3, 3),
                strides=1,
                padding='VALID',
                kernel_init=init,
                name='conv2',
            )(x)
        )
        x = nn.relu(
            nn.Conv(
                32,
                kernel_size=(3, 3),
                strides=1,
                padding='VALID',
                kernel_init=init,
                name='conv3',
            )(x)
        )

        x = x.reshape((*x.shape[:-3], -1))
        x = nn.Dense(self.feature_dim, kernel_init=init, name='proj')(x)
        x = nn.LayerNorm(epsilon=1e-5, name='proj_ln')(x)

        return jnp.tanh(x)
#################################################################


class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals.

        If `goal_encoded` is True, `goals` is assumed to be already encoded representations. In this case, either
        `goal_encoder` or `concat_encoder` must be None.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'drq': DrQEncoder,
}
