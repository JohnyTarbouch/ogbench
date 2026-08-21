"""Focused structural tests for the LCBC + BYOL-gamma hybrid agent."""

import pickle
import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.language_byol import LanguageBYOLAgent, get_config


def _test_config(alignment):
    config = get_config()
    config.encoder = 'impala_debug'
    config.num_language_tasks = 135
    config.frame_stack = 1
    config.use_obs_latent_dim = False
    config.value_latent_dim = 8
    config.value_hidden_dims = (8,)
    config.actor_hidden_dims = (16,)
    config.language_embedding_dim = 4
    config.language_hidden_dims = (8,)
    config.language_latent_dim = 4
    config.ensemble_size = 2
    # Exercise the same forward+backward, action-conditioned auxiliary branch
    # used by the production visual experiment.
    config.pred_both = True
    config.action_forward = True
    config.target = True
    config.alignment = alignment
    config.bc_weight = 1.0
    return config


def _tree_norm(tree):
    return jnp.sqrt(
        sum(jnp.sum(leaf * leaf) for leaf in jax.tree_util.tree_leaves(tree))
    )


class LanguageBYOLAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        keys = jax.random.split(jax.random.PRNGKey(17), 5)
        cls.example_observations = jax.random.randint(
            keys[0], (1, 32, 32, 3), 0, 256, dtype=jnp.uint8
        )
        cls.example_actions = jnp.zeros((1, 5), dtype=jnp.float32)
        cls.alignment_zero = LanguageBYOLAgent.create(
            3,
            cls.example_observations,
            cls.example_actions,
            _test_config(0.0),
        )
        cls.alignment_ten = LanguageBYOLAgent.create(
            3,
            cls.example_observations,
            cls.example_actions,
            _test_config(10.0),
        )
        cls.batch = {
            'observations': jax.random.randint(
                keys[1], (2, 32, 32, 3), 0, 256, dtype=jnp.uint8
            ),
            'value_goals': jax.random.randint(
                keys[2], (2, 32, 32, 3), 0, 256, dtype=jnp.uint8
            ),
            'language_embeddings': jax.random.normal(keys[3], (2, 4)),
            'actions': jax.random.uniform(
                keys[4], (2, 5), minval=-0.5, maxval=0.5
            ),
        }

    def test_alignment_weight_does_not_change_architecture_or_initialization(self):
        self.assertIs(get_config().get_type('num_language_tasks'), int)
        zero_leaves, zero_tree = jax.tree_util.tree_flatten(
            self.alignment_zero.network.params
        )
        ten_leaves, ten_tree = jax.tree_util.tree_flatten(
            self.alignment_ten.network.params
        )
        self.assertEqual(zero_tree, ten_tree)
        self.assertTrue(
            all(
                bool(jnp.array_equal(zero, ten))
                for zero, ten in zip(zero_leaves, ten_leaves)
            )
        )

    def test_bc_and_byol_both_update_the_shared_visual_encoder(self):
        bc_grads = jax.grad(
            lambda params: self.alignment_zero.actor_loss(self.batch, params)[0]
        )(self.alignment_zero.network.params)
        byol_grads = jax.grad(
            lambda params: self.alignment_ten.pred_loss(
                self.batch['observations'],
                self.batch['value_goals'],
                self.batch['actions'],
                params,
            )[0]
        )(self.alignment_ten.network.params)

        self.assertGreater(float(_tree_norm(bc_grads['modules_encoder'])), 0.0)
        self.assertGreater(float(_tree_norm(byol_grads['modules_encoder'])), 0.0)
        # The LCBC actor consumes the direct encoder feature, not the auxiliary
        # BYOL projector.  This preserves the baseline LCBC actor architecture.
        self.assertEqual(float(_tree_norm(bc_grads['modules_value'])), 0.0)

    def test_total_loss_decomposition_and_update_are_finite(self):
        rng = jax.random.PRNGKey(23)
        loss_zero, _ = self.alignment_zero.total_loss(
            self.batch,
            self.alignment_zero.network.params,
            rng=rng,
        )
        loss_ten, info_ten = self.alignment_ten.total_loss(
            self.batch,
            self.alignment_zero.network.params,
            rng=rng,
        )
        prediction_loss = (
            info_ten['pred_f/pred_loss'] + info_ten['pred_b/pred_loss']
        )

        self.assertTrue(bool(jnp.isfinite(loss_zero)))
        self.assertTrue(bool(jnp.isfinite(loss_ten)))
        np.testing.assert_allclose(
            np.asarray(loss_ten - loss_zero),
            np.asarray(10.0 * prediction_loss),
            rtol=1e-6,
            atol=1e-6,
        )

        updated, update_info = self.alignment_ten.update(self.batch)
        self.assertEqual(int(updated.network.step), int(self.alignment_ten.network.step) + 1)
        for key in (
            'actor/actor_loss',
            'pred_f/pred_loss',
            'pred_b/pred_loss',
            'grad/norm',
        ):
            self.assertTrue(bool(jnp.isfinite(update_info[key])), key)
        self.assertGreater(
            sum(
                int(not bool(jnp.array_equal(before, after)))
                for before, after in zip(
                    jax.tree_util.tree_leaves(
                        self.alignment_ten.network.params['modules_encoder']
                    ),
                    jax.tree_util.tree_leaves(
                        updated.network.params['modules_encoder']
                    ),
                )
            ),
            0,
        )
        # The reference update uses the pre-gradient online parameters, so the
        # initially identical target changes on the second optimizer step.
        updated_twice, _ = updated.update(self.batch)
        self.assertGreater(
            sum(
                int(not bool(jnp.array_equal(before, after)))
                for before, after in zip(
                    jax.tree_util.tree_leaves(
                        updated.network.params['modules_target_encoder']
                    ),
                    jax.tree_util.tree_leaves(
                        updated_twice.network.params['modules_target_encoder']
                    ),
                )
            ),
            0,
        )

    def test_actor_is_independent_of_future_visual_targets(self):
        actor_batch = {
            key: value
            for key, value in self.batch.items()
            if key != 'value_goals'
        }
        actor_loss_without_goals, _ = self.alignment_ten.actor_loss(
            actor_batch,
            self.alignment_ten.network.params,
        )
        altered_batch = dict(self.batch)
        altered_batch['value_goals'] = 255 - self.batch['value_goals']
        actor_loss_with_altered_goals, _ = self.alignment_ten.actor_loss(
            altered_batch,
            self.alignment_ten.network.params,
        )
        np.testing.assert_array_equal(
            np.asarray(actor_loss_without_goals),
            np.asarray(actor_loss_with_altered_goals),
        )

        action_seed = jax.random.PRNGKey(29)
        action_a = self.alignment_ten.sample_actions(
            self.batch['observations'][0],
            self.batch['language_embeddings'][0],
            seed=action_seed,
        )
        action_b = self.alignment_ten.sample_actions(
            self.batch['observations'][0],
            self.batch['language_embeddings'][0],
            seed=action_seed,
        )
        np.testing.assert_array_equal(np.asarray(action_a), np.asarray(action_b))

    def test_drq_actor_uses_direct_256_dimensional_encoder_feature(self):
        config = _test_config(0.0)
        config.encoder = 'drq'
        agent = LanguageBYOLAgent.create(
            31,
            self.example_observations,
            self.example_actions,
            config,
        )
        features = agent._observation_representation(self.example_observations)
        self.assertEqual(features.shape, (1, 256))
        actor_kernel = agent.network.params['modules_actor']['actor']['actor_net'][
            'Dense_0'
        ]['kernel']
        expected_actor_input_dim = 256 + config.language_latent_dim
        self.assertEqual(actor_kernel.shape[0], expected_actor_input_dim)

    def test_checkpoint_state_round_trip_preserves_loss(self):
        state = flax.serialization.to_state_dict(self.alignment_ten)
        restored_state = pickle.loads(pickle.dumps(state))
        restored = flax.serialization.from_state_dict(
            self.alignment_ten,
            restored_state,
        )

        original_leaves = jax.tree_util.tree_leaves(self.alignment_ten.network.params)
        restored_leaves = jax.tree_util.tree_leaves(restored.network.params)
        self.assertTrue(
            all(
                bool(jnp.array_equal(original, round_trip))
                for original, round_trip in zip(original_leaves, restored_leaves)
            )
        )
        rng = jax.random.PRNGKey(37)
        original_loss, _ = self.alignment_ten.total_loss(
            self.batch,
            self.alignment_ten.network.params,
            rng=rng,
        )
        restored_loss, _ = restored.total_loss(
            self.batch,
            restored.network.params,
            rng=rng,
        )
        np.testing.assert_array_equal(
            np.asarray(original_loss),
            np.asarray(restored_loss),
        )


if __name__ == '__main__':
    unittest.main()
