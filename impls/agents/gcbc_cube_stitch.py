from agents.gcbc_stitch import get_config as get_stitch_config


def get_config():
    config = get_stitch_config()
    config.update(
        dict(
            # probability of temporal goal replacement
            stitch_p_aug=0.5,
            # for cube xyz dims
            stitch_space='xy',
            stitch_xy_dims=(19, 20, 21),
            stitch_mode='kmeans',
            # keepthe waypoint close for relabeling
            stitch_nclusters=200,
            stitch_kmeans_random_state=0,
            stitch_cross_traj_only=True,
            stitch_future_min=0,
            stitch_debug_samples=2000,
        )
    )
    return config
