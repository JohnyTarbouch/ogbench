"""
Delivery-aware phase stitching with guarded k-nearest-neighbor retrieval.
"""

from agents.stitching_experiments.gcbc_cube_stitch_phase_delivery_segment_guard_p01 import (
    get_config as get_delivery_segment_config,
)


def get_config():
    config = get_delivery_segment_config()
    config.update(
        dict(
            stitch_retrieval_mode='knn',
            stitch_knn_k=512,
            stitch_knn_algorithm='auto',
            stitch_knn_n_jobs=1,
            stitch_debug_samples=2000,
        )
    )
    return config
