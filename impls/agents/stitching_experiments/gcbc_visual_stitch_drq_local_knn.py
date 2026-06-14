"""GCBC visual stitching with DrQ features and local kNN retrieval."""

from agents.stitching_experiments.gcbc_visual_stitch_drq import get_config as get_kmeans_config


def get_config():
    config = get_kmeans_config()
    config.update(
        dict(
            dataset_class='VisualFeatureLocalKnnTemporalStitchGCDataset',
            stitch_local_knn_k=32,
            stitch_local_knn_max_candidates=256,
            stitch_local_knn_sample_topk=True,
        )
    )
    return config
