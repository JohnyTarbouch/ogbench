import ml_collections

from agents.gcbc import get_config as get_base_config


def get_config():
    """
    Use the standard GCBC agent with fixed endpoint image-goal GCBC.
    """

    config = get_base_config()
    config.update(
        ml_collections.ConfigDict(
            dict(
                dataset_class='EndpointGoalDataset',
                endpoint_dataset_mode='stable_achieved_endpoint',
                endpoint_sampling='uniform_tasks',
                endpoint_goal_stack_mode='repeat_endpoint',
                endpoint_train_manifest_path='',
                endpoint_val_manifest_path='',
            )
        )
    )
    return config
