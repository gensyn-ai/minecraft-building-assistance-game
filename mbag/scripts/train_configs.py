import torch
from sacred import Experiment

# flake8: noqa: F841


def make_named_configs(ex: Experiment):

    @ex.named_config
    def alphazero_human():
        run = "MbagAlphaZero"
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        sample_batch_size = 8000
        sgd_minibatch_size = 512
        random_start_locations = True
        num_training_iters = 2000
        train_batch_size = 8
        use_replay_buffer = True
        replay_buffer_size = 20
        num_workers = 10
        num_envs_per_worker = 8
        evaluation_num_workers = 0
        num_gpus = 0.5 if torch.cuda.is_available() else 0
        num_gpus_per_worker = 0.025 if torch.cuda.is_available() else 0
        model = "transformer"
        hidden_size = 64
        use_separated_transformer = True
        num_layers = 9
        vf_share_layers = True
        num_simulations = 100
        num_sgd_iter = 1
        save_freq = 5
        horizon = 1500
        teleportation = False
        inf_blocks = True
        noop_reward = -0.2
        get_resources_reward = 0.0
        action_reward = 0.0
        use_goal_predictor = False
        use_bilevel_action_selection = True
        fix_bilevel_action_selection = True
        temperature = 1.5
        dirichlet_noise = 0.25
        dirichlet_action_subtype_noise_multiplier = 10.0
        dirichlet_epsilon = 0.25
        prior_temperature = 1.0
        init_q_with_max = False
        gamma = 0.95
        lr = 0.001
        puct_coefficient = 1.0
        scale_obs = True
        randomize_first_episode_length = True
        line_of_sight_masking = True
        grad_clip = 0.1
        rollout_fragment_length = 100

    @ex.named_config
    def bc_human():
        run = "BC"
        data_split = "human_alone"
        inf_blocks = True
        teleportation = False
        input = (
            f"data/human_data_cleaned/{data_split}/"
            f"infinite_blocks_{str(inf_blocks).lower()}/"
            "rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_0"
        )
        train_batch_size = {
            True: {
                "human_alone": 9642,
                "combined": 19121,
            },
        }[
            inf_blocks
        ][data_split]
        num_workers = 0
        evaluation_interval = 1
        save_freq = 1
        evaluation_num_workers = 8
        evaluation_duration = 64
        num_envs_per_worker = 8
        num_gpus_per_worker = 0.0625 if torch.cuda.is_available() else 0
        use_extra_features = True
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        model = "transformer"
        line_of_sight_masking = True
        hidden_size = 64
        sgd_minibatch_size = 128
        use_separated_transformer = True
        num_layers = 9
        vf_share_layers = True
        num_sgd_iter = 1
        inf_blocks = True
        teleportation = False
        random_start_locations = True
        policies_to_train = ["human"]
        compress_observations = True
        horizon = 1500
        mask_action_distribution = True
        num_training_iters = 20
        entropy_coeff_start = 0
        evaluation_explore = True
        checkpoint_name = None
        checkpoint_to_load_policies = None
        if checkpoint_to_load_policies is not None:
            load_policies_mapping = {"human": "human"}
        overwrite_loaded_policy_type = True
        lr_start = 1e-3 if checkpoint_to_load_policies is None else 1e-4
        lr_schedule = [[0, lr_start], [train_batch_size * 10, lr_start / 10]]
        vf_loss_coeff = 0
        gamma = 0.95
        scale_obs = True
        permute_block_types = True
        experiment_tag = (
            f"bc_human/infinite_blocks_{str(inf_blocks).lower()}/{data_split}"
        )
        if checkpoint_to_load_policies is not None:
            experiment_tag += f"/init_{checkpoint_name}"
