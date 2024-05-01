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
        num_gpus = 0.5
        num_gpus_per_worker = 0.025
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
