def env_override_defaults(env, parser):
    if env.startswith('doom'):
        from seed_rl.envs.doom.doom_params import doom_override_defaults
        doom_override_defaults(env, parser)
    elif env.startswith('dmlab'):
        from seed_rl.envs.dmlab.dmlab_params import dmlab_override_defaults
        dmlab_override_defaults(env, parser)
    elif env.startswith('atari'):
        from seed_rl.envs.atari.atari_params import atari_override_defaults
        atari_override_defaults(env, parser)


def add_env_args(env, parser):
    p = parser

    p.add_argument('--env_frameskip', default=None, type=int, help='Number of frames for action repeat (frame skipping). Default (None) means use default environment value')
    p.add_argument('--env_framestack', default=4, type=int, help='Frame stacking (only used in Atari?)')
    p.add_argument('--pixel_format', default='CHW', type=str, help='PyTorch expects CHW by default, Ray & TensorFlow expect HWC')

    if env.startswith('doom'):
        from seed_rl.envs.doom.doom_params import add_doom_env_args
        add_doom_env_args(env, parser)
    elif env.startswith('dmlab'):
        from seed_rl.envs.dmlab.dmlab_params import add_dmlab_env_args
        add_dmlab_env_args(env, parser)
