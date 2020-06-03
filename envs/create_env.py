def create_env(env, **kwargs):
    """Expected names are: doom_battle, atari_montezuma, etc."""

    if env.startswith('doom_'):
        from seed_rl.envs.doom.doom_utils import make_doom_env
        env = make_doom_env(env, **kwargs)
    elif env.startswith('MiniGrid'):
        from seed_rl.envs.minigrid.minigrid_utils import make_minigrid_env
        env = make_minigrid_env(env, **kwargs)
    elif env.startswith('atari_'):
        from seed_rl.envs.atari.atari_utils import make_atari_env
        return make_atari_env(env, **kwargs)
    elif env.startswith('dmlab_'):
        from seed_rl.envs.dmlab.dmlab_env import make_dmlab_env
        return make_dmlab_env(env, **kwargs)
    elif env.startswith('quadrotor_'):
        from seed_rl.envs.quadrotors.quad_utils import make_quadrotor_env
        return make_quadrotor_env(env, **kwargs)
    else:
        raise Exception('Unsupported env {0}'.format(env))

    return env
