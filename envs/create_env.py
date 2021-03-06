# global user env registry. Add key-value pairs such as 'env_name': make_env_func
# See examples of make_env_funcs below (e.g. make_doom_env)
ENV_REGISTRY = dict()


def register_custom_env(env_name, make_env_func):
    """Call this before algo initialize."""
    assert callable(make_env_func), 'make_env_func should be callable'
    assert env_name not in ENV_REGISTRY, f'env {env_name} is already in the registry'
    ENV_REGISTRY[env_name] = make_env_func


def create_env(env, **kwargs):
    """Expected names are: doom_battle, atari_montezuma, etc."""

    if env in ENV_REGISTRY:
        return ENV_REGISTRY[env](env, **kwargs)

    # env is not in env registry, try one of the default envs

    if env.startswith('doom_'):
        from seed_rl.envs.doom.doom_utils import make_doom_env
        env = make_doom_env(env, **kwargs)
    elif env.startswith('atari_'):
        from seed_rl.envs.atari.atari_utils import make_atari_env
        return make_atari_env(env, **kwargs)
    elif env.startswith('dmlab_'):
        from seed_rl.envs.dmlab.dmlab_env import make_dmlab_env
        return make_dmlab_env(env, **kwargs)
    else:
        raise Exception('Unsupported env {0}'.format(env))

    return env
