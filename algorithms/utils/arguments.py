import argparse
import copy
import json
import os
import sys

from seed_rl.algorithms.utils.agent import Agent
from seed_rl.algorithms.utils.evaluation_config import add_eval_args
from seed_rl.envs.env_config import add_env_args, env_override_defaults
from seed_rl.utils.utils import log, AttrDict, cfg_file


def get_algo_class(algo):
    algo_class = Agent

    if algo == 'PPO':
        from seed_rl.algorithms.ppo.agent_ppo import AgentPPO
        algo_class = AgentPPO
    elif algo == 'APPO':
        from seed_rl.algorithms.appo.appo import APPO
        algo_class = APPO
    else:
        log.warning('Algorithm %s is not supported', algo)

    return algo_class


def parse_args(argv=None, evaluation=False):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('--algo', type=str, default=None, required=True)
    parser.add_argument('--env', type=str, default=None, required=True)
    parser.add_argument('--experiment', type=str, default=None, required=True)
    parser.add_argument('--experiments_root', type=str, default=None, required=False, help='If not None, store experiment data in the specified subfolder of train_dir. Useful for groups of experiments (e.g. gridsearch)')

    basic_args, _ = parser.parse_known_args(argv)
    algo = basic_args.algo
    env = basic_args.env

    # algorithm-specific parameters (e.g. for PPO)
    algo_class = get_algo_class(algo)
    algo_class.add_cli_args(parser)

    # env-specific parameters (e.g. for Doom env)
    add_env_args(env, parser)

    # env-specific default values for algo parameters (e.g. model size and convolutional head configuration)
    env_override_defaults(env, parser)

    if evaluation:
        add_eval_args(parser)

    # parse all the arguments (algo, env, and optionally evaluation)
    args = parser.parse_args(argv)
    args.command_line = ' '.join(argv)

    # following is the trick to get only the args passed from the command line
    # We copy the parser and set the default value of None for every argument. Since one cannot pass None
    # from command line, we can differentiate between args passed from command line and args that got initialized
    # from their default values. This will allow us later to load missing values from the config file without
    # overriding anything passed from the command line
    no_defaults_parser = copy.copy(parser)
    for arg_name in vars(args).keys():
        no_defaults_parser.set_defaults(**{arg_name: None})
    cli_args = no_defaults_parser.parse_args(argv)

    for arg_name in list(vars(cli_args).keys()):
        if cli_args.__dict__[arg_name] is None:
            del cli_args.__dict__[arg_name]

    args.cli_args = vars(cli_args)

    return args


def default_cfg(algo='PPO', env='env', experiment='test'):
    """Useful for tests."""
    return parse_args(argv=[f'--algo={algo}', f'--env={env}', f'--experiment={experiment}'])


def load_from_checkpoint(cfg):
    filename = cfg_file(cfg)
    if not os.path.isfile(filename):
        raise Exception(f'Could not load saved parameters for experiment {cfg.experiment}')

    with open(filename, 'r') as json_file:
        json_params = json.load(json_file)
        log.warning('Loading existing experiment configuration from %s', filename)
        loaded_cfg = AttrDict(json_params)

    # override the parameters in config file with values passed from command line
    for key, value in vars(cfg.cli_args).items():
        if loaded_cfg[key] != value:
            log.debug('Overriding arg %r with value %r passed from command line', key, value)
            loaded_cfg[key] = value

    # incorporate extra CLI parameters that were not present in JSON file
    for key, value in vars(cfg).items():
        if key not in loaded_cfg:
            log.debug('Adding new argument %r=%r that is not in the saved config file!', key, value)
            loaded_cfg[key] = value

    return loaded_cfg


def maybe_load_from_checkpoint(cfg):
    filename = cfg_file(cfg)
    if not os.path.isfile(filename):
        log.warning('Saved parameter configuration for experiment %s not found!', cfg.experiment)
        log.warning('Starting experiment from scratch!')
        return AttrDict(vars(cfg))

    return load_from_checkpoint(cfg)
