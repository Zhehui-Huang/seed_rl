# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""R2D2 binary for ATARI-57.

Actor and learner are in the same binary so that all flags are shared.
"""

import logging
from logging import handlers
from absl import app
from absl import flags
from seed_rl.agents.r2d2 import learner
from seed_rl.atari import env
from seed_rl.atari import networks
from seed_rl.common import actor
from seed_rl.common import common_flags
import tensorflow as tf

from seed_rl.algorithms.utils.arguments import default_cfg
from seed_rl.envs.atari.atari_utils import ATARI_W, ATARI_H
from seed_rl.envs.create_env import create_env


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,
                 fmt='%(asctime)s : %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-6, 'Adam epsilon.')

flags.DEFINE_integer('stack_size', 4, 'Number of frames to stack.')


def create_agent(env_output_specs, num_actions):
  return networks.DuelingLSTMDQNNet(
      num_actions, env_output_specs.observation.shape, FLAGS.stack_size)


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate,
                                       epsilon=FLAGS.adam_epsilon)
  return optimizer, learning_rate_fn

def create_atari_env(x):
    env_name = 'atari_breakout'
    cfg = default_cfg(env=env_name, algo=None)
    cfg.pixel_format = 'HWC'
    cfg.res_w = ATARI_W
    cfg.res_h = ATARI_H
    cfg.env_framestack = 1
    return create_env(env_name, cfg=cfg)

def main(argv):
  fps_log = Logger('atari_r2d2_fps.log', level='info')

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    actor.actor_loop(create_atari_env)
  elif FLAGS.run_mode == 'learner':
    learner.learner_loop(create_atari_env,
                         create_agent,
                         create_optimizer,
                         fps_log)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
