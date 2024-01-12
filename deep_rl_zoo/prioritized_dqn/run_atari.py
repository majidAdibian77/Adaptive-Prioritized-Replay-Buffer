# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
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
# ==============================================================================
"""
From the paper "Prioritized Experience Replay" http://arxiv.org/abs/1511.05952.
"""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
import os

# pylint: disable=import-error
from deep_rl_zoo.networks.value import DqnConvNet
from deep_rl_zoo.prioritized_dqn import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo.schedule import LinearSchedule, ExponentialSchedule, PriorityExponentSchedule
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib

"""
python3 -m deep_rl_zoo.prioritized_dqn.run_atari --run_name "orignal_alpha0.6" --environment_name=Seaquest --replay_capacity 100000 \
                --min_replay_size 10000 --exploration_epsilon_decay_step 10000000 --exploration_epsilon_begin_value 1 \
                --exploration_epsilon_end_value 0.0001 --num_iterations 501 --num_train_steps 20000 --eval_exploration_epsilon 0 \
                --num_eval_episods 5 --save_ckpt_iter 100 --max_episode_steps 10000 \
                --use_fixed_priority_exponent  --fixed_priority_exponent 0.6

python3 -m deep_rl_zoo.prioritized_dqn.run_atari --run_name "adaptive_alpha" --environment_name=Seaquest --replay_capacity 100000 \
                --min_replay_size 10000 --exploration_epsilon_decay_step 10000000 --exploration_epsilon_begin_value 1 \
                --exploration_epsilon_end_value 0.0001 --num_iterations 501 --num_train_steps 20000 --eval_exploration_epsilon 0 \
                --num_eval_episods 5 --save_ckpt_iter 100 --max_episode_steps 10000 \
                --priority_exponent_begin_value 0.6  --priority_exponent_end_value 0.1

python3 -m deep_rl_zoo.prioritized_dqn.run_atari --run_name "exponent-reward-proj-TDerror" --environment_name=Seaquest --replay_capacity 100000 \
                --min_replay_size 10000 --exploration_epsilon_decay_step 10000000 --exploration_epsilon_begin_value 1 \
                --exploration_epsilon_end_value 0.0001 --num_iterations 501 --num_train_steps 20000 --eval_exploration_epsilon 0 \
                --num_eval_episods 5 --save_ckpt_iter 100 --max_episode_steps 10000 \
                --use_fixed_priority_exponent  --fixed_priority_exponent 1 \
                --use_reward_in_priortized_replay

python3 -m deep_rl_zoo.prioritized_dqn.run_atari --run_name "exponent-counter-proj-TDerror" --environment_name=Seaquest --replay_capacity 100000 \
                --min_replay_size 10000 --exploration_epsilon_decay_step 10000000 --exploration_epsilon_begin_value 1 \
                --exploration_epsilon_end_value 0.0001 --num_iterations 501 --num_train_steps 20000 --eval_exploration_epsilon 0 \
                --num_eval_episods 5 --save_ckpt_iter 100 --max_episode_steps 10000 \
                --use_fixed_priority_exponent  --fixed_priority_exponent 1 \
                --use_replay_counter
                
python3 -m deep_rl_zoo.prioritized_dqn.run_atari --run_name "exponent-Pi-softmax" --environment_name=Seaquest --replay_capacity 100000 \
                --min_replay_size 10000 --exploration_epsilon_decay_step 10000000 --exploration_epsilon_begin_value 1 \
                --exploration_epsilon_end_value 0.0001 --num_iterations 501 --num_train_steps 20000 --eval_exploration_epsilon 0 \
                --num_eval_episods 5 --save_ckpt_iter 100 --max_episode_steps 10000 \
                --use_fixed_priority_exponent  --fixed_priority_exponent 1 \
                --use_Pi_in_priortize_replay
"""

FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.')
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_bool('compress_state', True, 'Compress state images when store in experience replay.')
flags.DEFINE_integer('replay_capacity', int(1e6), 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 50000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 32, 'Sample batch size when updating the neural network.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 10.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.01, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float(
    'exploration_epsilon_decay_step',
    int(1e6),
    'Total steps (after frame skip) to decay value of the exploration rate.',
)

flags.DEFINE_float('eval_exploration_epsilon', 0.01, 'Fixed exploration rate in e-greedy policy for evaluation.')

# flags.DEFINE_float('priority_exponent', 0.6, 'Priority exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent_begin_value', 0.4, 'Importance sampling exponent begin value.')
flags.DEFINE_float('importance_sampling_exponent_end_value', 1.0, 'Importance sampling exponent end value after decay.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize sampling weights in prioritized replay.')

flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer(
    'num_train_steps', int(5e5), 'Number of training steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer('learn_interval', 4, 'The frequency (measured in agent steps) to update parameters.')
flags.DEFINE_integer(
    'target_net_update_interval',
    2500,
    'The frequency (measured in number of Q network parameter updates) to update target networks.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/per_dqn_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')

flags.DEFINE_integer('save_ckpt_iter', 1, 'save checkpint evry n iterations')
flags.DEFINE_string('device', 'cuda:1', 'device to run train model: cpu or cuda')
flags.DEFINE_integer('num_eval_episods', 5, 'Number of episods in evaluation env to run per iteration.')
# flags.DEFINE_integer('max_eval_episod_steps', int(1e3), 'max_eval_episod_steps')
flags.DEFINE_string('run_name', 'v1', 'name of current run')
flags.DEFINE_float('priority_exponent_begin_value', 0.6, 'Contibution 4:Priority exponent begin value.')
flags.DEFINE_float('priority_exponent_end_value', 0.1, 'Contibution 4: Priority exponentt end value after decay.')
flags.DEFINE_float('fixed_priority_exponent', 0.6, 'Priority exponent used in prioritized replay.')
flags.DEFINE_bool('use_fixed_priority_exponent', False, 'Contibution 4: Use fixed priority_exponent (alpha) in priortize replay buffer')
flags.DEFINE_bool('use_replay_counter', False, 'Contibution 2: Use counter in priortize replay buffer')
flags.DEFINE_bool('use_reward_in_priortized_replay', False, 'Contribution 1: Use reward in priortized replay buffer')
flags.DEFINE_bool('use_Pi_in_priortize_replay', False, 'Contribution 3: Use Pi(a|s) os samples in priortize replay buffer')


def main(argv):
    """Trains DQN agent with prioritized replay on Atari."""
    del argv
    runtime_device = torch.device(FLAGS.device)
    logging.info(f'Runs DQN agent with prioritized replay on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    # Create environment.
    def environment_builder():
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**10),
            noop_max=30,
            terminal_on_life_loss=True,
        )

    train_env = environment_builder()
    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', train_env.action_space.n)
    logging.info('Observation spec: %s', train_env.observation_space.shape)

    state_dim = train_env.observation_space.shape
    action_dim = train_env.action_space.n

    # Test environment and state shape.
    obs = train_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)

    network = DqnConvNet(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate)

    # Test network input and output
    q = network(torch.from_numpy(obs[None, ...]).float()).q_values
    assert q.shape == (1, action_dim)

    # Create e-greedy exploration epsilon schedule
    exploration_epsilon_schedule = ExponentialSchedule(
        begin_t=int(FLAGS.min_replay_size),
        decay_steps=int(FLAGS.exploration_epsilon_decay_step),
        begin_value=FLAGS.exploration_epsilon_begin_value,
        end_value=FLAGS.exploration_epsilon_end_value,
    )

    # Create prioritized transition replay
    # Note the t in the replay is not exactly aligned with the agent t.
    importance_sampling_exponent_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        end_t=(FLAGS.num_iterations * int(FLAGS.num_train_steps)),
        begin_value=FLAGS.importance_sampling_exponent_begin_value,
        end_value=FLAGS.importance_sampling_exponent_end_value,
    )
    if FLAGS.use_fixed_priority_exponent:
        priority_exponent_schedule = FLAGS.fixed_priority_exponent
    else:
        priority_exponent_schedule = PriorityExponentSchedule(
            begin_t=int(FLAGS.min_replay_size),
            end_t=(FLAGS.num_iterations * int(FLAGS.num_train_steps)),
            begin_value=FLAGS.priority_exponent_begin_value,
            end_value=FLAGS.priority_exponent_end_value,
        )

    if FLAGS.compress_state:

        def encoder(transition):
            return transition._replace(
                s_tm1=replay_lib.compress_array(transition.s_tm1),
                s_t=replay_lib.compress_array(transition.s_t),
            )

        def decoder(transition):
            return transition._replace(
                s_tm1=replay_lib.uncompress_array(transition.s_tm1),
                s_t=replay_lib.uncompress_array(transition.s_t),
            )

    else:
        encoder = None
        decoder = None

    replay = replay_lib.PrioritizedReplay(
        capacity=FLAGS.replay_capacity,
        structure=replay_lib.TransitionStructure,
        priority_exponent=priority_exponent_schedule,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
        encoder=encoder,
        decoder=decoder,
        use_reward_in_priortized_replay=FLAGS.use_reward_in_priortized_replay,
        use_counter=FLAGS.use_replay_counter,
        use_Pi_in_priortize_replay=FLAGS.use_Pi_in_priortize_replay,
    )

    # Create PrioritizedDqn agent instance
    train_agent = agent.PrioritizedDqn(
        network=network,
        optimizer=optimizer,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        replay=replay,
        exploration_epsilon=exploration_epsilon_schedule,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        learn_interval=FLAGS.learn_interval,
        target_net_update_interval=FLAGS.target_net_update_interval,
        discount=FLAGS.discount,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        action_dim=action_dim,
        random_state=random_state,
        device=runtime_device,
        agent_name = f"PER-DQN-{FLAGS.run_name}-{FLAGS.environment_name}",
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
        name=f'PER-DQN-{FLAGS.run_name}-{FLAGS.environment_name}-greedy-eval',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name, agent_name=f'PER-DQN-{FLAGS.run_name}-{FLAGS.environment_name}', save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('network', network))

    # Run the training and evaluation for N iterations.
    main_loop.run_single_thread_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        train_agent=train_agent,
        train_env=train_env,
        eval_agent=eval_agent,
        eval_env=eval_env,
        checkpoint=checkpoint,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_interval=FLAGS.debug_screenshots_interval,

        max_eval_episod_steps = FLAGS.max_episode_steps,
        save_ckpt_iter = FLAGS.save_ckpt_iter,
        num_eval_episods=FLAGS.num_eval_episods,
        csv_file=os.path.join(FLAGS.results_csv_path, f"per_dqn_{FLAGS.run_name}-{FLAGS.environment_name}_results.csv"),
    )


if __name__ == '__main__':
    app.run(main)
