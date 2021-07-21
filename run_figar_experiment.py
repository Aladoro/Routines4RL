import json
import time
from collections import OrderedDict
import os.path as osp
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import dmc2gym

import logz
from sac_models import Critic
from figar_models import FiGARDDPG, FiGARActor, StandardFiGARActor, StandardFiGARCritic
from samplers import FiGARSampler
from buffers import FiGARReplayBuffer

tfl = tf.keras.layers
tfm = tf.keras.models
tfo = tf.keras.optimizers
tfi = tf.keras.initializers
tfd = tfp.distributions


def make_env(task_params):
    domain_name = task_params.get('domain_name', 'walker')
    task_name = task_params.get('task_name', 'walk')
    seed = task_params.get('seed', 1)
    image_size = task_params.get('image_size', 84)
    action_repeat = task_params.get('action_repeat', 1)
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=False,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat
    )
    return env


def make_models(task_params, training_params, agent_params):
    episode_limit = task_params.get('episode_limit', 1000)

    initial_random_samples = training_params.get('initial_random_samples', 1000)
    buffer_size = training_params.get('buffer_size', 1000000)
    actor_lr = training_params.get('actor_lr', 1e-3)
    actor_beta1 = training_params.get('actor_beta1', 0.9)
    clip_actor_gradients = training_params.get('clip_actor_gradients', False)

    critic_lr = training_params.get('critic_lr', 1e-3)
    critic_beta1 = training_params.get('critic_beta1', 0.9)

    algo = agent_params.get('algo', 'custom')
    gamma = agent_params.get('gamma', 0.99)
    q_polyak = agent_params.get('q_polyak', 0.99)
    actor_layers_dim = agent_params.get('actor_layers_dim', 256)
    critic_layers_dim = agent_params.get('critic_layers_dim', 256)
    q_target_noise = agent_params.get('q_target_noise', 0.1)
    max_q_target_noise = agent_params.get('max_q_target_noise', 0.5)
    possible_repetitions = agent_params.get('possible_repetitions', [1, 2, 3, 4])
    use_min_q_act_opt = agent_params.get('use_min_q_act_opt', False)
    sample_all_intermediate = agent_params.get('sample_all_intermediate', False)

    if sample_all_intermediate:
        assert algo == 'custom'
    orthogonal_init = tfi.Orthogonal(gain=np.sqrt(2))

    env = make_env(task_params)
    eval_env = make_env(task_params)

    action_size = tf.reduce_prod(env.action_space.shape)
    print(action_size)
    state_size = tf.reduce_prod(env.observation_space.shape)

    if algo == 'custom':
        def make_actor():
            actor = FiGARActor([tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                                tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                                tf.keras.layers.Dense(action_size + len(possible_repetitions),
                                                      kernel_initializer=orthogonal_init)],
                               possible_repetitions=possible_repetitions)
            return actor

        def make_critic():
            critic = Critic(
                [tf.keras.layers.Dense(critic_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                 tf.keras.layers.Dense(critic_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                 tf.keras.layers.Dense(1, kernel_initializer=orthogonal_init)])
            return critic
    elif algo == 'standard':
        def make_actor():
            return StandardFiGARActor(action_dim=action_size)

        def make_critic():
            return StandardFiGARCritic(observation_dims=state_size)
    else:
        raise NotImplementedError

    sampler = FiGARSampler(env, eval_env, episode_limit=episode_limit,
                           init_random_samples=initial_random_samples)

    print('Max episode length - {}'.format(sampler._max_episode_steps))

    replay_buffer = FiGARReplayBuffer(buffer_size)

    actor_optimizer = tf.keras.optimizers.Adam(actor_lr, beta_1=actor_beta1)
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr, beta_1=critic_beta1)

    if algo == 'custom':
        agent = FiGARDDPG(possible_repetitions=possible_repetitions,
                          make_actor=make_actor,
                          make_critic=make_critic,
                          make_critic2=make_critic,
                          actor_optimizer=actor_optimizer,
                          critic_optimizer=critic_optimizer,
                          gamma=gamma,
                          q_polyak=q_polyak,
                          train_actor_noise=q_target_noise,
                          max_train_actor_noise=max_q_target_noise,
                          clip_actor_gradients=clip_actor_gradients,
                          use_min_q_act_opt=use_min_q_act_opt, )
    elif algo == 'standard':
        agent = FiGARDDPG(possible_repetitions=list(range(1, 16)),
                          make_actor=make_actor,
                          make_critic=make_critic,
                          make_critic2=None,
                          actor_optimizer=actor_optimizer,
                          critic_optimizer=critic_optimizer,
                          gamma=gamma,
                          q_polyak=q_polyak,
                          train_actor_noise=0.0,
                          max_train_actor_noise=max_q_target_noise,
                          clip_actor_gradients=clip_actor_gradients,
                          use_min_q_act_opt=False, )

    obs = np.expand_dims(env.reset().astype('float32'), axis=0)
    agent(obs)
    agent.summary()
    return env, eval_env, agent, replay_buffer, sampler


def run_exp(task_params, training_params, agent_params, agent, replay_buffer, sampler):
    exp_name = task_params.get('exp_name', None)
    dir_prefix = task_params.get('dir_prefix', '')
    exp_suffix = task_params.get('exp_suffix', '')
    domain_name = task_params.get('domain_name', 'walker')
    task_name = task_params.get('task_name', 'walk')

    epochs = training_params.get('epochs', 100)
    test_runs_per_epoch = training_params.get('test_runs_per_epoch', 10)
    steps_per_epoch = training_params.get('steps_per_epoch', 10000)
    start_training = training_params.get('start_training', 1000)
    return_threshold = training_params.get('return_threshold', 1001)
    batch_size = training_params.get('batch_size', 128)
    updates_per_step = training_params.get('updates_per_step', 1)
    actor_delay = training_params.get('actor_delay', 2)
    action_sampling_noise = training_params.get('action_sampling_noise', 0.2)

    algo = agent_params.get('algo', 'custom')
    possible_repetitions = agent_params.get('possible_repetitions', [1, 2, 3, 4])
    start_epsilon = agent_params.get('start_epsilon', 0.2)
    annealing_steps = agent_params.get('annealing_steps', 0)
    sample_all_intermediate = agent_params.get('sample_all_intermediate', False)

    description_prefix = 'state'

    if algo == 'standard':
        brief_exp_description = 'FiGARR_standard_{}_{}b'.format(
            description_prefix, batch_size)
    elif algo == 'custom':
        brief_exp_description = 'FiGARR_custom_{}_{}b_{}reps'.format(
            description_prefix, batch_size, possible_repetitions)

    log_exp_name = exp_name or brief_exp_description
    log_exp_name += exp_suffix

    log_dir = osp.join('experiments_data/', '{}{}_{}/{}/{}'.format(
        dir_prefix, domain_name, task_name, log_exp_name, time.time()))

    params = OrderedDict({'task_params': task_params, 'training_params': training_params,
                          'agent_params': agent_params})

    logz.configure_output_dir(log_dir)
    logz.save_params(params)

    mean_test_returns = []
    mean_test_std = []
    steps = []

    step_counter = 0
    logz.log_tabular('epoch', 0)
    logz.log_tabular('number_collected_observations', step_counter)
    print('Epoch {}/{} - total steps {}'.format(0, epochs, step_counter))
    start_training_time = time.time()
    evaluation_stats = sampler.evaluate(agent, test_runs_per_epoch, log=False)
    for k, v in evaluation_stats.items():
        logz.log_tabular(k, v)
    mean_test_returns.append(evaluation_stats['episode_returns_mean'])
    mean_test_std.append(evaluation_stats['episode_returns_std'])
    steps.append(step_counter)
    epoch_end_eval_time = time.time()
    evaluation_time = epoch_end_eval_time - start_training_time

    logz.log_tabular('training_time', 0.0)
    logz.log_tabular('evaluation_time', np.around(evaluation_time, decimals=3))
    logz.dump_tabular()

    eps = start_epsilon

    if annealing_steps > 0:
        epsilon_step = eps / annealing_steps
    else:
        epsilon_step = 0.0

    first_training_iter = True
    for e in range(epochs):
        logz.log_tabular('epoch', e + 1)
        epoch_start_time = time.time()
        while step_counter < (e + 1) * steps_per_epoch:
            if sample_all_intermediate and algo == 'custom':
                traj_data = sampler.sample_all_intermediate_steps(agent, action_sampling_noise,
                                                                  epsilon=eps, n_steps=1)
            else:
                traj_data = sampler.sample_steps(agent, action_sampling_noise,
                                                 epsilon=eps, n_steps=1)

            eps = np.maximum(eps - epsilon_step * traj_data['actual_steps'], 0.0)

            replay_buffer.add(traj_data)
            step_counter += traj_data['actual_steps']
            if step_counter > start_training and (not first_training_iter):
                agent.train(replay_buffer, batch_size=batch_size,
                            n_updates=updates_per_step * traj_data['actual_steps'],
                            act_delay=actor_delay, )
            elif step_counter >= start_training and first_training_iter:
                agent.train(replay_buffer, batch_size=batch_size,
                            n_updates=updates_per_step * step_counter,
                            act_delay=actor_delay, )
                first_training_iter = False

        logz.log_tabular('number_collected_observations', step_counter)
        epoch_end_training_time = time.time()
        training_time = epoch_end_training_time - epoch_start_time
        print('Epoch {}/{} - total steps {}'.format(e + 1, epochs, step_counter))
        evaluation_stats = sampler.evaluate(agent, test_runs_per_epoch, log=False)
        for k, v in evaluation_stats.items():
            logz.log_tabular(k, v)
        mean_test_returns.append(evaluation_stats['episode_returns_mean'])
        mean_test_std.append(evaluation_stats['episode_returns_std'])
        steps.append(step_counter)
        epoch_end_eval_time = time.time()
        evaluation_time = epoch_end_eval_time - epoch_end_training_time

        logz.log_tabular('training_time', np.around(training_time, decimals=3))
        logz.log_tabular('evaluation_time', np.around(evaluation_time, decimals=3))
        logz.dump_tabular()
        if evaluation_stats['episode_returns_mean'] > return_threshold:
            print('Early termination due to reaching return threshold')
            break
    total_training_time = time.time() - start_training_time
    print('Total training time: {}'.format(total_training_time))
    plt.errorbar(steps, mean_test_returns, mean_test_std)
    plt.xlabel('steps')
    plt.ylabel('returns')
    plt.show(block=False)


def run_experiment_from_params(param_path, repetitions=1, gpu=0, memory_limit=10000, randomize_seed=False):
    if gpu >= 0:
        GPU_TO_USE = gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[GPU_TO_USE], 'GPU')
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[GPU_TO_USE],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                print(e)
    else:
        tf.config.set_visible_devices([], 'GPU')

    with open(param_path, 'r') as inp:
        data = json.loads(inp.read())
    task_params = data['task_params']
    training_params = data['training_params']
    agent_params = data['agent_params']

    for i in range(repetitions):
        if randomize_seed:
            task_params['seed'] = np.random.randint(10000)
        env, eval_env, agent, replay_buffer, sampler = make_models(task_params, training_params, agent_params)
        run_exp(task_params, training_params, agent_params, agent, replay_buffer, sampler)
    return agent, replay_buffer, sampler
