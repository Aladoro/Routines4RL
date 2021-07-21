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
from sac_models import StochasticActor, Critic, SAC
from td3_models import DDPG, Actor
from tr_models import RTDDPG, RTActor
from tr_sac_models import RTSAC
from samplers import Sampler
from buffers import FastReplayBuffer
from routine_encoders import RoutineEncoder, RoutineAE, SimpleFCRoutineDecoder, StochasticRoutineAE, \
    SimpleStochasticFCRoutineDecoder
from layers_utils import DenseRecursiveSum

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
    rt_ae_warmup = training_params.get('rt_ae_warmup', 0)
    rt_ae_lr = training_params.get('rt_ae_lr', 1e-3)
    rt_ae_beta1 = training_params.get('rt_ae_beta1', 0.9)
    actor_lr = training_params.get('actor_lr', 1e-3)
    actor_beta1 = training_params.get('actor_beta1', 0.9)
    clip_actor_gradients = training_params.get('clip_actor_gradients', False)
    critic_lr = training_params.get('critic_lr', 1e-3)
    critic_beta1 = training_params.get('critic_beta1', 0.9)
    entropy_coeff_lr = training_params.get('entropy_coeff_lr', 1e-4)
    entropy_coeff_beta1 = training_params.get('entropy_coeff_beta1', 0.5)
    rt_sampling_noise = training_params.get('rt_sampling_noise', 0.0)

    algo = agent_params.get('algo', 'SAC')
    gamma = agent_params.get('gamma', 0.99)
    q_polyak = agent_params.get('q_polyak', 0.99)
    clip_td_backup = agent_params.get('clip_td_backup', None)
    huber_td_loss = agent_params.get('huber_td_loss', False)
    use_target_actor = agent_params.get('use_target_actor', True)
    use_min_q_actor = agent_params.get('use_min_q_actor', False)
    rt_ae_type = agent_params.get('rt_ae_type', 'recurrent')
    targ_rt_ae = agent_params.get('targ_rt_ae', False)
    ema_rt_ae = agent_params.get('ema_rt_ae', False)
    rt_ae_polyak = agent_params.get('rt_ae_polyak', 0.99)
    joint_rt_ae_critic_optim = agent_params.get('joint_rt_ae_critic_optim', False)
    double_joint_rt_ae_critic_optim = agent_params.get('double_joint_rt_ae_critic_optim', False)
    joint_rt_ae_actor_optim = agent_params.get('joint_rt_ae_actor_optim', False)
    rt_ae_recon_optim = agent_params.get('rt_ae_recon_optim', True)
    rt_ae_actor_consistency_loss = agent_params.get('rt_ae_actor_consistency_loss', None)
    max_plan_len = agent_params.get('max_plan_len', 4)
    enc_embedding_dim = agent_params.get('enc_embedding_dim', 16)
    joint_enc_embedding_dim = agent_params.get('joint_enc_embedding_dim', 64)
    rt_dim = agent_params.get('rt_dim', 32)
    dec_embedding_dim = agent_params.get('dec_embedding_dim', enc_embedding_dim * 4)
    rt_l2_reg = agent_params.get('rt_l2_reg', 0.0)
    actor_layers_dim = agent_params.get('actor_layers_dim', 256)
    critic_layers_dim = agent_params.get('critic_layers_dim', 256)
    initial_entropy_coefficient = agent_params.get('initial_entropy_coefficient', 0.1)
    tune_entropy_coefficient = agent_params.get('tune_entropy_coefficient', True)
    target_entropy = agent_params.get('target_entropy', None)
    approximate_current_entropy = agent_params.get('approximate_current_entropy', True)
    lower_bound_q_target_entropy = agent_params.get('lower_bound_q_target_entropy', True)
    q_target_noise = agent_params.get('q_target_noise', 0.1)
    max_q_target_noise = agent_params.get('max_q_target_noise', 0.5)

    orthogonal_init = tfi.Orthogonal(gain=np.sqrt(2))

    env = make_env(task_params)
    eval_env = make_env(task_params)

    action_size = tf.reduce_prod(env.action_space.shape)
    state_size = tf.reduce_prod(env.observation_space.shape)

    if target_entropy == None:
        target_entropy = -1 * action_size

    if algo == 'SAC':
        def make_actor():
            actor = StochasticActor([tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                                     tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                                     tf.keras.layers.Dense(action_size * 2, kernel_initializer=orthogonal_init)])
            return actor
    elif algo == 'TD3':
        def make_actor():
            actor = Actor([tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                           tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                           tf.keras.layers.Dense(action_size, kernel_initializer=orthogonal_init)])
            return actor
    elif algo == 'RTTD3':
        assert dec_embedding_dim % max_plan_len == 0, 'The decoder embedding dim needs to be partitioned in {} action' \
                                                      'embeddings'.format(max_plan_len)
        if rt_ae_type == 'recurrent':
            def make_rt_ae():
                enc_rnn_cells = [tfl.GRU(units=joint_enc_embedding_dim, return_sequences=True)]
                encoder = RoutineEncoder([tfl.Dense(enc_embedding_dim)],
                                         enc_rnn_cells,
                                         [tfl.Dense(rt_dim, activation='tanh')])

                decoder = SimpleFCRoutineDecoder(max_plan_len,
                                                 [tfl.Dense(dec_embedding_dim, activation='relu'),
                                                  tfl.Reshape([max_plan_len, dec_embedding_dim // max_plan_len]),
                                                  tfl.Dense(action_size + 1)])

                rt_ae = RoutineAE(encoder, decoder, plan_l2_reg=rt_l2_reg)
                return rt_ae
        elif rt_ae_type == 'denserecursive':
            def make_rt_ae():
                enc_rnn_cells = [DenseRecursiveSum(max_plan_len, joint_enc_embedding_dim)]
                encoder = RoutineEncoder([tfl.Dense(enc_embedding_dim)],
                                         enc_rnn_cells,
                                         [tfl.Dense(rt_dim, activation='tanh')])

                decoder = SimpleFCRoutineDecoder(max_plan_len,
                                                 [tfl.Dense(dec_embedding_dim, activation='relu'),
                                                  tfl.Reshape([max_plan_len, dec_embedding_dim // max_plan_len]),
                                                  tfl.Dense(action_size + 1)])

                rt_ae = RoutineAE(encoder, decoder, plan_l2_reg=rt_l2_reg)
                return rt_ae
        else:
            raise NotImplementedError

        def make_actor():
            actor = RTActor([tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                             tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                             tf.keras.layers.Dense(rt_dim, kernel_initializer=orthogonal_init)])
            return actor
    elif algo == 'RTSAC':
        assert dec_embedding_dim % max_plan_len == 0
        if rt_ae_type == 'recurrent':
            def make_rt_ae():
                enc_rnn_cells = [tfl.GRU(units=joint_enc_embedding_dim, return_sequences=True)]
                encoder = RoutineEncoder([tfl.Dense(enc_embedding_dim)],
                                         enc_rnn_cells,
                                         [tfl.Dense(rt_dim, activation='tanh')])

                decoder = SimpleStochasticFCRoutineDecoder(max_plan_len,
                                                           [tfl.Dense(dec_embedding_dim, activation='relu'),
                                                            tfl.Reshape(
                                                                [max_plan_len, dec_embedding_dim // max_plan_len]),
                                                            tfl.Dense(action_size + action_size + 1)])

                rt_ae = StochasticRoutineAE(encoder, decoder, plan_l2_reg=rt_l2_reg)
                return rt_ae
        elif rt_ae_type == 'denserecursive':
            def make_rt_ae():
                enc_rnn_cells = [DenseRecursiveSum(max_plan_len, joint_enc_embedding_dim)]
                encoder = RoutineEncoder([tfl.Dense(enc_embedding_dim)],
                                         enc_rnn_cells,
                                         [tfl.Dense(rt_dim, activation='tanh')])

                decoder = SimpleStochasticFCRoutineDecoder(max_plan_len,
                                                           [tfl.Dense(dec_embedding_dim, activation='relu'),
                                                            tfl.Reshape(
                                                                [max_plan_len, dec_embedding_dim // max_plan_len]),
                                                            tfl.Dense(action_size + action_size + 1)])

                rt_ae = StochasticRoutineAE(encoder, decoder, plan_l2_reg=rt_l2_reg)
                return rt_ae
        else:
            raise NotImplementedError

        def make_actor():
            actor = RTActor([tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                             tfl.Dense(actor_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                             tf.keras.layers.Dense(rt_dim, kernel_initializer=orthogonal_init)])
            return actor
    else:
        raise NotImplementedError

    def make_critic():
        critic = Critic([tf.keras.layers.Dense(critic_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                         tf.keras.layers.Dense(critic_layers_dim, 'relu', kernel_initializer=orthogonal_init),
                         tf.keras.layers.Dense(1, kernel_initializer=orthogonal_init)])
        return critic

    sampler = Sampler(env, eval_env, episode_limit=episode_limit,
                      init_random_samples=initial_random_samples)

    print('Max episode length - {}'.format(sampler._max_episode_steps))

    replay_buffer = FastReplayBuffer(buffer_size, visual_data=False)

    rt_ae_optimizer = tf.keras.optimizers.Adam(rt_ae_lr, beta_1=rt_ae_beta1)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr, beta_1=actor_beta1)
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr, beta_1=critic_beta1)
    entropy_optimizer = tf.keras.optimizers.Adam(entropy_coeff_lr, beta_1=entropy_coeff_beta1)

    if algo == 'SAC':
        agent = SAC(make_actor=make_actor,
                    make_critic=make_critic,
                    make_critic2=make_critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    entropy_optimizer=entropy_optimizer,
                    gamma=gamma,
                    q_polyak=q_polyak,
                    entropy_coefficient=initial_entropy_coefficient,
                    tune_entropy_coefficient=tune_entropy_coefficient,
                    target_entropy=target_entropy,
                    clip_actor_gradients=clip_actor_gradients, )
    elif algo == 'TD3':
        agent = DDPG(make_actor=make_actor,
                     make_critic=make_critic, make_critic2=make_critic,
                     actor_optimizer=actor_optimizer,
                     critic_optimizer=critic_optimizer,
                     gamma=gamma,
                     q_polyak=q_polyak,
                     train_actor_noise=q_target_noise,
                     max_train_actor_noise=max_q_target_noise,
                     clip_actor_gradients=clip_actor_gradients, )
    elif algo == 'RTTD3':
        agent = RTDDPG(
            make_routine_ae=make_rt_ae,
            make_actor=make_actor,
            make_critic=make_critic, make_critic2=make_critic,

            use_target_actor=use_target_actor,
            use_min_q_actor=use_min_q_actor,

            routine_ae_optimizer=rt_ae_optimizer,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=gamma,
            q_polyak=q_polyak,
            train_actor_noise=q_target_noise,
            max_train_actor_noise=max_q_target_noise,
            clip_actor_gradients=clip_actor_gradients,
            max_plan_len=max_plan_len,
            routine_ae_warmup=rt_ae_warmup,
            clip_td_backup=clip_td_backup,
            huber_td_loss=huber_td_loss,
            routine_sampling_noise=rt_sampling_noise,
            target_routine_ae=targ_rt_ae,
            ema_routine_ae=ema_rt_ae,
            routine_ae_polyak=rt_ae_polyak,
            routine_ae_critic_optim=joint_rt_ae_critic_optim,
            double_joint_routine_ae_critic_optim=double_joint_rt_ae_critic_optim,
            joint_routine_ae_actor_optim=joint_rt_ae_actor_optim,
            routine_ae_recon_optim=rt_ae_recon_optim,
            routine_ae_actor_consistency_loss=rt_ae_actor_consistency_loss, )
    elif algo == 'RTSAC':
        agent = RTSAC(
            make_routine_ae=make_rt_ae,
            make_actor=make_actor,
            make_critic=make_critic,
            make_critic2=make_critic,
            routine_ae_optimizer=rt_ae_optimizer,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            entropy_optimizer=entropy_optimizer,
            gamma=gamma,
            q_polyak=q_polyak,
            entropy_coefficient=initial_entropy_coefficient,
            tune_entropy_coefficient=tune_entropy_coefficient,
            target_entropy=target_entropy,
            approximate_current_entropy=approximate_current_entropy,
            lower_bound_q_target_entropy=lower_bound_q_target_entropy,
            clip_actor_gradients=clip_actor_gradients,
            max_plan_len=max_plan_len,
            routine_ae_warmup=rt_ae_warmup,
            clip_td_backup=clip_td_backup,
            huber_td_loss=huber_td_loss,
            routine_ae_critic_optim=joint_rt_ae_critic_optim,
            double_joint_routine_ae_critic_optim=double_joint_rt_ae_critic_optim,
            joint_routine_ae_actor_optim=joint_rt_ae_actor_optim,
            routine_ae_recon_optim=rt_ae_recon_optim,
            routine_ae_actor_consistency_loss=rt_ae_actor_consistency_loss, )

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
    use_plan_buffer = training_params.get('use_plan_buffer', True)
    drop_plan_chance = training_params.get('drop_plan_chance', None)
    use_test_plan_buffer = training_params.get('use_test_plan_buffer', True)

    algo = agent_params.get('algo', 'SAC')
    description_prefix = 'state'

    if algo == 'RTSAC' or algo == 'RTTD3':
        max_plan_len = agent_params.get('max_plan_len', 4)
        brief_exp_description = '{}_{}_{}b_{}ml'.format(
            algo, description_prefix, batch_size, max_plan_len)
    else:
        use_plan_buffer = False
        use_test_plan_buffer = False
        brief_exp_description = '{}_{}_{}b'.format(
            algo, description_prefix, batch_size)

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
    evaluation_stats = sampler.evaluate(agent, test_runs_per_epoch, log=False, use_plan_buffer=use_test_plan_buffer)
    for k, v in evaluation_stats.items():
        logz.log_tabular(k, v)
    mean_test_returns.append(evaluation_stats['episode_returns_mean'])
    mean_test_std.append(evaluation_stats['episode_returns_std'])
    steps.append(step_counter)
    epoch_end_eval_time = time.time()
    evaluation_time = epoch_end_eval_time - start_training_time

    if use_plan_buffer and not use_test_plan_buffer:
        evaluation_stats = sampler.evaluate(agent, test_runs_per_epoch, log=False, use_plan_buffer=use_plan_buffer)
        for k, v in evaluation_stats.items():
            logz.log_tabular('pb_{}'.format(k), v)

    logz.log_tabular('training_time', 0.0)
    logz.log_tabular('evaluation_time', np.around(evaluation_time, decimals=3))
    logz.dump_tabular()
    for e in range(epochs):
        logz.log_tabular('epoch', e + 1)
        epoch_start_time = time.time()
        while step_counter < (e + 1) * steps_per_epoch:
            traj_data = sampler.sample_steps(agent, action_sampling_noise, n_steps=1, use_plan_buffer=use_plan_buffer,
                                             drop_plan=drop_plan_chance)
            replay_buffer.add(traj_data)
            step_counter += traj_data['n']
            if step_counter > start_training:
                agent.train(replay_buffer, batch_size=batch_size,
                            n_updates=updates_per_step * traj_data['n'],
                            act_delay=actor_delay, )
            elif step_counter == start_training:
                agent.train(replay_buffer, batch_size=batch_size,
                            n_updates=updates_per_step * start_training,
                            act_delay=actor_delay, )

        logz.log_tabular('number_collected_observations', step_counter)
        epoch_end_training_time = time.time()
        training_time = epoch_end_training_time - epoch_start_time
        print('Epoch {}/{} - total steps {}'.format(e + 1, epochs, step_counter))
        evaluation_stats = sampler.evaluate(agent, test_runs_per_epoch, log=False, use_plan_buffer=use_test_plan_buffer)
        for k, v in evaluation_stats.items():
            logz.log_tabular(k, v)
        mean_test_returns.append(evaluation_stats['episode_returns_mean'])
        mean_test_std.append(evaluation_stats['episode_returns_std'])
        steps.append(step_counter)
        epoch_end_eval_time = time.time()
        evaluation_time = epoch_end_eval_time - epoch_end_training_time

        if use_plan_buffer and not use_test_plan_buffer:
            evaluation_stats = sampler.evaluate(agent, test_runs_per_epoch, log=False, use_plan_buffer=use_plan_buffer)
            for k, v in evaluation_stats.items():
                logz.log_tabular('pb_{}'.format(k), v)

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
