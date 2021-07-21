import numpy as np
import tensorflow as tf
from td3_models import DDPG

tfl = tf.keras.layers
tfm = tf.keras.models
LN4 = np.log(4)


def huber_loss(y_true, y_pred, delta=1.0):
    abs_diffs = tf.abs(y_true - y_pred)
    return tf.where(abs_diffs > delta, x=delta * abs_diffs, y=tf.square(abs_diffs))


def huber_scaling(errors, delta=1.0):
    abs_diffs = tf.abs(errors)
    return tf.where(abs_diffs > delta, x=delta * abs_diffs, y=tf.square(abs_diffs))


class FiGARActor(tf.keras.layers.Layer):
    """FiGAR policy model."""

    def __init__(self, layers, possible_repetitions, norm_mean=None, norm_stddev=None):
        super(FiGARActor, self).__init__()
        self._act_layers = layers
        self._out_dim = layers[-1].units
        self._possible_repetitions = possible_repetitions
        self._repetition_dims = len(possible_repetitions)
        self._action_dim = self._out_dim - self._repetition_dims

        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev

    def call(self, inputs):
        out = inputs
        for layer in self._act_layers:
            out = layer(out)
        return out

    def get_action_and_rep_logits(self, observation_batch, noise_stddev, clip_noise=False, max_noise=0.5, **kwargs):
        batch_dim = tf.shape(observation_batch)[:-1]
        pre_obs_batch = self.preprocess_obs(observation_batch)
        raw_action, rep_logits = tf.split(self.__call__(pre_obs_batch),
                                          [self._action_dim, self._repetition_dims],
                                          axis=-1)

        action = tf.nn.tanh(raw_action)
        if noise_stddev > 0.0:
            noise = tf.random.normal(shape=tf.concat((batch_dim, [self._action_dim]), axis=0),
                                     stddev=noise_stddev)
            noise_upper_limit = 1.0 - action
            noise_lower_limit = -1.0 - action
            if clip_noise:
                noise_upper_limit = tf.minimum(noise_upper_limit, max_noise)
                noise_lower_limit = tf.maximum(noise_lower_limit, -max_noise)
            noise = tf.clip_by_value(noise, clip_value_min=noise_lower_limit,
                                     clip_value_max=noise_upper_limit)
        else:
            noise = 0.0
        return action + noise, rep_logits

    def preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class StandardFiGARCritic(tf.keras.layers.Layer):
    """FiGAR policy model using the original parameters from FiGAR-DDPG."""

    def __init__(self, observation_dims):
        super(StandardFiGARCritic, self).__init__()
        self._cri_layers = [tfl.Dense(300, activation='relu'),
                            tfl.Dense(600, activation='relu'),
                            tfl.Dense(600, activation='relu'),
                            tfl.Dense(1)]
        self._observation_dims = observation_dims

    def call(self, inputs):
        observations, actions = inputs
        obs_rep = self._cri_layers[0](observations)
        out = tf.concat([obs_rep, actions], axis=-1)
        for layer in self._cri_layers[1:]:
            out = layer(out)
        return out

    def get_q(self, observation_batch, action_batch):
        return self.__call__([observation_batch, action_batch])


class StandardFiGARActor(FiGARActor):
    def __init__(self, action_dim, ):
        possible_repetitions = list(range(1, 16))
        layers = [tfl.Dense(300, 'relu'),
                  tfl.Dense(600, 'relu'),
                  tfl.Dense(action_dim + 15)]
        super(StandardFiGARActor, self).__init__(layers=layers, possible_repetitions=possible_repetitions)


class FiGARDDPG(DDPG):
    """Implementation of FiGAR-DDPG/FiGAR-TD3 algorithm

    References
    ----------
    Lakshminarayanan, Aravind, Sharma, Sahil, and Ravindran, Balaraman. "Dynamic action repetition for deep reinforcement learning".
    In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, 2017.
    """

    def __init__(self,
                 possible_repetitions,
                 make_actor,
                 make_critic,
                 make_critic2=None,
                 actor_optimizer=tf.keras.optimizers.Adam(1e-3),
                 critic_optimizer=tf.keras.optimizers.Adam(1e-3),
                 gamma=0.99,
                 q_polyak=0.995,
                 train_actor_noise=0.1,
                 max_train_actor_noise=0.5,
                 clip_actor_gradients=True,
                 use_min_q_act_opt=False,
                 **kwargs):
        initialize_model = kwargs.get('initialize_model', True)
        if initialize_model:
            tfm.Model.__init__(self, )
        self._possible_repetitions = possible_repetitions
        self._num_possible_repetitions = len(possible_repetitions)
        self._repetitions_tensor = tf.constant(self._possible_repetitions)
        self._use_min_q_act_opt = use_min_q_act_opt
        DDPG.__init__(self, make_actor=make_actor,
                      make_critic=make_critic,
                      make_critic2=make_critic2,
                      actor_optimizer=actor_optimizer,
                      critic_optimizer=critic_optimizer,
                      gamma=gamma,
                      q_polyak=q_polyak,
                      train_actor_noise=train_actor_noise,
                      max_train_actor_noise=max_train_actor_noise,
                      clip_actor_gradients=clip_actor_gradients,
                      initialize_model=False)

    def call(self, inputs):
        out = {}
        out['act'], out['rep_logits'] = self._act.get_action_and_rep_logits(inputs, 0.0)
        out['t_act'], out['t_rep_logits'] = self._targ_act.get_action_and_rep_logits(inputs, 0.0)
        q_act_feed = tf.concat([out['act'], tf.nn.softmax(out['rep_logits'], axis=-1)], axis=-1)
        if self._double_q:
            out['q'] = tf.minimum(self._cri.get_q(inputs, q_act_feed),
                                  self._cri2.get_q(inputs, q_act_feed))
            out['t_q'] = tf.minimum(self._targ_cri.get_q(inputs, q_act_feed),
                                    self._targ_cri2.get_q(inputs, q_act_feed))
        else:
            out['q'] = self._cri.get_q(inputs, q_act_feed)
            out['t_q'] = self._targ_cri.get_q(inputs, q_act_feed)
        return out

    def get_action_and_reps(self, observation_batch, noise_stddev, max_noise=0.5, epsilon=0.2):
        eps_tensor = tf.convert_to_tensor(epsilon, dtype=tf.float32)
        return self._internal_get_action_reps(observation_batch=observation_batch, noise_stddev=noise_stddev,
                                              epsilon=eps_tensor, max_noise=max_noise)

    @tf.function
    def _internal_get_action_reps(self, observation_batch, noise_stddev, epsilon, max_noise=0.5):
        batch_size = tf.shape(observation_batch)[0]
        action, rep_logits = self._act.get_action_and_rep_logits(observation_batch, noise_stddev, max_noise)
        u_sample = tf.random.uniform(shape=[batch_size])
        rep_sample = tf.reshape(tf.random.categorical(logits=rep_logits, num_samples=1, dtype=tf.int32), [batch_size])
        uni_sample = tf.random.uniform(shape=[batch_size], maxval=self._num_possible_repetitions, dtype=tf.int32)
        rep_indices = tf.where(u_sample > epsilon, x=rep_sample, y=uni_sample)
        repetitions = tf.gather(self._repetitions_tensor, indices=rep_indices)
        return action, repetitions, rep_indices

    def train(self, buffer, batch_size=128, n_updates=1, act_delay=2, **kwargs):
        for i in range(n_updates):
            self._training_steps += 1
            b = buffer.get_random_batch(batch_size)
            (observations, actions, repetitions, next_observations, rewards, done_mask) = (
                b['obs'], b['act'], b['reps'], b['nobs'], b['rew'], b['don'])
            if self._training_steps % act_delay == 0:
                self.run_full_training(observations, actions, repetitions, next_observations, rewards, done_mask)
            else:
                self.run_delayed_training(observations, actions, repetitions, next_observations, rewards, done_mask)

    @tf.function
    def run_full_training(self, observations, actions, repetitions, next_observations, rewards, done_mask):
        done_mask = tf.cast(done_mask, tf.float32)
        q1_loss = self._train_cri(observations, actions, repetitions,
                                  next_observations, rewards, done_mask)
        if self._double_q:
            q2_loss = self._train_cri2(observations, actions, repetitions,
                                       next_observations, rewards,
                                       done_mask)
        actor_loss = self._train_act(observations)
        self._targ_act_update()
        self._targ_cri_update()
        if self._double_q:
            self._targ_cri2_update()

        if self._double_q:
            return q1_loss, q2_loss, actor_loss
        return q1_loss, actor_loss

    @tf.function
    def run_delayed_training(self, observations, actions, repetitions, next_observations, rewards, done_mask):
        done_mask = tf.cast(done_mask, tf.float32)
        q1_loss = self._train_cri(observations, actions, repetitions,
                                  next_observations, rewards, done_mask)
        if self._double_q:
            q2_loss = self._train_cri2(observations, actions, repetitions,
                                       next_observations, rewards,
                                       done_mask)
        if self._double_q:
            return q1_loss, q2_loss
        return q1_loss

    def make_critic_train_op(self, critic, discount):
        if self._double_q:
            def q_estimator(observations, actions):
                q_1 = self._targ_cri.get_q(observations, actions)
                q_2 = self._targ_cri2.get_q(observations, actions)
                return tf.minimum(q_1, q_2)
        else:
            def q_estimator(observations, actions):
                return self._targ_cri.get_q(observations, actions)

        def train(observation_batch, action_batch, repetition_batch, next_observation_batch,
                  reward_batch, done_mask):
            maximizing_action, max_repetition_logits = self._targ_act.get_action_and_rep_logits(
                next_observation_batch, self._train_actor_noise, clip_noise=True,
                max_noise=self._max_train_actor_noise)
            max_repetition_probs = tf.nn.softmax(max_repetition_logits, axis=-1)

            targ_q_act_inputs = tf.concat([maximizing_action, max_repetition_probs], axis=-1)
            targ_q = q_estimator(next_observation_batch, targ_q_act_inputs)

            targets = tf.reshape(reward_batch, [-1, 1]) + tf.reshape(
                1 - done_mask, [-1, 1]) * discount * targ_q

            one_hot_repetition_batch = tf.one_hot(indices=repetition_batch,
                                                  depth=self._num_possible_repetitions,
                                                  dtype=tf.float32)
            q_act_inputs = tf.concat([action_batch, one_hot_repetition_batch], axis=-1)
            with tf.GradientTape() as tape:
                loss = 0.5 * tf.reduce_mean(tf.square(
                    critic.get_q(observation_batch, q_act_inputs) - targets))
            gradients = tape.gradient(loss, self.get_critic_trainable_variables(critic))
            self.critic_opt.apply_gradients(zip(gradients, self.get_critic_trainable_variables(critic)))
            return loss

        return tf.function(train)

    def make_actor_train_op(self, ):
        if self._use_min_q_act_opt:
            def get_actor_loss(observation_batch):
                actions, max_repetition_logits = self._act.get_action_and_rep_logits(
                    observation_batch, 0.0)
                q_act_inputs = tf.concat([actions, tf.nn.softmax(max_repetition_logits, axis=-1)],
                                         axis=-1)
                advantage = self._cri.get_q(observation_batch, q_act_inputs)
                advantage2 = self._cri2.get_q(observation_batch, q_act_inputs)
                return -1 * tf.reduce_mean(tf.minimum(advantage, advantage2))
        else:
            def get_actor_loss(observation_batch):
                actions, max_repetition_logits = self._act.get_action_and_rep_logits(
                    observation_batch, 0.0)
                q_act_inputs = tf.concat([actions, tf.nn.softmax(max_repetition_logits, axis=-1)],
                                         axis=-1)
                advantage = self._cri.get_q(observation_batch, q_act_inputs)
                return -1 * tf.reduce_mean(advantage)

        def train(observation_batch):
            with tf.GradientTape() as tape:
                loss = get_actor_loss(observation_batch)
            gradients = tape.gradient(loss, self.get_actor_trainable_variables())
            if self._clip_actor_gradients:
                gradients, _ = tf.clip_by_global_norm(gradients, 40)
            self.actor_opt.apply_gradients(zip(gradients, self.get_actor_trainable_variables()))
            return loss

        return tf.function(train)
