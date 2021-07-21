import numpy as np
import tensorflow as tf

tfl = tf.keras.layers
LN4 = np.log(4)


class StochasticActor(tf.keras.layers.Layer):
    """Stochastic policy model."""

    def __init__(self, layers, norm_mean=None, norm_stddev=None, min_log_stddev=-10,
                 max_log_stddev=2):
        super(StochasticActor, self).__init__()
        self._act_layers = layers
        self._out_dim = layers[-1].units
        self._action_dim = self._out_dim // 2

        self._log_prob_offset = self._action_dim / 2 * np.log(np.pi * 2)

        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev
        self._min_log_stddev = min_log_stddev
        self._range_log_stddev = max_log_stddev - min_log_stddev

    def call(self, inputs):
        out = inputs
        for layer in self._act_layers:
            out = layer(out)
        mean, log_stddev = tf.split(out, 2, -1)
        scaled_log_stddev = self._min_log_stddev + (tf.tanh(log_stddev) + 1) / 2 * self._range_log_stddev
        stddev = tf.exp(scaled_log_stddev)
        return mean, stddev

    def get_action(self, observation_batch, noise_stddev, *args, **kwargs):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        mean, stddev = self.__call__(pre_obs_batch)
        if noise_stddev == 0.0:
            return tf.tanh(mean)
        return tf.tanh(mean + tf.random.normal(tf.shape(mean)) * stddev)

    def get_action_probability(self, observation_batch):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        mean, stddev = self.__call__(pre_obs_batch)
        random_component = tf.random.normal(tf.shape(mean))
        raw_actions = mean + random_component * stddev
        actions = tf.tanh(raw_actions)
        log_probs = (-1 / 2 * tf.reduce_sum(tf.square(random_component), axis=-1) -
                     tf.reduce_sum(tf.math.log(stddev), axis=-1) - self._log_prob_offset)

        squash_features = -2 * raw_actions
        squash_correction = tf.reduce_sum(LN4 + squash_features - 2 * tf.math.softplus(squash_features), axis=1)
        log_probs -= squash_correction
        log_probs = tf.reshape(log_probs, [-1, 1])
        return actions, log_probs

    def get_probability(self, observation_batch, action_batch):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        mean, stddev = self.__call__(pre_obs_batch)
        raw_actions = 0.5 * tf.math.log((1 + action_batch) / (1 - action_batch))
        random_component = (raw_actions - mean) / stddev

        log_probs = (-1 / 2 * tf.reduce_sum(tf.square(random_component), axis=-1) -
                     tf.reduce_sum(tf.math.log(stddev), axis=-1) - self._log_prob_offset)

        squash_features = -2 * raw_actions
        squash_correction = tf.reduce_sum(LN4 + squash_features - 2 * tf.math.softplus(squash_features), axis=1)
        log_probs -= squash_correction
        log_probs = tf.reshape(log_probs, [-1, 1])
        return log_probs

    def preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class Critic(tf.keras.layers.Layer):
    """Simple Q-function model."""

    def __init__(self, layers, norm_mean=None, norm_stddev=None):
        super(Critic, self).__init__()
        self._cri_layers = layers
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev

    def call(self, inputs):
        out = inputs
        for layer in self._cri_layers:
            out = layer(out)
        return out

    def get_q(self, observation_batch, action_batch):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        input_batch = tf.concat([pre_obs_batch, action_batch], axis=-1)
        return self.__call__(input_batch)

    def preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class SAC(tf.keras.Model):
    """Implementation of Soft Actor-Critic algorithm

        References
        ----------
        Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
        """

    def __init__(self, make_actor, make_critic, make_critic2=None,
                 actor_optimizer=tf.keras.optimizers.Adam(1e-3),
                 critic_optimizer=tf.keras.optimizers.Adam(1e-3),
                 entropy_optimizer=tf.keras.optimizers.Adam(1e-4), gamma=0.99,
                 q_polyak=0.995, entropy_coefficient=0.1, tune_entropy_coefficient=False,
                 target_entropy=-6, clip_actor_gradients=True, **kwargs):
        """
        Parameters
        ----------
        make_actor : Function outputting the policy model.
        make_critic : Function outputting the first Q-function model.
        make_critic2 : Function outputting the second Q-function model for double Q-learning, optional.
        actor_optimizer : Optimizer for policy model, default is Adam.
        critic_optimizer : Optimizer for Q-function model, default is Adam.
        gamma : Discount factor, default is 0.99.
        q_polyak : Polyak update coefficient for Q-function target models, default is 0.995.
        entropy_coefficient : Starting SAC entropy coefficient, default is 0.1.
        tune_entropy_coefficient : Automatically tune entropy coefficient, default is False.
        target_entropy : Target entropy used when automatically tuning entropy coefficient, default is -6.
        clip_actor_gradients : Clip gradients for the policy parameters, default is True.
        """
        initialize_model = kwargs.get('initialize_model', True)
        if initialize_model:
            super(SAC, self).__init__()
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer
        self.entropy_opt = entropy_optimizer
        self._entropy_coefficient = entropy_coefficient
        self._tune_entropy_coefficient = tune_entropy_coefficient
        self._target_entropy = float(target_entropy)
        self._act = make_actor()
        self._cri = make_critic()
        self._targ_cri = make_critic()
        self._clip_actor_gradients = clip_actor_gradients
        if make_critic2 is not None:
            self._double_q = True
            self._cri2 = make_critic2()
            self._targ_cri2 = make_critic2()
            self._train_cri2 = self.make_critic_train_op(self._cri2, gamma)
            self._targ_cri2_update = self.make_target_update_op(self._cri2,
                                                                self._targ_cri2,
                                                                q_polyak)
        else:
            self._double_q = False

        self._train_cri = self.make_critic_train_op(self._cri, gamma)
        self._targ_cri_update = self.make_target_update_op(self._cri,
                                                           self._targ_cri, q_polyak)
        self._train_act_and_alpha = self.make_actor_and_alpha_train_op()

        if self._tune_entropy_coefficient:
            self._log_alpha = tf.Variable(tf.math.log(
                self._entropy_coefficient), trainable=True)
        else:
            self._log_alpha = tf.Variable(tf.math.log(
                self._entropy_coefficient), trainable=False)

        self._training_steps = 0

    def call(self, inputs):
        out = {}
        out['act'] = self._act.get_action(inputs, noise_stddev=0.0)
        if self._double_q:
            out['q'] = tf.minimum(self._cri.get_q(inputs, out['act']),
                                  self._cri2.get_q(inputs, out['act']))
            out['t_q'] = tf.minimum(self._targ_cri.get_q(inputs, out['act']),
                                    self._targ_cri2.get_q(inputs, out['act']))
        else:
            out['q'] = self._cri.get_q(inputs, out['act'])
            out['t_q'] = self._targ_cri.get_q(inputs, out['act'])
        return out

    @tf.function
    def get_action(self, observation_batch, noise_stddev, max_noise=0.5):
        return self._act.get_action(observation_batch, noise_stddev=noise_stddev, max_noise=max_noise)

    def train(self, buffer, batch_size=128, n_updates=1, act_delay=1, **kwargs):
        for _ in range(n_updates):
            self._training_steps += 1

            b = buffer.get_random_batch(batch_size)
            (observations, actions, next_observations, rewards, done_mask) = (
                b['obs'], b['act'], b['nobs'], b['rew'], tf.cast(b['don'], tf.float32))
            if self._training_steps % act_delay == 0:
                losses = self.run_full_training(observations, actions, next_observations, rewards, done_mask)
            else:
                losses = self.run_delayed_training(observations, actions, next_observations, rewards, done_mask)

    @tf.function
    def run_full_training(self, observations, actions, next_observations, rewards, done_mask):
        loss_critic = self._train_cri(observations, actions, next_observations,
                                      rewards, done_mask)
        if self._double_q:
            loss_critic2 = self._train_cri2(observations, actions,
                                            next_observations, rewards,
                                            done_mask)
        loss_actor, loss_alpha = self._train_act_and_alpha(observations)
        self._targ_cri_update()
        if self._double_q:
            self._targ_cri2_update()
        if self._double_q:
            return loss_critic, loss_critic2, loss_actor, loss_alpha
        else:
            return loss_critic, loss_actor, loss_alpha

    @tf.function
    def run_delayed_training(self, observations, actions, next_observations, rewards, done_mask):
        loss_critic = self._train_cri(observations, actions, next_observations,
                                      rewards, done_mask)
        if self._double_q:
            loss_critic2 = self._train_cri2(observations, actions,
                                            next_observations, rewards,
                                            done_mask)

        if self._double_q:
            return loss_critic, loss_critic2
        else:
            return loss_critic

    def make_critic_train_op(self, critic, discount):
        if self._double_q:
            def q_estimator(observations, actions):
                q_1 = self._targ_cri.get_q(observations, actions)
                q_2 = self._targ_cri2.get_q(observations, actions)
                return tf.minimum(q_1, q_2)
        else:
            def q_estimator(observations, actions):
                return self._targ_cri.get_q(observations, actions)

        def train(observation_batch, action_batch, next_observation_batch,
                  reward_batch, done_mask):
            with tf.GradientTape() as tape:
                next_actions, next_log_probs = self._act.get_action_probability(
                    next_observation_batch)
                next_q = q_estimator(next_observation_batch, next_actions)
                targets = tf.reshape(reward_batch, [-1, 1]) + tf.reshape(
                    1 - done_mask, [-1, 1]) * discount * (
                                  next_q - tf.exp(self._log_alpha) * next_log_probs)
                loss = 0.5 * tf.reduce_mean(tf.square(
                    critic.get_q(observation_batch, action_batch) - tf.stop_gradient(targets)))
            gradients = tape.gradient(loss, self.get_critic_trainable_weights(critic))
            self.critic_opt.apply_gradients(zip(gradients, self.get_critic_trainable_weights(critic)))
            return loss

        return tf.function(train)

    def make_actor_and_alpha_train_op(self, ):
        if self._double_q:
            def q_estimator(observations, actions):
                q_1 = self._cri.get_q(observations, actions)
                q_2 = self._cri2.get_q(observations, actions)
                return tf.minimum(q_1, q_2)
        else:
            def q_estimator(observations, actions):
                return self._cri.get_q(observations, actions)
        if self._tune_entropy_coefficient:
            def step_alpha(log_probs):
                with tf.GradientTape() as tape:
                    loss = -tf.reduce_mean(self._log_alpha * tf.stop_gradient(
                        (log_probs + self._target_entropy)))
                    gradients = tape.gradient(loss, [self._log_alpha])
                self.entropy_opt.apply_gradients(zip(gradients, [self._log_alpha]))
                return loss
        else:
            def step_alpha(log_probs):
                return 0.0

        def train(observation_batch):
            with tf.GradientTape() as tape:
                actions, log_probs = self._act.get_action_probability(observation_batch)
                q_estimates = q_estimator(observation_batch, actions)
                actor_loss = tf.reduce_mean(tf.exp(self._log_alpha) * log_probs - q_estimates)
            gradients = tape.gradient(actor_loss, self.get_actor_trainable_weights())
            if self._clip_actor_gradients:
                gradients, _ = tf.clip_by_global_norm(gradients, 40)
            self.actor_opt.apply_gradients(zip(gradients, self.get_actor_trainable_weights()))
            alpha_loss = step_alpha(log_probs=log_probs)
            return actor_loss, alpha_loss

        return tf.function(train)

    def make_target_update_op(self, model, target_model, polyak):
        def update_target():
            critic_weights = model.trainable_weights
            target_weights = target_model.trainable_weights
            for c_w, t_w in zip(critic_weights, target_weights):
                t_w.assign((polyak) * t_w + (1 - polyak) * c_w)

        return tf.function(update_target)

    def make_alpha_train_op(self, ):
        if self._tune_entropy_coefficient:
            def train(log_probs):
                with tf.GradientTape() as tape:
                    loss = -tf.reduce_mean(self._log_alpha * tf.stop_gradient(
                        (log_probs + self._target_entropy)))
                    gradients = tape.gradient(loss, [self._log_alpha])
                self.entropy_opt.apply_gradients(zip(gradients, [self._log_alpha]))
                return loss
        else:
            def train(observation_batch):
                return 0.0
        return tf.function(train)

    def get_critic_trainable_weights(self, critic):
        return critic.trainable_weights

    def get_actor_trainable_weights(self, ):
        return self._act.trainable_weights
