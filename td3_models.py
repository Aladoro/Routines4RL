import numpy as np
import tensorflow as tf
from sac_models import StochasticActor

tfl = tf.keras.layers
LN4 = np.log(4)


class Actor(tf.keras.layers.Layer):
    """Deterministic policy model."""
    def __init__(self, layers, norm_mean=None, norm_stddev=None):
        super(Actor, self).__init__()
        self._act_layers = layers
        self._out_dim = layers[-1].units
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev

    def call(self, inputs):
        out = inputs
        for layer in self._act_layers:
            out = layer(out)
        return out

    def get_action(self, observation_batch, noise_stddev, clip_noise=False, max_noise=0.5, **kwargs):
        batch_dim = tf.shape(observation_batch)[:-1]
        pre_obs_batch = self.preprocess_obs(observation_batch)
        if noise_stddev > 0.0:
            noise = tf.random.normal(shape=tf.concat((batch_dim, [self._out_dim]), axis=0),
                                     stddev=noise_stddev)
            if clip_noise:
                noise = tf.clip_by_value(noise, -max_noise, max_noise)
        else:
            noise = 0.0
        return tf.clip_by_value(self.__call__(pre_obs_batch) + noise, -1.0, 1.0)

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


class DDPG(tf.keras.Model):
    """Implementation of Twin Delayed DDPG algorithm

    References
    ----------
    Fujimoto, Scott, Herke Hoof, and David Meger. "Addressing function approximation error in actor-critic methods."
    International Conference on Machine Learning. PMLR, 2018.
    """
    def __init__(self, make_actor, make_critic, make_critic2=None,
                 actor_optimizer=tf.keras.optimizers.Adam(1e-3),
                 critic_optimizer=tf.keras.optimizers.Adam(1e-3), gamma=0.99,
                 q_polyak=0.995, train_actor_noise=0.1, max_train_actor_noise=0.5,
                 clip_actor_gradients=True,
                 entropy_bonus=None,
                 log_losses_every=-1, **kwargs):
        """
        Parameters
        ----------
        make_actor : Function outputting the policy model.
        make_critic : Function outputting the first Q-function model.
        make_critic2 : Function outputting the second Q-function model for double Q-learning, optional.
        actor_optimizer : Optimizer for policy model, default is Adam.
        critic_optimizer : Optimizer for Q-function model, default is Adam.
        gamma : Discount factor, default is 0.99.
        polyak : Polyak update coefficient for target models, default is 0.995.
        train_actor_noise : Noise to utilize for target policy smoothing, default is 0.1.
        clip_actor_gradients : Clip gradients for the policy parameters, default is True.
        """
        initialize_model = kwargs.get('initialize_model', True)
        if initialize_model:
            super(DDPG, self).__init__()
            self._log_losses_every = log_losses_every
            self._log_losses = False
        self.actor_opt = actor_optimizer
        self.critic_opt = critic_optimizer

        self._entropy_bonus = entropy_bonus

        self._act = make_actor()

        if self._entropy_bonus is not None:
            assert isinstance(self._act, StochasticActor), 'if an entropy bonus is utilized, ' \
                                                           'the actor used should be a StochasticActor'

        self._cri = make_critic()
        self._targ_act = make_actor()
        self._targ_cri = make_critic()
        self._clip_actor_gradients = clip_actor_gradients
        self._train_actor_noise = train_actor_noise
        self._max_train_actor_noise = max_train_actor_noise
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
        self._train_act = self.make_actor_train_op()
        self._targ_act_update = self.make_target_update_op(self._act,
                                                           self._targ_act, q_polyak)
        self._training_steps = 0

    def call(self, inputs):
        out = {}
        out['act'] = self._act.get_action(inputs, 0.0)
        out['t_act'] = self._targ_act.get_action(inputs, 0.0)
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
        return self._act.get_action(observation_batch, noise_stddev, max_noise)

    def train(self, buffer, batch_size=128, n_updates=1, act_delay=2, **kwargs):
        for i in range(n_updates):
            self._training_steps += 1
            b = buffer.get_random_batch(batch_size)
            (observations, actions, next_observations, rewards, done_mask) = (
                b['obs'], b['act'], b['nobs'], b['rew'], b['don'])
            if self._training_steps % act_delay == 0:
                self.run_full_training(observations, actions, next_observations, rewards, done_mask)
            else:
                self.run_delayed_training(observations, actions, next_observations, rewards, done_mask)

    @tf.function
    def run_full_training(self, observations, actions, next_observations, rewards, done_mask):
        done_mask = tf.cast(done_mask, tf.float32)
        q1_loss = self._train_cri(observations, actions, next_observations,
                                  rewards, done_mask)
        if self._double_q:
            q2_loss = self._train_cri2(observations, actions,
                                       next_observations, rewards,
                                       done_mask)
        actor_loss = self._train_act(observations)
        self._targ_act_update()
        self._targ_cri_update()
        if self._double_q:
            self._targ_cri2_update()

        return q1_loss, q2_loss, actor_loss

    @tf.function
    def run_delayed_training(self, observations, actions, next_observations, rewards, done_mask):
        done_mask = tf.cast(done_mask, tf.float32)
        q1_loss = self._train_cri(observations, actions, next_observations,
                                  rewards, done_mask)
        if self._double_q:
            q2_loss = self._train_cri2(observations, actions,
                                       next_observations, rewards,
                                       done_mask)
        return q1_loss, q2_loss


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
            maximizing_action = self._targ_act.get_action(next_observation_batch,
                                                          self._train_actor_noise,
                                                          clip_noise=True,
                                                          max_noise=self._max_train_actor_noise)
            targ_q = q_estimator(next_observation_batch, maximizing_action)
            targets = tf.reshape(reward_batch, [-1, 1]) + tf.reshape(
                1 - done_mask, [-1, 1]) * discount * targ_q
            with tf.GradientTape() as tape:
                loss = 0.5 * tf.reduce_mean(tf.square(
                    critic.get_q(observation_batch, action_batch) - targets))
            gradients = tape.gradient(loss, self.get_critic_trainable_variables(critic))
            self.critic_opt.apply_gradients(zip(gradients, self.get_critic_trainable_variables(critic)))
            return loss

        return tf.function(train)

    def make_actor_train_op(self, ):
        if self._entropy_bonus is not None:
            def get_actor_loss(observation_batch):
                actions, log_probs = self._act.get_action_probability(observation_batch)
                advantage = self._cri.get_q(observation_batch, actions)
                return self._entropy_bonus*tf.reduce_mean(log_probs)-tf.reduce_mean(advantage)
        else:
            def get_actor_loss(observation_batch):
                actions = self._act.get_action(observation_batch, 0.0)
                advantage = self._cri.get_q(observation_batch, actions)
                return -1*tf.reduce_mean(advantage)
        def train(observation_batch):
            with tf.GradientTape() as tape:
                loss = get_actor_loss(observation_batch)
            gradients = tape.gradient(loss, self.get_actor_trainable_variables())
            if self._clip_actor_gradients:
                gradients, _ = tf.clip_by_global_norm(gradients, 40)
            self.actor_opt.apply_gradients(zip(gradients, self.get_actor_trainable_variables()))
            return loss
        return tf.function(train)

    def make_target_update_op(self, model, target_model, polyak):
        def update_target():
            critic_weights = model.trainable_weights
            target_weights = target_model.trainable_weights
            for c_w, t_w in zip(critic_weights, target_weights):
                t_w.assign((polyak) * t_w + (1 - polyak) * c_w)

        return tf.function(update_target)

    def get_critic_trainable_variables(self, critic):
        return critic.trainable_variables

    def get_actor_trainable_variables(self, ):
        return self._act.trainable_variables

