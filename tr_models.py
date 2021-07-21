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


class RTActor(tf.keras.layers.Layer):
    """Deterministic routine policy model."""

    def __init__(self, layers, norm_mean=None, norm_stddev=None):
        super(RTActor, self).__init__()
        self._act_layers = layers
        self._out_dim = layers[-1].units
        self._norm_mean = norm_mean
        self._norm_stddev = norm_stddev

    def call(self, inputs):
        out = inputs
        for layer in self._act_layers:
            out = layer(out)
        return out

    def get_plan(self, observation_batch, noise_stddev, clip_noise=False, max_noise=0.5, **kwargs):
        batch_dim = tf.shape(observation_batch)[:-1]
        pre_obs_batch = self.preprocess_obs(observation_batch)
        raw_plan_rep = tf.nn.tanh(self.__call__(pre_obs_batch))
        if noise_stddev > 0.0:
            noise = tf.random.normal(shape=tf.concat((batch_dim, [self._out_dim]), axis=0),
                                     stddev=noise_stddev)
            noise_upper_limit = 1.0 - raw_plan_rep
            noise_lower_limit = -1.0 - raw_plan_rep
            if clip_noise:
                noise_upper_limit = tf.minimum(noise_upper_limit, max_noise)
                noise_lower_limit = tf.maximum(noise_lower_limit, -max_noise)
            noise = tf.clip_by_value(noise, clip_value_min=noise_lower_limit,
                                     clip_value_max=noise_upper_limit)
        else:
            noise = 0.0
        return raw_plan_rep + noise

    def preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class RTStochasticActor(tf.keras.layers.Layer):
    """Stochastic routine policy model."""

    def __init__(self, layers, norm_mean=None, norm_stddev=None, min_log_stddev=-10,
                 max_log_stddev=2):
        super(RTStochasticActor, self).__init__()
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

    def get_plan(self, observation_batch, noise_stddev, *args, **kwargs):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        mean, stddev = self.__call__(pre_obs_batch)
        if noise_stddev == 0.0:
            return tf.tanh(mean)
        return tf.tanh(mean + tf.random.normal(tf.shape(mean)) * stddev)

    def get_plan_probability(self, observation_batch):
        pre_obs_batch = self.preprocess_obs(observation_batch)
        mean, stddev = self.__call__(pre_obs_batch)
        random_component = tf.random.normal(tf.shape(mean))
        raw_actions = mean + random_component * stddev
        actions = tf.tanh(raw_actions)
        log_probs = (-1 / 2 * tf.reduce_sum(tf.square(random_component), axis=-1) -
                     tf.reduce_sum(tf.math.log(stddev), axis=-1) - self._log_prob_offset)

        squash_features = -2 * raw_actions
        squash_correction = tf.reduce_sum(LN4 + squash_features - 2 * tf.math.softplus(squash_features), axis=-1)
        log_probs -= squash_correction

        log_probs = tf.expand_dims(log_probs, axis=-1)
        return actions, log_probs

    def preprocess_obs(self, observation_batch):
        if self._norm_mean is not None and self._norm_stddev is not None:
            return (observation_batch - self.norm_mean) / (self.norm_stddev + 1e-7)
        return observation_batch


class RTDDPG(DDPG):
    """Implementation of the Routine-based Twin Delayed DDPG algorithm."""

    def __init__(self, make_routine_ae, make_actor, make_critic, make_critic2=None,
                 use_target_actor=True,
                 use_min_q_actor=False,
                 routine_ae_optimizer=tf.keras.optimizers.Adam(1e-3),
                 actor_optimizer=tf.keras.optimizers.Adam(1e-3),
                 critic_optimizer=tf.keras.optimizers.Adam(1e-3),
                 gamma=0.99,
                 q_polyak=0.995,
                 train_actor_noise=0.1,
                 max_train_actor_noise=0.5,
                 clip_actor_gradients=True,
                 max_plan_len=4,
                 routine_ae_warmup=0,
                 clip_td_backup=None,
                 huber_td_loss=False,
                 routine_sampling_noise=0.0,
                 target_routine_ae=False,
                 ema_routine_ae=False,
                 routine_ae_polyak=0.99,
                 routine_ae_critic_optim=False,
                 double_joint_routine_ae_critic_optim=False,
                 joint_routine_ae_actor_optim=False,
                 routine_ae_recon_optim=True,
                 routine_ae_actor_consistency_loss=None,
                 **kwargs):
        """
        Parameters
        ----------
        make_routine_ae: Function outputting the routine autoencoder model.
        make_actor: Function outputting the routine actor model.
        make_critic: Function outputting the routine critic model.
        make_critic2: Function outputting the second routine critic model.
        use_target_actor: Use a separate target actor for the TD-targets.
        use_min_q_actor: Use the minimum q value to optimize the actor.
        routine_ae_optimizer: Optimizer for the routine autoencoder model.
        actor_optimizer: Optimizer for the routine actor model.
        critic_optimizer: Optimizer for the routine critic model.
        gamma: Discount factor.
        q_polyak: Polyak coefficient for the target Q-networks (and target actor network if use_target_actor=True).
        train_actor_noise: Standard deviation for the policy smoothing noise of the TD-targets.
        max_train_actor_noise: Maximum policy smoothing noise of the TD-targets.
        clip_actor_gradients: Clip the magnitude of the actor network gradient.
        max_plan_len: Maximum number of actions each routine can map to (L).
        routine_ae_warmup: Number of routine autoencoder pre-training using a reconstruction loss.
        clip_td_backup: Clip the TD-targets magnitude to some maximum value.
        huber_td_loss: Utilize a huber TD-loss instead of MSE.
        routine_sampling_noise: Standard deviation of routine space exploration noise.
        target_routine_ae: Use a separate target routine autoencoder for the TD-targets.
        ema_routine_ae: Always use a routine autoencoder updated via exponentially moving average for the TD3 actor/critic losses.
        routine_ae_polyak: Polyak coefficient for the target routine autoencoder/ema routine autoencoder.
        routine_ae_critic_optim: Optimize the routine autoencoder end-to-end with the critic.
        double_joint_routine_ae_critic_optim: Optimize the routine autoencoder end-to-end with both critics.
        joint_routine_ae_actor_optim: Optimize the routine autoencoder end-to-end with the actor.
        routine_ae_recon_optim: Optimize the routine autoencoder with an additional L2 reconstruction optimization.
        routine_ae_actor_consistency_loss: Loss for the routine consistency many-to-one penalty.
        """

        initialize_model = kwargs.get('initialize_model', True)
        if initialize_model:
            tfm.Model.__init__(self, )
        self._use_target_actor = use_target_actor
        self._use_min_q_actor = use_min_q_actor

        self._rt_ae = make_routine_ae()
        self._target_rt_ae = target_routine_ae
        self._ema_rt_ae = ema_routine_ae

        self._joint_rt_ae_critic_optim = routine_ae_critic_optim
        self._double_joint_rt_ae_critic_optim = double_joint_routine_ae_critic_optim

        self._joint_rt_ae_actor_optim = joint_routine_ae_actor_optim

        self._rt_ae_actor_consistency_loss = routine_ae_actor_consistency_loss

        self._plan_ae_recon_optim = routine_ae_recon_optim

        self.rt_ae_opt = routine_ae_optimizer
        self._max_plan_len = max_plan_len

        self._rt_ae_update = self._rt_ae.make_rt_ae_train_op(self.rt_ae_opt)
        self._rt_ae_warmup = routine_ae_warmup
        self._perform_rt_ae_warmup = routine_ae_warmup > 0

        if self._target_rt_ae or self._ema_rt_ae:
            self._plan_ae_polyak = routine_ae_polyak
            self._targ_rt_ae = make_routine_ae()
            self._targ_rt_ae_update = self.make_target_update_op(self._rt_ae,
                                                                 self._targ_rt_ae,
                                                                 routine_ae_polyak)

        self._clip_td_backup = clip_td_backup
        self._huber_td_loss = huber_td_loss
        self._rt_sampling_noise = routine_sampling_noise
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
        if self._joint_rt_ae_critic_optim:
            self._train_rt_ae_and_cri = self.make_critic_and_rt_ae_train_op(self._cri,
                                                                            gamma)
            if self._double_joint_rt_ae_critic_optim:
                self._train_rt_ae_and_cri2 = self.make_critic_and_rt_ae_train_op(self._cri2,
                                                                                 gamma)

    def call(self, inputs):
        out = {}
        out['plan'] = self._act.get_plan(inputs, 0.0)
        out['t_plan'] = self._targ_act.get_plan(inputs, 0.0)
        out = {**out, **self._rt_ae(out['plan'])}
        if self._ema_rt_ae or self._target_rt_ae:
            out['t_plan_ae_dict'] = self._targ_rt_ae(out['t_plan'])
        if self._double_q:
            out['q'] = tf.minimum(self._cri.get_q(inputs, out['plan']),
                                  self._cri2.get_q(inputs, out['plan']))
            out['t_q'] = tf.minimum(self._targ_cri.get_q(inputs, out['plan']),
                                    self._targ_cri2.get_q(inputs, out['plan']))
        else:
            out['q'] = self._cri.get_q(inputs, out['plan'])
            out['t_q'] = self._targ_cri.get_q(inputs, out['plan'])
        return out

    def get_target_actor(self, ):
        if self._use_target_actor:
            return self._targ_act
        else:
            return self._act

    def get_target_rt_ae(self, ):
        if self._ema_rt_ae or self._target_rt_ae:
            return self._targ_rt_ae
        else:
            return self._rt_ae

    def get_rt_ae(self, ):
        if self._ema_rt_ae:
            return self._targ_rt_ae
        else:
            return self._rt_ae

    def get_action_plan(self, observation_batch, noise_stddev, max_noise=0.5):
        actions, end_seq = self.get_actions(observation_batch, noise_stddev, max_noise)
        return tf.unstack(actions[..., :end_seq[0], :], axis=-2)

    @tf.function
    def get_action(self, observation_batch, noise_stddev, max_noise=0.5):
        actions, _ = self.get_actions(observation_batch, noise_stddev, max_noise)
        return actions[:, 0, :]

    @tf.function
    def get_actions(self, observation_batch, noise_stddev, max_noise=0.5):
        if noise_stddev > 0.0:
            plan = self._act.get_plan(observation_batch, self._rt_sampling_noise)
            actions, end_seq = self.get_rt_ae().get_actions(plan)
            noise = tf.random.normal(shape=tf.shape(actions), stddev=noise_stddev)
            noise = tf.clip_by_value(noise, -max_noise, max_noise)
        else:
            plan = self._act.get_plan(observation_batch, 0.0)
            actions, end_seq = self.get_rt_ae().get_actions(plan)
            noise = 0.0
        return tf.clip_by_value(actions + noise, -1.0, 1.0), end_seq

    def train(self, buffer, batch_size=128, n_updates=1, act_delay=2, **kwargs):
        if self._perform_rt_ae_warmup:
            for i in range(self._rt_ae_warmup):
                b = buffer.get_n_steps_random_batch(batch_size, self._max_plan_len)
                actions_seq, ep_mask = b['acts'], tf.cast(b['mask'], tf.float32)
                _ = self._rt_ae_update(actions_seq, ep_mask)
            if self._ema_rt_ae or self._target_rt_ae:
                self._targ_rt_ae.set_weights(self._rt_ae.get_weights())
            self._perform_rt_ae_warmup = False
        for i in range(n_updates):
            self._training_steps += 1
            b = buffer.get_n_steps_random_batch(batch_size, self._max_plan_len)
            (observations, actions_seq, next_observations_seq, rewards_seq, done_seq, ep_mask) = (
                b['obs'], b['acts'], b['nobses'], b['rews'], b['dones'], b['mask'])
            if self._training_steps % act_delay == 0:
                self.run_full_training(observations, actions_seq, next_observations_seq, rewards_seq, done_seq, ep_mask)
            else:
                self.run_delayed_training(observations, actions_seq, next_observations_seq, rewards_seq, done_seq,
                                          ep_mask)

    @tf.function
    def run_full_training(self, observations, actions_seq, next_observations_seq, rewards_seq, done_seq, ep_mask):
        ep_mask = tf.cast(ep_mask, tf.float32)
        done_seq = tf.cast(done_seq, tf.float32)
        if self._joint_rt_ae_critic_optim:
            q1_loss, recon_loss, end_seq_loss, reg_loss = self._train_rt_ae_and_cri(observations,
                                                                                    actions_seq,
                                                                                    next_observations_seq,
                                                                                    rewards_seq, done_seq,
                                                                                    ep_mask)
        else:
            recon_loss, end_seq_loss, reg_loss = self._rt_ae_update(actions_seq, ep_mask)
            q1_loss = self._train_cri(observations, actions_seq, next_observations_seq,
                                      rewards_seq, done_seq, ep_mask)
        if self._double_q:
            if self._double_joint_rt_ae_critic_optim:
                q2_loss, recon2_loss, end_seq2_loss, reg2_loss = self._train_rt_ae_and_cri2(observations,
                                                                                            actions_seq,
                                                                                            next_observations_seq,
                                                                                            rewards_seq,
                                                                                            done_seq,
                                                                                            ep_mask)
            else:
                q2_loss = self._train_cri2(observations, actions_seq, next_observations_seq,
                                           rewards_seq, done_seq, ep_mask)

        else:
            q2_loss = 0.0

        actor_loss, cons_loss = self._train_act(observations)

        if self._target_rt_ae or self._ema_rt_ae:
            self._targ_rt_ae_update()
        if self._use_target_actor:
            self._targ_act_update()
        self._targ_cri_update()
        if self._double_q:
            self._targ_cri2_update()

        return recon_loss, end_seq_loss, reg_loss, q1_loss, q2_loss, actor_loss

    @tf.function
    def run_delayed_training(self, observations, actions_seq, next_observations_seq, rewards_seq, done_seq, ep_mask):
        ep_mask = tf.cast(ep_mask, tf.float32)
        done_seq = tf.cast(done_seq, tf.float32)
        if self._joint_rt_ae_critic_optim:
            q1_loss, recon_loss, end_seq_loss, reg_loss = self._train_rt_ae_and_cri(observations,
                                                                                    actions_seq,
                                                                                    next_observations_seq,
                                                                                    rewards_seq, done_seq,
                                                                                    ep_mask)
        else:
            recon_loss, end_seq_loss, reg_loss = self._rt_ae_update(actions_seq,
                                                                    ep_mask)
            q1_loss = self._train_cri(observations, actions_seq, next_observations_seq,
                                      rewards_seq, done_seq, ep_mask)
        if self._double_q:
            if self._double_joint_rt_ae_critic_optim:
                q2_loss, recon2_loss, end_seq2_loss, reg2_loss = self._train_rt_ae_and_cri2(observations,
                                                                                            actions_seq,
                                                                                            next_observations_seq,
                                                                                            rewards_seq,
                                                                                            done_seq,
                                                                                            ep_mask)
            else:
                q2_loss = self._train_cri2(observations, actions_seq, next_observations_seq,
                                           rewards_seq, done_seq, ep_mask)
        else:
            q2_loss = 0.0
        return recon_loss, end_seq_loss, reg_loss, q1_loss, q2_loss

    def make_rt_ae_train_op(self, ):
        def train(action_seq_batch, episode_mask):
            with tf.GradientTape() as tape:
                plan_representations, decoded_action_seq, decoded_end_seq, reconstruction_loss, end_seq_loss, reg_loss = (
                    self._rt_ae.get_masked_ae_losses(action_seq_batch, episode_mask))
                ae_loss = reconstruction_loss + end_seq_loss + reg_loss
            ae_gradients = tape.gradient(ae_loss, self._rt_ae.trainable_variables)
            self.rt_ae_opt.apply_gradients(zip(ae_gradients, self._rt_ae.trainable_variables))
            return reconstruction_loss, end_seq_loss, reg_loss

        return tf.function(train)

    def make_critic_train_op(self, critic, discount):
        reward_accumulation_mx = tf.linalg.band_part(tf.ones((self._max_plan_len, self._max_plan_len)), num_lower=0,
                                                     num_upper=-1) * tf.constant([[discount ** i] for i in
                                                                                  range(self._max_plan_len)])
        q_discounting_vector = tf.constant([[[discount ** (i + 1)] for i in range(self._max_plan_len)]])
        if self._double_q:
            def q_estimator(observations, plan_reps):
                q_1 = self._targ_cri.get_q(observations, plan_reps)
                q_2 = self._targ_cri2.get_q(observations, plan_reps)
                return tf.minimum(q_1, q_2)
        else:
            def q_estimator(observations, plan_reps):
                return self._targ_cri.get_q(observations, plan_reps)

        if self._clip_td_backup is not None:
            assert self._clip_td_backup > 0

            def process_td_backup(error):
                return tf.clip_by_value(error, clip_value_min=-self._clip_td_backup,
                                        clip_value_max=self._clip_td_backup)
        else:
            def process_td_backup(error):
                return error

        if self._huber_td_loss:
            def td_error(target, q_pred):
                return huber_scaling(process_td_backup(target) - q_pred)
        else:
            def td_error(target, q_pred):
                return tf.square(process_td_backup(target) - q_pred)

        def train(observation_batch, action_seq_batch, next_observation_seq_batch,
                  reward_seq_batch, done_seq_batch, episode_mask):
            next_max_plan_reps = self.get_target_actor().get_plan(next_observation_seq_batch,  # bs x max_n x obs_dim
                                                                  self._train_actor_noise,
                                                                  clip_noise=True,
                                                                  max_noise=self._max_train_actor_noise)

            decoded_next_max_plan_reps, _, _ = self.get_target_rt_ae().autoencode_plans_batch(next_max_plan_reps)
            targ_q = q_estimator(next_observation_seq_batch, decoded_next_max_plan_reps)
            discounted_rw_seq_batch = tf.expand_dims(
                tf.matmul(reward_seq_batch, reward_accumulation_mx), axis=-1)
            # b_s x max_n x 1
            discounted_targ_q = targ_q * q_discounting_vector
            # b_s x max_n x 1
            targets = discounted_rw_seq_batch + tf.expand_dims(
                1 - done_seq_batch, axis=-1) * discounted_targ_q
            # b_s x max_n x 1
            tiled_obs_batch = tf.tile(tf.expand_dims(observation_batch,  # b_s x obs_dim
                                                     axis=1), [1, self._max_plan_len, 1])
            # b_s x max_n x obs_dim
            plan_representations, decoded_action_seq, decoded_end_seq = self.get_rt_ae().autoencode_sequence(
                action_seq_batch)
            with tf.GradientTape(persistent=True) as tape:
                pred_q = critic.get_q(tiled_obs_batch, plan_representations)
                # b_s x max_n x 1
                masked_td_errors = td_error(target=targets, q_pred=pred_q) * tf.expand_dims(episode_mask, axis=-1)
                q_loss = 0.5 * tf.reduce_mean(masked_td_errors)

            q_gradients = tape.gradient(q_loss, self.get_critic_trainable_variables(critic))
            self.critic_opt.apply_gradients(zip(q_gradients, self.get_critic_trainable_variables(critic)))
            del tape
            return q_loss

        return tf.function(train)

    def make_critic_and_rt_ae_train_op(self, critic, discount):
        reward_accumulation_mx = tf.linalg.band_part(tf.ones((self._max_plan_len, self._max_plan_len)), num_lower=0,
                                                     num_upper=-1) * tf.constant([[discount ** i] for i in
                                                                                  range(self._max_plan_len)])
        # max_n x max_n * max_n * 1
        q_discounting_vector = tf.constant([[[discount ** (i + 1)] for i in range(self._max_plan_len)]])
        # 1 x max_n x 1
        if self._double_q:
            def q_estimator(observations, plan_reps):
                q_1 = self._targ_cri.get_q(observations, plan_reps)
                q_2 = self._targ_cri2.get_q(observations, plan_reps)
                return tf.minimum(q_1, q_2)
        else:
            def q_estimator(observations, plan_reps):
                return self._targ_cri.get_q(observations, plan_reps)
        if self._clip_td_backup is not None:
            assert self._clip_td_backup > 0

            def process_td_backup(error):
                return tf.clip_by_value(error, clip_value_min=-self._clip_td_backup,
                                        clip_value_max=self._clip_td_backup)
        else:
            def process_td_backup(error):
                return error

        if self._huber_td_loss:
            def td_error(target, q_pred):
                return huber_loss(process_td_backup(target), q_pred)
        else:
            def td_error(target, q_pred):
                return tf.square(process_td_backup(target) - q_pred)

        if self._plan_ae_recon_optim:
            def get_ae_loss(q_loss, reconstruction_loss, end_seq_loss, reg_loss):
                return q_loss + reconstruction_loss + end_seq_loss + reg_loss
        else:
            def get_ae_loss(q_loss, reconstruction_loss, end_seq_loss, reg_loss):
                return q_loss + end_seq_loss + reg_loss

        def train(observation_batch, action_seq_batch, next_observation_seq_batch,
                  reward_seq_batch, done_seq_batch, episode_mask):
            next_max_plan_reps = self.get_target_actor().get_plan(next_observation_seq_batch,  # bs x max_n x obs_dim
                                                                  self._train_actor_noise,
                                                                  clip_noise=True,
                                                                  max_noise=self._max_train_actor_noise)

            decoded_next_max_plan_reps, _, _ = self.get_target_rt_ae().autoencode_plans_batch(next_max_plan_reps)
            targ_q = q_estimator(next_observation_seq_batch, decoded_next_max_plan_reps)
            discounted_rw_seq_batch = tf.expand_dims(
                tf.matmul(reward_seq_batch, reward_accumulation_mx), axis=-1)
            # b_s x max_n x 1
            discounted_targ_q = targ_q * q_discounting_vector
            # b_s x max_n x 1
            targets = discounted_rw_seq_batch + tf.expand_dims(
                1 - done_seq_batch, axis=-1) * discounted_targ_q
            # b_s x max_n x 1
            tiled_obs_batch = tf.tile(tf.expand_dims(observation_batch,  # b_s x obs_dim
                                                     axis=1), [1, self._max_plan_len, 1])
            # b_s x max_n x obs_dim
            with tf.GradientTape(
                    persistent=True) as tape:
                plan_representations, decoded_action_seq, decoded_end_seq, reconstruction_loss, end_seq_loss, reg_loss = (
                    self.get_rt_ae().get_masked_ae_losses(action_seq_batch, episode_mask))
                if self._ema_rt_ae:
                    online_plan_representations, decoded_action_seq, decoded_end_seq, reconstruction_loss, end_seq_loss, reg_loss = (
                        self._rt_ae.get_masked_ae_losses(action_seq_batch, episode_mask))
                pred_q = critic.get_q(tiled_obs_batch, plan_representations)
                # b_s x max_n x 1
                masked_td_errors = td_error(target=targets, q_pred=pred_q) * tf.expand_dims(episode_mask, axis=-1)
                q_loss = 0.5 * tf.reduce_mean(masked_td_errors)
                ae_loss = get_ae_loss(q_loss, reconstruction_loss, end_seq_loss,
                                      reg_loss)
            q_gradients = tape.gradient(q_loss, self.get_critic_trainable_variables(critic))
            self.critic_opt.apply_gradients(zip(q_gradients, self.get_critic_trainable_variables(critic)))
            plan_ae_optimized_vars = self._rt_ae.trainable_variables
            ae_gradients = tape.gradient(ae_loss, plan_ae_optimized_vars)
            self.rt_ae_opt.apply_gradients(zip(ae_gradients, plan_ae_optimized_vars))
            del tape
            return q_loss, reconstruction_loss, end_seq_loss, reg_loss

        return tf.function(train)

    def make_actor_train_op(self, ):
        if self._rt_ae_actor_consistency_loss is None:
            def consistency_loss(plan_reps, decoded_plan_reps):
                return 0.0
        elif self._rt_ae_actor_consistency_loss == 'L2':
            def consistency_loss(plan_reps, decoded_plan_reps):
                return tf.reduce_mean(tf.square(plan_reps - decoded_plan_reps))
        else:
            raise NotImplementedError
        if self._use_min_q_actor:
            def advantage_estimator(observation_batch, plan_reps):
                advantage1 = self._cri.get_q(observation_batch, plan_reps)
                advantage2 = self._cri2.get_q(observation_batch, plan_reps)
                return tf.minimum(advantage1, advantage2)
        else:
            def advantage_estimator(observation_batch, plan_reps):
                return self._cri.get_q(observation_batch, plan_reps)

        def train(observation_batch):
            with tf.GradientTape(persistent=True) as tape:
                plan_representations = self._act.get_plan(observation_batch, 0.0)
                decoded_plan_reps, _, _ = self.get_rt_ae().autoencode_plans(plan_representations)
                advantage = advantage_estimator(observation_batch=observation_batch,
                                                plan_reps=decoded_plan_reps)
                advantage_loss = tf.reduce_mean(-1 * advantage)
                cons_loss = consistency_loss(plan_representations, decoded_plan_reps)
                loss = advantage_loss + cons_loss
            gradients = tape.gradient(loss, self.get_actor_trainable_variables())

            if self._clip_actor_gradients:
                gradients, _ = tf.clip_by_global_norm(gradients, 40)
            self.actor_opt.apply_gradients(zip(gradients, self.get_actor_trainable_variables()))
            if self._joint_rt_ae_actor_optim:
                ae_optimized_vars = self._rt_ae._plan_decoder.trainable_variables
                ae_gradients = tape.gradient(loss, ae_optimized_vars)
                self.rt_ae_opt.apply_gradients(zip(ae_gradients, ae_optimized_vars))
            del tape
            return advantage_loss, cons_loss

        return train

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
