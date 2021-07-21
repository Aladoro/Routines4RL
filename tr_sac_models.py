import numpy as np
import tensorflow as tf
from sac_models import SAC

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
    """Routine actor network."""

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


class RTSAC(SAC):
    """Implementation of the Routine-based Soft Actor-Critic algorithm."""

    def __init__(self, make_routine_ae, make_actor, make_critic, make_critic2=None,
                 routine_ae_optimizer=tf.keras.optimizers.Adam(1e-3),
                 actor_optimizer=tf.keras.optimizers.Adam(1e-3),
                 critic_optimizer=tf.keras.optimizers.Adam(1e-3),
                 entropy_optimizer=tf.keras.optimizers.Adam(1e-4),
                 gamma=0.99,
                 q_polyak=0.995,
                 entropy_coefficient=0.1,
                 tune_entropy_coefficient=False,
                 target_entropy=-6,
                 approximate_current_entropy=True,
                 lower_bound_q_target_entropy=True,
                 clip_actor_gradients=True,
                 max_plan_len=4,
                 routine_ae_warmup=0,
                 clip_td_backup=None,
                 huber_td_loss=False,
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
        routine_ae_optimizer: Optimizer for the routine autoencoder model.
        actor_optimizer: Optimizer for the routine actor model.
        critic_optimizer: Optimizer for the routine critic model.
        entropy_optimizer: Optimizer for the entropy parameter.
        gamma: Discount factor.
        q_polyak: Polyak coefficient for the target networks.
        entropy_coefficient: Entropy coefficient (initial) value.
        tune_entropy_coefficient: Tune entropy coefficient.
        target_entropy: Target entropy for tuning the entropy coefficient.
        approximate_current_entropy: Approximate the entropy for current policy routines with the entropy over the first decoded actions.
        lower_bound_q_target_entropy: Approximate the entropy for the TD-target routines with the entropy over the first next decoded actions.
        clip_actor_gradients: Clip the magnitude of the actor network gradient.
        max_plan_len: Maximum number of actions each routine can map to (L).
        routine_ae_warmup: Number of routine autoencoder pre-training using a reconstruction loss.
        clip_td_backup: Clip the TD-targets magnitude to some maximum value.
        huber_td_loss: Utilize a huber TD-loss instead of MSE.
        routine_ae_critic_optim: Optimize the routine autoencoder end-to-end with the critic.
        double_joint_routine_ae_critic_optim: Optimize the routine autoencoder end-to-end with both critics.
        joint_routine_ae_actor_optim: Optimize the routine autoencoder end-to-end with the actor.
        routine_ae_recon_optim: Optimize the routine autoencoder with an additional L2 reconstruction optimization.
        routine_ae_actor_consistency_loss: Loss for the routine consistency many-to-one penalty.
        """
        initialize_model = kwargs.get('initialize_model', True)
        if initialize_model:
            tfm.Model.__init__(self, )

        self._rt_ae = make_routine_ae()
        self._joint_rt_ae_critic_optim = routine_ae_critic_optim
        self._double_joint_rt_ae_critic_optim = double_joint_routine_ae_critic_optim

        self._joint_rt_ae_actor_optim = joint_routine_ae_actor_optim

        self._rt_ae_actor_consistency_loss = routine_ae_actor_consistency_loss

        self._rt_ae_recon_optim = routine_ae_recon_optim

        self.rt_ae_opt = routine_ae_optimizer
        self._max_plan_len = max_plan_len

        self.upper_triangular = tf.linalg.band_part(tf.ones((self._max_plan_len, self._max_plan_len)),
                                                    num_lower=0, num_upper=-1)
        self.reward_accumulation_mx = self.upper_triangular * tf.constant(
            [[gamma ** i] for i in range(self._max_plan_len)])

        self._rt_ae_update = self._rt_ae.make_rt_ae_train_op(self.rt_ae_opt)
        self._rt_ae_warmup = routine_ae_warmup
        self._perform_rt_ae_warmup = routine_ae_warmup > 0

        self._clip_td_backup = clip_td_backup
        self._huber_td_loss = huber_td_loss

        self._approximate_current_entropy = approximate_current_entropy
        self._lower_bound_q_target_entropy = lower_bound_q_target_entropy

        SAC.__init__(self, make_actor=make_actor,
                     make_critic=make_critic,
                     make_critic2=make_critic2,
                     actor_optimizer=actor_optimizer,
                     critic_optimizer=critic_optimizer,
                     entropy_optimizer=entropy_optimizer,
                     gamma=gamma,
                     q_polyak=q_polyak,
                     entropy_coefficient=entropy_coefficient,
                     tune_entropy_coefficient=tune_entropy_coefficient,
                     target_entropy=target_entropy,
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
        out = {**out, **self._rt_ae(out['plan'])}
        if self._double_q:
            out['q'] = tf.minimum(self._cri.get_q(inputs, out['plan']),
                                  self._cri2.get_q(inputs, out['plan']))
            out['t_q'] = tf.minimum(self._targ_cri.get_q(inputs, out['plan']),
                                    self._targ_cri2.get_q(inputs, out['plan']))
        else:
            out['q'] = self._cri.get_q(inputs, out['plan'])
            out['t_q'] = self._targ_cri.get_q(inputs, out['plan'])
        return out

    def get_action_plan(self, observation_batch, noise_stddev, *args, **kwargs):
        actions, end_seq = self.get_actions(observation_batch, noise_stddev)
        return tf.unstack(actions[..., :end_seq[0], :], axis=-2)

    @tf.function
    def get_action(self, observation_batch, noise_stddev, *args, **kwargs):
        actions, _ = self.get_actions(observation_batch, noise_stddev)
        return actions[:, 0, :]

    @tf.function
    def get_actions(self, observation_batch, noise_stddev, **kwargs):
        plan = self._act.get_plan(observation_batch, 0.0)
        actions, end_seq = self._rt_ae.get_actions(plan, apply_noise=(noise_stddev > 0.0))
        return actions, end_seq

    def train(self, buffer, batch_size=128, n_updates=1, act_delay=2, **kwargs):
        if self._perform_rt_ae_warmup:
            for i in range(self._rt_ae_warmup):
                b = buffer.get_n_steps_random_batch(batch_size, self._max_plan_len)
                actions_seq, ep_mask = b['acts'], tf.cast(b['mask'], tf.float32)
                _ = self._rt_ae_update(actions_seq, ep_mask)
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
        actor_loss, cons_loss, alpha_loss = self._train_act_and_alpha(observations)

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
                return huber_scaling((process_td_backup(target) - q_pred))
        else:
            def td_error(target, q_pred):
                return tf.square((process_td_backup(target) - q_pred))

        if self._lower_bound_q_target_entropy:
            def get_target_log_probs(log_plan_probs, end_seq):
                return log_plan_probs[..., 0, :]
        else:
            def get_target_log_probs(log_plan_probs, end_seq):
                cumulative_log_plan_probs = tf.matmul(log_plan_probs[..., 0], self.reward_accumulation_mx)
                log_plan_probs = tf.gather(cumulative_log_plan_probs, indices=tf.expand_dims(end_seq, axis=-1) - 1,
                                           axis=2, batch_dims=2)
                return log_plan_probs

        def train(observation_batch, action_seq_batch, next_observation_seq_batch,
                  reward_seq_batch, done_seq_batch, episode_mask):
            next_max_plan_reps = self._act.get_plan(next_observation_seq_batch, 0.0)  # bs x max_n x obs_dim
            decoded_next_max_plan_reps, _, log_plan_probs, end_seq = self._rt_ae.autoencode_plans_batch_probability(
                next_max_plan_reps)
            # bs x max_n x plan_len
            targ_q = q_estimator(next_observation_seq_batch, decoded_next_max_plan_reps)
            discounted_rw_seq_batch = tf.expand_dims(
                tf.matmul(reward_seq_batch, self.reward_accumulation_mx), axis=-1)
            target_log_probs = get_target_log_probs(log_plan_probs, end_seq)
            # b_s x max_n x 1
            discounted_targ_q = (targ_q - tf.exp(self._log_alpha) * target_log_probs) * q_discounting_vector
            # b_s x max_n x 1
            targets = discounted_rw_seq_batch + tf.expand_dims(
                1 - done_seq_batch, axis=-1) * discounted_targ_q
            # b_s x max_n x 1
            tiled_obs_batch = tf.tile(tf.expand_dims(observation_batch,  # b_s x obs_dim
                                                     axis=1), [1, self._max_plan_len, 1])
            # b_s x max_n x obs_dim
            plan_representations, decoded_action_seq, decoded_end_seq = self.get_plan_ae().autoencode_sequence(
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

        if self._rt_ae_recon_optim:
            def get_ae_loss(q_loss, reconstruction_loss, end_seq_loss, reg_loss):
                return q_loss + reconstruction_loss + end_seq_loss + reg_loss
        else:
            def get_ae_loss(q_loss, reconstruction_loss, end_seq_loss, reg_loss):
                return q_loss + end_seq_loss + reg_loss

        if self._lower_bound_q_target_entropy:
            def get_target_log_probs(log_plan_probs, end_seq):
                return log_plan_probs[..., 0, :]
        else:
            def get_target_log_probs(log_plan_probs, end_seq):
                cumulative_log_plan_probs = tf.matmul(log_plan_probs[..., 0], self.reward_accumulation_mx)
                log_plan_probs = tf.gather(cumulative_log_plan_probs, indices=tf.expand_dims(end_seq, axis=-1) - 1,
                                           axis=2, batch_dims=2)
                return log_plan_probs

        def train(observation_batch, action_seq_batch, next_observation_seq_batch,
                  reward_seq_batch, done_seq_batch, episode_mask):
            next_max_plan_reps = self._act.get_plan(
                next_observation_seq_batch, 0.0)  # bs x max_n x plan_dim
            decoded_next_max_plan_reps, _, log_plan_probs, end_seq = self._rt_ae.autoencode_plans_batch_probability(
                next_max_plan_reps)
            targ_q = q_estimator(next_observation_seq_batch, decoded_next_max_plan_reps)
            discounted_rw_seq_batch = tf.expand_dims(
                tf.matmul(reward_seq_batch, self.reward_accumulation_mx), axis=-1)
            target_log_probs = get_target_log_probs(log_plan_probs, end_seq)
            # b_s x max_n x 1
            discounted_targ_q = (targ_q - tf.exp(self._log_alpha) * target_log_probs) * q_discounting_vector
            # b_s x max_n x 1
            targets = discounted_rw_seq_batch + tf.expand_dims(
                1 - done_seq_batch, axis=-1) * discounted_targ_q
            # b_s x max_n x 1
            tiled_obs_batch = tf.tile(tf.expand_dims(observation_batch,  # b_s x obs_dim
                                                     axis=1), [1, self._max_plan_len, 1])
            # b_s x max_n x obs_dim
            with tf.GradientTape(persistent=True) as tape:
                plan_representations, decoded_action_seq, decoded_end_seq, reconstruction_loss, end_seq_loss, reg_loss = (
                    self._rt_ae.get_masked_ae_losses(action_seq_batch, episode_mask))
                pred_q = critic.get_q(tiled_obs_batch, plan_representations)
                # b_s x max_n x 1
                masked_td_errors = td_error(target=targets, q_pred=pred_q) * tf.expand_dims(episode_mask, axis=-1)
                q_loss = 0.5 * tf.reduce_mean(masked_td_errors)
                ae_loss = get_ae_loss(q_loss, reconstruction_loss, end_seq_loss, reg_loss)
            q_gradients = tape.gradient(q_loss, self.get_critic_trainable_variables(critic))
            self.critic_opt.apply_gradients(zip(q_gradients, self.get_critic_trainable_variables(critic)))
            plan_ae_optimized_vars = self._rt_ae.trainable_variables
            ae_gradients = tape.gradient(ae_loss, plan_ae_optimized_vars)
            self.rt_ae_opt.apply_gradients(zip(ae_gradients, plan_ae_optimized_vars))
            del tape
            return q_loss, reconstruction_loss, end_seq_loss, reg_loss

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
        if self._approximate_current_entropy:
            def get_average_log_probs(log_probs, end_seq):
                return tf.reduce_mean(log_probs[:, 0, :])
        else:
            def get_average_log_probs(log_probs, end_seq):
                cumulative_log_probs = tf.matmul(log_probs[..., 0], self.upper_triangular)
                log_summed_probs = tf.gather(cumulative_log_probs, indices=tf.expand_dims(end_seq, axis=-1) - 1,
                                             axis=1, batch_dims=1)
                return tf.reduce_sum(log_summed_probs) / tf.reduce_sum(tf.cast(end_seq, dtype=tf.float32))

        if self._tune_entropy_coefficient:
            def step_alpha(mean_log_probs):
                with tf.GradientTape() as tape:
                    loss = -self._log_alpha * tf.stop_gradient(mean_log_probs + self._target_entropy)
                    gradients = tape.gradient(loss, [self._log_alpha])
                self.entropy_opt.apply_gradients(zip(gradients, [self._log_alpha]))
                return loss
        else:
            def step_alpha(mean_log_probs):
                return 0.0
        if self._rt_ae_actor_consistency_loss is None:
            def consistency_loss(plan_reps, decoded_plan_reps):
                return 0.0
        elif self._rt_ae_actor_consistency_loss == 'L2':
            def consistency_loss(plan_reps, decoded_plan_reps):
                return tf.reduce_mean(tf.square(plan_reps - decoded_plan_reps))
        else:
            raise NotImplementedError

        def train(observation_batch):
            with tf.GradientTape(persistent=True) as tape:
                plan_representations = self._act.get_plan(observation_batch, 0.0)
                decoded_plan_reps, _, log_action_probs, end_seq = self._rt_ae.autoencode_plans_probability(
                    plan_representations)
                # b_s x plan_dim, _, b_s x max_plan_len x 1, b_s x 1
                cumulative_log_action_probs = tf.matmul(log_action_probs[..., 0], self.reward_accumulation_mx)
                log_plan_probs = tf.gather(cumulative_log_action_probs, indices=tf.expand_dims(end_seq, axis=1) - 1,
                                           axis=1, batch_dims=1)
                # b_s x 1
                q_estimates = q_estimator(observation_batch, decoded_plan_reps)
                advantage_loss = tf.reduce_mean(tf.exp(self._log_alpha) * log_plan_probs - q_estimates)
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
            alpha_loss = step_alpha(mean_log_probs=get_average_log_probs(
                log_probs=log_action_probs, end_seq=end_seq))
            return advantage_loss, cons_loss, alpha_loss

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
