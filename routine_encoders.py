import tensorflow as tf
import numpy as np
from layers_utils import run_layers, DeltaOrthogonal

tfm = tf.keras.models
tfl = tf.keras.layers
tfr = tf.keras.regularizers
tfi = tf.keras.initializers

R2 = np.sqrt(2)
LN4 = np.log(4)
orthogonal_init = tfi.Orthogonal(gain=R2)
delta_orthogonal_init = DeltaOrthogonal(gain=R2)


class RoutineEncoder(tfl.Layer):
    """Routine encoder model."""

    def __init__(self, action_encoding_layers, recurrent_layers,
                 plan_representation_layers):
        super(RoutineEncoder, self).__init__()
        self._action_encoding_layers = action_encoding_layers
        self._recurrent_layers = recurrent_layers
        self._plan_representation_layers = plan_representation_layers

    def call(self, inputs):
        return run_layers([*self._action_encoding_layers,
                           *self._recurrent_layers,
                           *self._plan_representation_layers], inputs)

    def get_plan_representation(self, plan):
        recurrent_encodings = run_layers([*self._action_encoding_layers,
                                          *self._recurrent_layers], plan)
        return run_layers(self._plan_representation_layers,
                          recurrent_encodings[:, -1, :])


class RoutineDecoder(tfl.Layer):
    """Deterministic routine decoder model."""

    def __init__(self, max_plan_len, recurrent_layers, shared_decoding_layers,
                 action_decoding_layers=[], done_layers=[]):
        super(RoutineDecoder, self).__init__()
        self._recurrent_layers = recurrent_layers
        self._shared_decoding_layers = shared_decoding_layers
        self._action_decoding_layers = action_decoding_layers
        self._done_layers = done_layers
        if len(action_decoding_layers) == 0 and len(done_layers) == 0:
            self._shared_out = True
            out_dim = self._shared_decoding_layers[-1].units
            self._action_dims = out_dim - 1
            self._split_dims = [self._action_dims, 1]
        else:
            self._shared_out = False
            self._action_dims = self._action_decoding_layers[-1].units
        self._max_plan_len = max_plan_len

    def get_heads_results(self, shared_rep):
        if self._shared_out:
            plans, dones = tf.split(shared_rep, self._split_dims, axis=-1)
        else:
            plans = run_layers(self._action_decoding_layers, shared_rep)
            dones = run_layers(self._done_layers, shared_rep)
        return tf.nn.tanh(plans), tf.nn.sigmoid(dones)

    def call(self, inputs):
        if len(self._recurrent_layers) > 0:
            plan_representations = tf.expand_dims(inputs, axis=-2)
            batched_representations = tf.tile(plan_representations, [1, self._max_plan_len, 1])
        else:
            batched_representations = inputs
        shared_rep = run_layers([*self._recurrent_layers,
                                 *self._shared_decoding_layers],
                                batched_representations)
        return self.get_heads_results(shared_rep)

    def decode_batch_actions(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_sizes = inputs_shape[:-1]
        flat_input = tf.reshape(inputs, [tf.reduce_prod(batch_sizes), -1])
        flat_plans, flat_dones = self.__call__(flat_input)
        out_shape = tf.concat((batch_sizes, [self._max_plan_len, -1]), axis=-1)
        plans = tf.reshape(flat_plans, out_shape)
        dones = tf.reshape(flat_dones, out_shape)
        return plans, dones


class StochasticRoutineDecoder(tfl.Layer):
    """Stochastic routine decoder model."""

    def __init__(self, max_plan_len,
                 recurrent_layers, shared_decoding_layers,
                 action_decoding_layers=[], done_layers=[],
                 min_log_stddev=-10, max_log_stddev=2, ):
        super(StochasticRoutineDecoder, self).__init__()
        self._recurrent_layers = recurrent_layers
        self._shared_decoding_layers = shared_decoding_layers
        self._action_decoding_layers = action_decoding_layers
        self._done_layers = done_layers
        if len(action_decoding_layers) == 0 and len(done_layers) == 0:
            self._shared_out = True
            out_dim = self._shared_decoding_layers[-1].units
            self._action_params_dims = out_dim - 1
            self._split_dims = [self._action_params_dims, 1]
        else:
            self._shared_out = False
            self._action_params_dims = self._action_decoding_layers[-1].units
        self._action_dim = self._action_params_dims // 2

        self._log_prob_offset = self._action_dim / 2 * np.log(np.pi * 2)

        self._min_log_stddev = min_log_stddev
        self._range_log_stddev = max_log_stddev - min_log_stddev

        self._max_plan_len = max_plan_len

    def get_heads_results(self, shared_rep):
        if self._shared_out:
            action_reps, dones = tf.split(shared_rep, self._split_dims, axis=-1)
        else:
            action_reps = run_layers(self._action_decoding_layers, shared_rep)
            dones = run_layers(self._done_layers, shared_rep)
        mean, log_stddev = tf.split(action_reps, 2, -1)
        scaled_log_stddev = self._min_log_stddev + (tf.tanh(log_stddev) + 1) / 2 * self._range_log_stddev
        stddev = tf.exp(scaled_log_stddev)
        return mean, stddev, tf.nn.sigmoid(dones)

    def call(self, inputs):
        if len(self._recurrent_layers) > 0:
            plan_representations = tf.expand_dims(inputs, axis=-2)
            batched_representations = tf.tile(plan_representations, [1, self._max_plan_len, 1])
        else:
            batched_representations = inputs
        shared_rep = run_layers([*self._recurrent_layers,
                                 *self._shared_decoding_layers],
                                batched_representations)
        return self.get_heads_results(shared_rep)

    def decode_actions(self, inputs, apply_noise):
        mean, stddev, dones = self.__call__(inputs)
        if apply_noise:
            return tf.tanh(mean + tf.random.normal(tf.shape(mean)) * stddev), dones
        return tf.tanh(mean), dones

    def decode_batch_actions(self, inputs, apply_noise):
        inputs_shape = tf.shape(inputs)
        batch_sizes = inputs_shape[:-1]
        flat_input = tf.reshape(inputs, [tf.reduce_prod(batch_sizes), -1])
        flat_plans, flat_dones = self.decode_actions(flat_input, apply_noise)
        out_shape = tf.concat((batch_sizes, [self._max_plan_len, -1]), axis=-1)
        plans = tf.reshape(flat_plans, out_shape)
        dones = tf.reshape(flat_dones, out_shape)
        return plans, dones

    def decode_actions_probability(self, inputs):
        mean, stddev, dones = self.__call__(inputs)

        random_component = tf.random.normal(tf.shape(mean))
        raw_actions = mean + random_component * stddev
        actions = tf.tanh(raw_actions)
        log_probs = (-1 / 2 * tf.reduce_sum(tf.square(random_component), axis=-1) -
                     tf.reduce_sum(tf.math.log(stddev), axis=-1) - self._log_prob_offset)
        squash_features = -2 * raw_actions
        squash_correction = tf.reduce_sum(LN4 + squash_features - 2 * tf.math.softplus(squash_features), axis=-1)
        log_probs -= squash_correction
        log_probs = tf.expand_dims(log_probs, axis=-1)
        return actions, log_probs, dones

    def decode_batch_actions_probability(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_sizes = inputs_shape[:-1]
        flat_input = tf.reshape(inputs, [tf.reduce_prod(batch_sizes), -1])
        flat_plans, flat_log_probs, flat_dones = self.decode_actions_probability(flat_input)
        out_shape = tf.concat((batch_sizes, [self._max_plan_len, -1]), axis=-1)
        plans = tf.reshape(flat_plans, out_shape)
        log_probs = tf.reshape(flat_log_probs, out_shape)
        dones = tf.reshape(flat_dones, out_shape)
        return plans, log_probs, dones


class SimpleFCRoutineDecoder(RoutineDecoder):
    """Deterministic routine decoder model utilizing only fully connected layer mappings (no recurrent layers)."""

    def __init__(self, max_plan_len, layers):
        super(SimpleFCRoutineDecoder, self).__init__(max_plan_len=max_plan_len,
                                                     recurrent_layers=[],
                                                     shared_decoding_layers=layers,
                                                     action_decoding_layers=[],
                                                     done_layers=[])


class SimpleStochasticFCRoutineDecoder(StochasticRoutineDecoder):
    """Stochastic routine decoder model utilizing only fully connected layer mappings (no recurrent layers)."""

    def __init__(self, max_plan_len, layers):
        super(SimpleStochasticFCRoutineDecoder, self).__init__(max_plan_len=max_plan_len,
                                                               recurrent_layers=[],
                                                               shared_decoding_layers=layers,
                                                               action_decoding_layers=[],
                                                               done_layers=[])


class RoutineAE(tfm.Model):
    """Deterministic routine autoencoder composed of a routine encoder and a deterministic routine decoder model."""

    def __init__(self, plan_encoder, plan_decoder, plan_l2_reg=0.0):
        super(RoutineAE, self).__init__()
        self._plan_encoder = plan_encoder
        self._plan_decoder = plan_decoder
        self._plan_l2_reg = plan_l2_reg
        self._max_plan_len = self._plan_decoder._max_plan_len
        self.end_seq_targets = tf.reshape(
            tf.eye(self._max_plan_len),
            [1, self._max_plan_len, self._max_plan_len, 1])
        self.loss_mask = tf.expand_dims(
            tf.linalg.band_part(tf.ones((self._max_plan_len, self._max_plan_len)),
                                num_lower=-1, num_upper=0), axis=0)

    def get_actions(self, plan_representations):
        actions, dones_probs = self.decode(plan_representations)
        dones_shape = tf.shape(dones_probs)
        dones_seeds = tf.random.uniform((dones_shape[0], dones_shape[1] - 1))
        dones = tf.concat((tf.cast((dones_seeds < dones_probs[:, :-1, 0]), tf.int8),
                           tf.ones((dones_shape[0], 1), dtype=tf.int8)), axis=-1)
        end_seq = tf.argmax(dones, axis=1)
        return actions, end_seq + 1

    def decode(self, plan_representations):
        return self._plan_decoder(plan_representations)

    def autoencode_sequence(self, action_sequences):  # bs x max_n x plan_rep
        plan_representations = self._plan_encoder(action_sequences)  # bs x max_n x plan_rep
        decoded_action_sequences, decoded_end_sequences = self._plan_decoder.decode_batch_actions(
            plan_representations)  # bs x max_n x max_n x (action_rep, 1)
        return plan_representations, decoded_action_sequences, decoded_end_sequences

    def autoencode_plans(self, plan_representations):  # bs x plan_rep
        action_seq, end_seq = self.get_actions(plan_representations)  # bs x max_n x (act_rep, 1)
        decoded_plan_representations = self._plan_encoder(action_seq)  # bs x max_n x plan_rep
        relevant_decoded_plans = tf.gather(params=decoded_plan_representations,
                                           indices=tf.expand_dims(end_seq, axis=1) - 1, axis=1, batch_dims=1)
        return relevant_decoded_plans[:, 0, :], action_seq, end_seq

    def autoencode_plans_batch(self, plan_representations):
        plan_shape = tf.shape(plan_representations)
        batch_dims = plan_shape[:-1]
        batch_size = tf.reduce_prod(batch_dims)
        reshaped_plan_reps = tf.reshape(plan_representations, [batch_size, -1])
        decoded_plan_reps, action_seq, end_seq = self.autoencode_plans(reshaped_plan_reps)
        decoded_plan_reps = tf.reshape(decoded_plan_reps, tf.concat((batch_dims, [-1]), axis=-1))
        action_seq = tf.reshape(action_seq, tf.concat((batch_dims, [self._max_plan_len, -1]), axis=-1))
        end_seq = tf.reshape(end_seq, batch_dims)
        return decoded_plan_reps, action_seq, end_seq

    def get_masked_ae_losses(self, action_sequences, episode_mask):
        if episode_mask is None:
            episode_mask = tf.ones((tf.shape(action_sequences)[0], self._max_plan_len))
        batch_loss_mask = self.loss_mask * tf.expand_dims(episode_mask, axis=-1)  # bs x max_n x max_n
        num_elements = tf.reduce_sum(batch_loss_mask)
        plan_representations, decoded_action_sequences, decoded_end_seq = self.autoencode_sequence(action_sequences)
        if self._plan_l2_reg > 0.0:
            plan_rep_l2 = tf.reduce_sum(tf.square(plan_representations), axis=-1)
            reg_loss = tf.reduce_sum(plan_rep_l2 * episode_mask) / num_elements * self._plan_l2_reg
        else:
            reg_loss = 0.0
        l2_diff = tf.reduce_mean(tf.square((tf.expand_dims(action_sequences, axis=1) - decoded_action_sequences))
                                 , axis=-1)  # bs x max_n x max_n # axis=-2
        reconstruction_loss = tf.reduce_sum(l2_diff * batch_loss_mask) / num_elements
        bce_loss = tf.keras.losses.binary_crossentropy(
            y_true=tf.broadcast_to(self.end_seq_targets, tf.shape(decoded_end_seq)),
            y_pred=decoded_end_seq)  # bs x max_n x max_n
        end_seq_loss = tf.reduce_sum(bce_loss * batch_loss_mask) / num_elements
        return (plan_representations, decoded_action_sequences, decoded_end_seq,
                reconstruction_loss, end_seq_loss, reg_loss)

    def make_rt_ae_train_op(self, optimizer):
        def train(action_seq_batch, episode_mask):
            with tf.GradientTape() as tape:
                plan_representations, decoded_action_seq, decoded_end_seq, reconstruction_loss, end_seq_loss, reg_loss = (
                    self.get_masked_ae_losses(action_seq_batch, episode_mask))
                ae_loss = reconstruction_loss + end_seq_loss + reg_loss
            ae_gradients = tape.gradient(ae_loss, self.trainable_variables)
            optimizer.apply_gradients(zip(ae_gradients, self.trainable_variables))
            return reconstruction_loss, end_seq_loss, reg_loss

        return tf.function(train)

    def call(self, inputs):
        action_seq, dones = self._plan_decoder(inputs)
        recovered_representations = self._plan_encoder(action_seq)
        return {'acts': action_seq, 'lens': dones,
                'rec_plans': recovered_representations}


class StochasticRoutineAE(tfm.Model):
    """Stochastic routine autoencoder composed of a routine encoder and a stochastic routine decoder model."""

    def __init__(self, plan_encoder, plan_decoder, plan_l2_reg=0.0):
        super(StochasticRoutineAE, self).__init__()
        self._plan_encoder = plan_encoder
        self._plan_decoder = plan_decoder
        assert isinstance(self._plan_decoder, StochasticRoutineDecoder), 'When using the StochasticPlanAE, the' \
                                                                         'decoder must be of a Stochastic subtype'
        self._plan_l2_reg = plan_l2_reg
        self._max_plan_len = self._plan_decoder._max_plan_len
        self.end_seq_targets = tf.reshape(
            tf.eye(self._max_plan_len),
            [1, self._max_plan_len, self._max_plan_len, 1])  # 1 x max_n x max_n (upper triangular) x 1
        self.loss_mask = tf.expand_dims(
            tf.linalg.band_part(tf.ones((self._max_plan_len, self._max_plan_len)),
                                num_lower=-1, num_upper=0), axis=0)  # 1 x max_n x max_n (lower triangular)

    def get_actions(self, plan_representations, apply_noise):
        actions, dones_probs = self.decode(plan_representations, apply_noise=apply_noise)
        dones_shape = tf.shape(dones_probs)
        dones_seeds = tf.random.uniform((dones_shape[0], dones_shape[1] - 1))
        dones = tf.concat((tf.cast((dones_seeds < dones_probs[:, :-1, 0]), tf.int8),
                           tf.ones((dones_shape[0], 1), dtype=tf.int8)), axis=-1)
        end_seq = tf.argmax(dones, axis=1)
        return actions, end_seq + 1

    def decode(self, plan_representations, apply_noise):
        return self._plan_decoder.decode_actions(plan_representations, apply_noise=apply_noise)

    def get_actions_probability(self, plan_representations):
        actions, log_probs, dones_probs = self.decode_probability(plan_representations)
        dones_shape = tf.shape(dones_probs)
        dones_seeds = tf.random.uniform((dones_shape[0], dones_shape[1] - 1))
        dones = tf.concat((tf.cast((dones_seeds < dones_probs[:, :-1, 0]), tf.int8),
                           tf.ones((dones_shape[0], 1), dtype=tf.int8)), axis=-1)
        end_seq = tf.argmax(dones, axis=1)
        return actions, log_probs, end_seq + 1

    def decode_probability(self, plan_representations):
        return self._plan_decoder.decode_actions_probability(plan_representations)

    def autoencode_sequence(self, action_sequences, apply_noise):  # bs x max_n x plan_rep
        plan_representations = self._plan_encoder(action_sequences)  # bs x max_n x plan_rep
        decoded_action_sequences, decoded_end_sequences = self._plan_decoder.decode_batch_actions(
            plan_representations, apply_noise)  # bs x max_n x max_n x (action_rep, -)
        return plan_representations, decoded_action_sequences, decoded_end_sequences

    def autoencode_sequence_probability(self, action_sequences):  # bs x max_n x plan_rep
        plan_representations = self._plan_encoder(action_sequences)  # bs x max_n x plan_rep
        decoded_action_sequences, decoded_actions_probabilities, decoded_end_sequences = self._plan_decoder.decode_batch_actions_probability(
            plan_representations)  # bs x max_n x max_n x (action_rep, 1, 1)
        return plan_representations, decoded_action_sequences, decoded_actions_probabilities, decoded_end_sequences

    def autoencode_plans(self, plan_representations, apply_noise):  # bs x plan_rep
        action_seq, end_seq = self.get_actions(plan_representations,
                                               apply_noise=apply_noise)  # bs x max_n x (act_rep, 1)
        decoded_plan_representations = self._plan_encoder(action_seq)  # bs x max_n x plan_rep

        relevant_decoded_plans = tf.gather(params=decoded_plan_representations,
                                           indices=tf.expand_dims(end_seq, axis=1) - 1, axis=1, batch_dims=1)
        return relevant_decoded_plans[:, 0, :], action_seq, end_seq

    def autoencode_plans_probability(self, plan_representations):  # bs x plan_rep
        action_seq, log_probs, end_seq = self.get_actions_probability(
            plan_representations)  # bs x max_n x (act_rep, 1, 1)
        decoded_plan_representations = self._plan_encoder(action_seq)  # bs x max_n x plan_rep

        relevant_decoded_plans = tf.gather(params=decoded_plan_representations,
                                           indices=tf.expand_dims(end_seq, axis=1) - 1, axis=1, batch_dims=1)
        return relevant_decoded_plans[:, 0, :], action_seq, log_probs, end_seq

    def autoencode_plans_batch(self, plan_representations, apply_noise):
        plan_shape = tf.shape(plan_representations)
        batch_dims = plan_shape[:-1]
        batch_size = tf.reduce_prod(batch_dims)
        reshaped_plan_reps = tf.reshape(plan_representations, [batch_size, -1])
        decoded_plan_reps, action_seq, end_seq = self.autoencode_plans(reshaped_plan_reps, apply_noise)
        decoded_plan_reps = tf.reshape(decoded_plan_reps, tf.concat((batch_dims, [-1]), axis=-1))
        action_seq = tf.reshape(action_seq, tf.concat((batch_dims, [self._max_plan_len, -1]), axis=-1))
        end_seq = tf.reshape(end_seq, batch_dims)
        return decoded_plan_reps, action_seq, end_seq

    def autoencode_plans_batch_probability(self, plan_representations):
        plan_shape = tf.shape(plan_representations)
        batch_dims = plan_shape[:-1]
        batch_size = tf.reduce_prod(batch_dims)
        reshaped_plan_reps = tf.reshape(plan_representations, [batch_size, -1])
        decoded_plan_reps, action_seq, log_probs, end_seq = self.autoencode_plans_probability(reshaped_plan_reps)
        decoded_plan_reps = tf.reshape(decoded_plan_reps, tf.concat((batch_dims, [-1]), axis=-1))
        log_probs = tf.reshape(log_probs, tf.concat((batch_dims, [self._max_plan_len, -1]), axis=-1))
        action_seq = tf.reshape(action_seq, tf.concat((batch_dims, [self._max_plan_len, -1]), axis=-1))
        end_seq = tf.reshape(end_seq, batch_dims)
        return decoded_plan_reps, action_seq, log_probs, end_seq

    def get_masked_ae_losses(self, action_sequences, episode_mask):
        if episode_mask is None:
            episode_mask = tf.ones((tf.shape(action_sequences)[0], self._max_plan_len))
        batch_loss_mask = self.loss_mask * tf.expand_dims(episode_mask, axis=-1)  # bs x max_n x max_n
        num_elements = tf.reduce_sum(batch_loss_mask)
        plan_representations, decoded_action_sequences, decoded_end_seq = self.autoencode_sequence(action_sequences,
                                                                                                   apply_noise=True)

        if self._plan_l2_reg > 0.0:
            plan_rep_l2 = tf.reduce_sum(tf.square(plan_representations), axis=-1)
            reg_loss = tf.reduce_sum(plan_rep_l2 * episode_mask) / num_elements * self._plan_l2_reg
        else:
            reg_loss = 0.0
        l2_diff = tf.reduce_mean(tf.square((tf.expand_dims(action_sequences, axis=1) - decoded_action_sequences))
                                 , axis=-1)  # bs x max_n x max_n # axis=-2
        reconstruction_loss = tf.reduce_sum(l2_diff * batch_loss_mask) / num_elements
        bce_loss = tf.keras.losses.binary_crossentropy(
            y_true=tf.broadcast_to(self.end_seq_targets, tf.shape(decoded_end_seq)),
            y_pred=decoded_end_seq)  # bs x max_n x max_n
        end_seq_loss = tf.reduce_sum(bce_loss * batch_loss_mask) / num_elements
        return (plan_representations, decoded_action_sequences, decoded_end_seq,
                reconstruction_loss, end_seq_loss, reg_loss)

    def make_rt_ae_train_op(self, optimizer):
        def train(action_seq_batch, episode_mask):
            with tf.GradientTape() as tape:
                plan_representations, decoded_action_seq, decoded_end_seq, reconstruction_loss, end_seq_loss, reg_loss = (
                    self.get_masked_ae_losses(action_seq_batch, episode_mask))
                ae_loss = reconstruction_loss + end_seq_loss + reg_loss
            ae_gradients = tape.gradient(ae_loss, self.trainable_variables)
            optimizer.apply_gradients(zip(ae_gradients, self.trainable_variables))
            return reconstruction_loss, end_seq_loss, reg_loss

        return tf.function(train)

    def call(self, inputs):
        action_seq, dones = self._plan_decoder.decode_actions(inputs, apply_noise=True)
        recovered_representations = self._plan_encoder(action_seq)
        return {'acts': action_seq, 'lens': dones,
                'rec_plans': recovered_representations}
