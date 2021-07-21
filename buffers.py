import numpy as np


class FastReplayBuffer(object):
    """Replay buffer model."""

    def __init__(self, buffer_size, initial_data={}, visual_data=True,
                 stacked_frames=3, ims_channels=3):
        self.buffer_size = buffer_size
        self.visual_data = visual_data
        if initial_data == {}:
            self.index = -1
        else:
            self._initial_setup(initial_data)
        self.stacked_frames = stacked_frames
        self.ims_channels = ims_channels
        self.full = False

    def gather_indices(self, indices):
        return {'obs': self.obs[indices], 'nobs': self.nobs[indices], 'act': self.act[indices],
                'rew': self.rew[indices], 'don': self.don[indices]}

    def _initial_setup(self, initial_data={}):
        self.index = 0
        self.obs_shape = initial_data['obs'].shape[1:]
        self.act_shape = initial_data['act'].shape[1:]
        if len(self.obs_shape) == 0:
            raise NotImplementedError
        else:
            self.obs = np.zeros([self.buffer_size, *self.obs_shape], dtype=np.float32)
            self.nobs = np.zeros([self.buffer_size, *self.obs_shape], dtype=np.float32)
            self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
            self.rew = np.zeros([self.buffer_size], dtype=np.float32)
            self.don = np.zeros([self.buffer_size], dtype=np.float32)
            self.first = np.zeros([self.buffer_size], dtype=np.bool_)
            if self.visual_data:
                self.ims_shape = initial_data['ims'].shape[1:]
                self.ims = np.zeros([self.buffer_size, *self.ims_shape], dtype=np.uint8)

    def add_data_point(self, index, data):
        np.copyto(self.obs[index], data['obs'])
        np.copyto(self.nobs[index], data['nobs'])
        np.copyto(self.act[index], data['act'])
        np.copyto(self.rew[index], data['rew'])
        np.copyto(self.don[index], data['don'])
        np.copyto(self.first[index], data['first'])
        if self.visual_data:
            np.copyto(self.ims[index], data['ims'])

    def add_data_batch(self, indexes, data):
        self.obs[indexes] = data['obs']
        self.nobs[indexes] = data['nobs']
        self.act[indexes] = data['act']
        self.rew[indexes] = data['rew']
        self.don[indexes] = data['don']
        self.first[indexes] = data['first']
        if self.visual_data:
            self.ims[indexes] = data['ims']

    def add_data_batch_indexes(self, indexes, data_indexes, data):
        self.obs[indexes] = data['obs'][data_indexes]
        self.nobs[indexes] = data['nobs'][data_indexes]
        self.act[indexes] = data['act'][data_indexes]
        self.rew[indexes] = data['rew'][data_indexes]
        self.don[indexes] = data['don'][data_indexes]
        self.first[indexes] = data['first'][data_indexes]
        if self.visual_data:
            self.ims[indexes] = data['ims'][data_indexes]

    def add_frame(self, frame):
        if self.index == -1:
            self._initial_setup(frame)
        self.add_data_point(index=self.index, data=frame)
        self.index += 1
        if self.index == self.buffer_size:
            self.index = 0
            self.full = True
        self.first[self.index] = True

    def add(self, other_data):
        if self.index == -1:
            self._initial_setup(other_data)
        end_index = self.index + other_data['n']
        if end_index > self.buffer_size:
            distance_to_end = self.buffer_size - self.index
            self.add_data_batch_indexes(indexes=np.arange(start=self.index, stop=self.buffer_size),
                                        data_indexes=np.arange(distance_to_end), data=other_data)
            remainder_index = end_index - self.buffer_size
            self.add_data_batch_indexes(indexes=np.arange(remainder_index),
                                        data_indexes=np.arange(start=distance_to_end, stop=other_data['n']),
                                        data=other_data)
            self.index = remainder_index
            self.full = True
        else:
            self.add_data_batch(indexes=np.arange(start=self.index, stop=end_index), data=other_data)
            self.index = end_index
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True
        self.first[self.index] = True

    def split_ims(self, ims):
        nims = ims[..., :self.ims_channels * self.stacked_frames]
        ims = ims[..., -self.ims_channels * self.stacked_frames:]
        return nims, ims

    def get_random_batch(self, batch_size):
        if self.full:
            indices = np.random.randint(self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(self.index, size=batch_size)
        return self.gather_indices(indices)

    def gather_n_steps_indices_slow(self, indices, n):
        n_samples = indices.shape[0]
        obs = self.obs[indices]
        nobses = np.empty((n_samples, n, *self.obs_shape))
        acts = np.empty((n_samples, n, *self.act_shape))
        rews = np.empty((n_samples, n))
        dones = np.empty((n_samples, n))
        mask = np.empty((n_samples, n))
        if self.visual_data:
            imses = np.empty((n_samples, n, *self.ims_shape))
        for i in range(n):
            if i == 0:
                mask[:, 0] = 1 - self.first[indices]
            else:
                mask[:, i] = mask[:, i - 1] * (1 - self.first[indices + i])
            nobses[:, i] = self.nobs[indices + i]
            acts[:, i] = self.act[indices + i]
            rews[:, i] = self.rew[indices + i]
            dones[:, i] = self.don[indices + i]
            if self.visual_data:
                imses[:, i] = self.ims[indices + i]
        imses, nimses = self.split_ims(imses)
        ims = imses[0]
        return {'obs': obs, 'nobses': nobses, 'acts': acts, 'rews': rews,
                'dones': dones, 'ims': ims, 'nimses': nimses, 'mask': mask}

    def gather_n_steps_indices(self, indices, n):
        n_samples = indices.shape[0]
        gather_ranges = np.stack([np.arange(indices[i], indices[i] + n)
                                  for i in range(n_samples)], axis=0) % self.buffer_size
        obs = self.obs[indices]
        nobses = self.nobs[gather_ranges]
        acts = self.act[gather_ranges]
        rews = self.rew[gather_ranges]
        dones = self.don[gather_ranges]
        mask = 1 - self.first[gather_ranges]
        mask[0] = 1
        for i in range(n - 2):
            mask[:, i + 2] = mask[:, i + 1] * mask[:, i + 2]
        if self.visual_data:
            imses = self.ims[gather_ranges]
            imses, nimses = self.split_ims(imses)
            ims = imses[:, 0]
            return {'obs': obs, 'nobses': nobses, 'acts': acts, 'rews': rews,
                    'dones': dones, 'ims': ims, 'nimses': nimses, 'mask': mask}
        return {'obs': obs, 'nobses': nobses, 'acts': acts, 'rews': rews,
                'dones': dones, 'mask': mask}

    def gather_n_steps_actions(self, indices, n):
        n_samples = indices.shape[0]
        gather_ranges = np.stack([np.arange(indices[i], indices[i] + n)
                                  for i in range(n_samples)], axis=0)
        acts = self.act[gather_ranges]
        mask = 1 - self.first[gather_ranges]
        return {'acts': acts, 'mask': mask}

    def get_n_steps_random_batch(self, batch_size, n):
        if self.full:
            indices = np.random.randint(self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(self.index, size=batch_size)
        return self.gather_n_steps_indices(indices, n)

    def get_n_steps_random_actions_batch(self, batch_size, n):
        if self.full:
            indices = np.random.randint(self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(self.index, size=batch_size)
        return self.gather_n_steps_actions(indices, n)

    def gather_n_steps_indices_for_representations(self, indices, n):
        assert self.visual_data
        n_samples = indices.shape[0]
        gather_ranges = np.stack([np.arange(indices[i], indices[i] + n)
                                  for i in range(n_samples)], axis=0)
        obs = self.obs[indices]
        nobs = self.nobs[indices]
        acts = self.act[gather_ranges]
        act = acts[:, 0]
        rew = self.rew[indices]
        done = self.don[indices]
        mask = 1 - self.first[gather_ranges]
        for i in range(n - 1):
            mask[:, i + 1] = mask[:, i] * mask[:, i + 1]
        imses = self.ims[gather_ranges]
        imses, nimses = self.split_ims(imses)
        ims = imses[:, 0]
        nims = nimses[:, 0]
        return {'obs': obs, 'nobs': nobs, 'act': act, 'acts': acts, 'rew': rew,
                'done': done, 'ims': ims, 'nims': nims, 'nimses': nimses, 'mask': mask}


class FiGARReplayBuffer(FastReplayBuffer):
    """Replay buffer model, storing FiGAR data."""

    def __init__(self, buffer_size, initial_data={}):
        super(FiGARReplayBuffer, self).__init__(buffer_size=buffer_size,
                                                initial_data=initial_data,
                                                visual_data=False)

    def gather_indices(self, indices):
        out_dict = super(FiGARReplayBuffer, self).gather_indices(indices=indices)
        out_dict['reps'] = self.reps[indices]
        return out_dict

    def _initial_setup(self, initial_data={}):
        super(FiGARReplayBuffer, self)._initial_setup(initial_data=initial_data)
        self.reps = np.zeros([self.buffer_size], dtype=np.int32)

    def add_data_point(self, index, data):
        super(FiGARReplayBuffer, self).add_data_point(index=index, data=data)
        np.copyto(self.reps[index], data['reps'])

    def add_data_batch(self, indexes, data):
        super(FiGARReplayBuffer, self).add_data_batch(indexes=indexes, data=data)
        self.reps[indexes] = data['reps']

    def add_data_batch_indexes(self, indexes, data_indexes, data):
        super(FiGARReplayBuffer, self).add_data_batch_indexes(indexes=indexes,
                                                              data_indexes=data_indexes,
                                                              data=data)
        self.reps[indexes] = data['reps'][data_indexes]
