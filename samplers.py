import numpy as np
import copy
import cv2


def log_trajectory_statistics(trajectory_rewards, log=True, trajectory_evals=None):
    out = {}
    out['number_test_episodes'] = len(trajectory_rewards)
    out['episode_returns_mean'] = np.mean(trajectory_rewards)
    out['episode_returns_max'] = np.max(trajectory_rewards)
    out['episode_returns_min'] = np.min(trajectory_rewards)
    out['episode_returns_std'] = np.std(trajectory_rewards)
    if trajectory_evals is not None:
        out['episode_trajectory_evals_mean'] = np.mean(trajectory_evals)
        out['episode_trajectory_evals_max'] = np.max(trajectory_evals)
        out['episode_trajectory_evals_min'] = np.min(trajectory_evals)
        out['episode_trajectory_evals_std'] = np.std(trajectory_evals)
    if log:
        print('Number of completed trajectories - {}'.format(out['number_test_episodes']))
        print('Latest trajectories mean reward - {}'.format(out['episode_returns_mean']))
        print('Latest trajectories max reward - {}'.format(out['episode_returns_max']))
        print('Latest trajectories min reward - {}'.format(out['episode_returns_min']))
        print('Latest trajectories std reward - {}'.format(out['episode_returns_std']))
    return out


class Sampler(object):
    """Sampler for both training and evaluation data."""

    def __init__(self, env, eval_env=None, episode_limit=1000, init_random_samples=1000,
                 visual_env=False, stacked_frames=3, ims_channels=3, augmentations=[],):
        self._env = env
        self._eval_env = eval_env or copy.deepcopy(self._env)
        self._max_episode_steps = self._env._max_episode_steps
        self._visual_env = visual_env
        self._stacked_frames = stacked_frames
        self._ims_channels = ims_channels
        self._augmentations = augmentations
        self._el = episode_limit
        self._nr = init_random_samples

        self._env_action_buffer = []

        self._tc = 0
        self._ct = 0

        self._ob = None
        self._agent_feed = None
        self._reset = True

    def handle_ob(self, ob):
        return ob.astype('float32'), ob.astype('float32')

    def get_action(self, policy, noise_stddev, buffer, observation, use_plan_buffer,
                   drop_plan=None):
        agent_input = np.expand_dims(observation, axis=0)
        if use_plan_buffer:
            action = np.array(self.get_buffer_action(policy, noise_stddev, buffer, agent_input))[0]
            if drop_plan is not None:
                if np.random.uniform() < drop_plan:
                    buffer[:] = []
            return action
        else:
            return np.array(policy.get_action(agent_input, noise_stddev=noise_stddev))[0]

    def get_buffer_action(self, policy, noise_stddev, buffer, observation):
        if len(buffer) == 0:
            buffer += policy.get_action_plan(observation, noise_stddev=noise_stddev)
        return buffer.pop(0)

    def sample_step(self, policy, noise_stddev, use_plan_buffer=False, drop_plan=None):
        if self._reset or self._ct >= self._el:
            self._env_action_buffer = []
            self._ct = 0
            self._reset = False
            first = True
            self._ob, self._agent_feed = self.handle_ob(self._env.reset())
        else:
            first = False
        if self._tc < self._nr:
            act = self._env.action_space.sample()
        else:
            act = self.get_action(policy=policy, noise_stddev=noise_stddev,
                                  buffer=self._env_action_buffer, observation=self._agent_feed,
                                  use_plan_buffer=use_plan_buffer, drop_plan=drop_plan)
        ob = self._ob
        self._ob, rew, self._reset, info = self._env.step(act)
        if self._visual_env:
            ims = self._ob['ims']
        self._ob, self._agent_feed = self.handle_ob(self._ob)
        nob = self._ob
        self._ct += 1
        if (self._ct == self._max_episode_steps) and self._reset:
            don = False
        else:
            don = self._reset
        self._tc += 1
        out = {'obs': ob, 'nobs': nob, 'act': act, 'rew': rew, 'don': don, 'first': first, 'n': 1}
        if self._visual_env:
            out['ims'] = ims
        return out

    def sample_steps(self, policy, noise_stddev, n_steps=1, use_plan_buffer=False, drop_plan=None):
        obs, nobs, acts, rews, dones, firsts, visual_obs = [], [], [], [], [], [], []
        for i in range(n_steps):
            if self._reset or self._ct >= self._el:
                self._env_action_buffer = []
                self._ct = 0
                self._reset = False
                firsts.append(True)
                self._ob, self._agent_feed = self.handle_ob(self._env.reset())
            else:
                firsts.append(False)
            if self._tc < self._nr:
                act = self._env.action_space.sample()
            else:
                act = self.get_action(policy=policy, noise_stddev=noise_stddev,
                                      buffer=self._env_action_buffer, observation=self._agent_feed,
                                      use_plan_buffer=use_plan_buffer, drop_plan=drop_plan)

            obs.append(self._ob)
            acts.append(act)
            self._ob, rew, self._reset, info = self._env.step(act)
            if self._visual_env:
                visual_obs.append(self._ob['ims'])
            self._ob, self._agent_feed = self.handle_ob(self._ob)
            nobs.append(self._ob)
            rews.append(rew)
            self._ct += 1
            if (self._ct == self._max_episode_steps) and self._reset:
                dones.append(False)
            else:
                dones.append(self._reset)
            self._tc += 1
        out = {'obs': np.stack(obs), 'nobs': np.stack(nobs), 'act': np.stack(acts),
               'rew': np.array(rews), 'don': np.array(dones), 'first': np.array(firsts),
               'n': n_steps}
        if self._visual_env:
            out['ims'] = np.stack(visual_obs)
        return out

    def sample_trajectory(self, policy, noise_stddev, use_plan_buffer=False, drop_plan=None):
        obs, nobs, acts, rews, dones, firsts, visual_obs = [], [], [], [], [], [], []
        ct = 0
        done = False
        first = True
        ob, agent_feed = self.handle_ob(self._env.reset())
        env_action_buffer = []
        while not done and ct < self._el:
            if self._tc < self._nr:
                act = self._env.action_space.sample()
            else:
                act = self.get_action(policy=policy, noise_stddev=noise_stddev,
                                      buffer=env_action_buffer, observation=agent_feed,
                                      use_plan_buffer=use_plan_buffer, drop_plan=drop_plan)
            firsts.append(first)
            first = False
            obs.append(ob)
            acts.append(act)
            ob, rew, done, info = self._env.step(act)
            if self._visual_env:
                visual_obs.append(ob['ims'])
            ob, agent_feed = self.handle_ob(ob)
            nobs.append(ob)
            rews.append(rew)
            ct += 1
            if (ct == self._max_episode_steps) and done:
                dones.append(False)
            else:
                dones.append(done)
            self._tc += 1
        self._reset = True
        out = {'obs': np.stack(obs), 'nobs': np.stack(nobs), 'act': np.stack(acts),
               'rew': np.array(rews), 'don': np.array(dones), 'first': np.array(firsts),
               'n': ct}
        if self._visual_env:
            out['ims'] = np.stack(visual_obs)
        return out

    def sample_test_trajectories(self, policy, noise_stddev, n=5, visualize=False,
                                 use_plan_buffer=False):
        obs, nobs, acts, rews, dones, rets, ids, visual_obs = [], [], [], [], [], [], [], []
        if use_plan_buffer:
            n_policy_evals = []
        for i in range(n):
            ret = 0
            ct = 0
            if use_plan_buffer:
                n_policy_eval = 0
            done = False
            ob, agent_feed = self.handle_ob(self._eval_env.reset())
            env_action_buffer = []
            while not done and ct < self._el:
                if policy is not None:
                    if use_plan_buffer and (len(env_action_buffer) == 0):
                        n_policy_eval += 1

                    act = self.get_action(policy=policy, noise_stddev=noise_stddev,
                                          buffer=env_action_buffer, observation=agent_feed,
                                          use_plan_buffer=use_plan_buffer, drop_plan=None)
                else:
                    act = self._eval_env.action_space.sample()
                ob, rew, done, info = self._eval_env.step(act)
                if visualize:
                    current_im = self._eval_env.render(mode='rgb_array')
                    cv2.imshow('visualization', current_im)
                    cv2.waitKey(10)
                ob, agent_feed = self.handle_ob(ob)
                ids.append(i)
                ret += rew
                ct += 1
            rets.append(ret)
            if use_plan_buffer:
                n_policy_evals.append(n_policy_eval)
        if visualize:
            cv2.destroyAllWindows()
        if use_plan_buffer:
            return rets, n_policy_evals
        return rets

    def evaluate(self, policy, n=10, log=True, use_plan_buffer=False):
        print('Evaluating the agent\'s behavior')
        rets = self.sample_test_trajectories(policy, 0.0, n, use_plan_buffer=use_plan_buffer)
        if use_plan_buffer:
            rets, n_policy_evals = rets
        else:
            n_policy_evals = None
        return log_trajectory_statistics(rets, log, n_policy_evals)


class FiGARSampler(object):
    """Sampler for both training and evaluation FiGAR data."""

    def __init__(self, env, eval_env=None, episode_limit=1000, init_random_samples=1000):
        self._env = env
        self._eval_env = eval_env or copy.deepcopy(self._env)
        self._max_episode_steps = self._env._max_episode_steps
        self._el = episode_limit
        self._nr = init_random_samples

        self._env_action_buffer = []

        self._tc = 0
        self._ct = 0

        self._ob = None
        self._agent_feed = None
        self._reset = True

    def handle_ob(self, ob):
        return ob.astype('float32'), ob.astype('float32')

    def get_action_and_reps(self, policy, noise_stddev, epsilon, observation):
        agent_input = np.expand_dims(observation, axis=0)
        action, repetition, rep_index = policy.get_action_and_reps(
            agent_input, noise_stddev=noise_stddev, epsilon=epsilon)
        return np.array(action)[0], np.array(repetition)[0], np.array(rep_index)[0]

    def step_action_reps(self, action, repetition):
        total_rew = 0.0
        for i in range(repetition):
            self._ob, rew, self._reset, info = self._env.step(action)
            self._ct += 1
            self._tc += 1
            total_rew += rew
            if (self._ct >= self._max_episode_steps) and self._reset:
                don = False
            else:
                don = self._reset
            if self._reset:
                return self._ob, total_rew, don, i + 1
        return self._ob, total_rew, don, repetition

    def sample_step(self, policy, noise_stddev, epsilon):
        if self._reset or self._ct >= self._el:
            self._env_action_buffer = []
            self._ct = 0
            self._reset = False
            first = True
            self._ob, self._agent_feed = self.handle_ob(self._env.reset())
        else:
            first = False
        if self._tc < self._nr:
            act = self._env.action_space.sample()
            rep_index = np.random.choice(policy._num_possible_repetitions)
            repetition = policy._possible_repetitions[rep_index]
        else:
            act, repetition, rep_index = self.get_action_and_reps(
                policy=policy, noise_stddev=noise_stddev, epsilon=epsilon,
                observation=self._agent_feed)
        ob = self._ob
        self._ob, rew, don, steps = self.step_action_reps(act, repetition)

        self._ob, self._agent_feed = self.handle_ob(self._ob)
        nob = self._ob
        out = {'obs': ob, 'nobs': nob, 'act': act, 'reps': rep_index, 'rew': rew,
               'don': don, 'first': first, 'n': 1, 'actual_steps': steps}
        return out

    def sample_steps(self, policy, noise_stddev, epsilon, n_steps=1):
        obs, nobs, acts, reps, rews, dones, firsts, visual_obs = [], [], [], [], [], [], [], []
        actual_steps = 0
        for i in range(n_steps):
            if self._reset or self._ct >= self._el:
                self._env_action_buffer = []
                self._ct = 0
                self._reset = False
                firsts.append(True)
                self._ob, self._agent_feed = self.handle_ob(self._env.reset())
            else:
                firsts.append(False)
            if self._tc < self._nr:
                act = self._env.action_space.sample()
                rep_index = np.random.choice(policy._num_possible_repetitions)
                repetition = policy._possible_repetitions[rep_index]
            else:
                act, repetition, rep_index = self.get_action_and_reps(
                    policy=policy, noise_stddev=noise_stddev, epsilon=epsilon,
                    observation=self._agent_feed)

            obs.append(self._ob)
            acts.append(act)
            reps.append(rep_index)
            self._ob, rew, don, steps = self.step_action_reps(act, repetition)
            self._ob, self._agent_feed = self.handle_ob(self._ob)
            nobs.append(self._ob)
            rews.append(rew)
            dones.append(don)
            actual_steps += steps
        out = {'obs': np.stack(obs), 'nobs': np.stack(nobs), 'act': np.stack(acts), 'reps': np.array(reps),
               'rew': np.array(rews), 'don': np.array(dones), 'first': np.array(firsts),
               'n': n_steps, 'actual_steps': actual_steps}
        return out

    def sample_all_intermediate_steps(self, policy, noise_stddev, epsilon, n_steps=1):
        obs, nobs, acts, reps, rews, dones, firsts, visual_obs = [], [], [], [], [], [], [], []
        actual_steps = 0
        for i in range(n_steps):
            if self._reset or self._ct >= self._el:
                self._env_action_buffer = []
                self._ct = 0
                self._reset = False
                firsts.append(True)
                self._ob, self._agent_feed = self.handle_ob(self._env.reset())
            else:
                firsts.append(False)
            if self._tc < self._nr:
                act = self._env.action_space.sample()
                rep_index = np.random.choice(policy._num_possible_repetitions)
                repetition = policy._possible_repetitions[rep_index]
            else:
                act, repetition, rep_index = self.get_action_and_reps(
                    policy=policy, noise_stddev=noise_stddev, epsilon=epsilon,
                    observation=self._agent_feed)

            assert rep_index == repetition - 1, "make sure repetitions are a range"

            start_obs = self._ob

            total_rew = 0.0
            j = 0
            while j < repetition and (not self._reset):
                obs.append(start_obs)
                acts.append(act)
                reps.append(j)

                self._ob, rew, self._reset, info = self._env.step(act)
                self._ct += 1
                self._tc += 1
                total_rew += rew
                if (self._ct >= self._max_episode_steps) and self._reset:
                    don = False
                else:
                    don = self._reset
                self._ob, self._agent_feed = self.handle_ob(self._ob)
                nobs.append(self._ob)
                rews.append(total_rew)
                dones.append(don)
                actual_steps += 1
                j += 1

        out = {'obs': np.stack(obs), 'nobs': np.stack(nobs), 'act': np.stack(acts), 'reps': np.array(reps),
               'rew': np.array(rews), 'don': np.array(dones), 'first': np.array(firsts),
               'n': actual_steps, 'actual_steps': actual_steps}
        return out

    def sample_test_trajectories(self, policy, noise_stddev, epsilon, n=5, visualize=False, ):
        obs, nobs, acts, rews, dones, rets, ids, visual_obs = [], [], [], [], [], [], [], []
        n_policy_evals = []
        for i in range(n):
            ret = 0
            ct = 0
            n_policy_eval = 0
            done = False
            ob, agent_feed = self.handle_ob(self._eval_env.reset())
            while not done and ct < self._el:
                n_policy_eval += 1
                act, repetition, rep_index = self.get_action_and_reps(
                    policy=policy, noise_stddev=noise_stddev, epsilon=epsilon,
                    observation=agent_feed)
                repetition = np.minimum(repetition, self._el - ct)
                for j in range(repetition):
                    ob, rew, done, info = self._eval_env.step(act)
                    if visualize:
                        current_im = self._eval_env.render(mode='rgb_array')
                        cv2.imshow('visualization', current_im)
                        cv2.waitKey(10)
                    ids.append(i)
                    ret += rew
                    ct += 1
                ob, agent_feed = self.handle_ob(ob)
            rets.append(ret)
            n_policy_evals.append(n_policy_eval)
        if visualize:
            cv2.destroyAllWindows()
        return rets, n_policy_evals

    def evaluate(self, policy, n=10, log=True):
        print('Evaluating the agent\'s behavior')
        rets, n_policy_evals = self.sample_test_trajectories(policy, 0.0, 0.0, n)
        return log_trajectory_statistics(rets, log, n_policy_evals)
