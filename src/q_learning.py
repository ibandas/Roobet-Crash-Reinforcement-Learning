import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      discount - (float) The discount factor. Controls the perceived value of
        future reward relative to short-term reward.
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
    """

    def __init__(self, epsilon=0.2, discount=0.6, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def fit(self, env):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        See page 131 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2018.pdf).
        Initialize your parameters as all zeros. For the step size (alpha), use
        1 / N, where N is the number of times the current action has been
        performed in the current state. Note that this is a different formula
        for the step size than was used in MultiArmedBandits. Use an
        epsilon-greedy policy for action selection. Note that unlike the
        pseudocode, we are looping over a total number of steps, and not a
        total number of episodes. This allows us to ensure that all of our
        trials have the same number of steps--and thus roughly the same amount
        of computation time.

        See (https://gym.openai.com/) for examples of how to use the OpenAI
        Gym Environment interface.

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "done" returned
            from env.step() is True.
          - If all values of a np.array are equal, np.argmax deterministically
            returns 0.
          - In order to avoid non-deterministic tests, use only np.random for
            random number generation.
          - Use the provided self._get_epsilon function whenever you need to
            obtain the current value of epsilon.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length 100.
            Let s = np.floor(steps / 100), then rewards[0] should contain the
            average reward over the first s steps, rewards[1] should contain
            the average reward over the next s steps, etc.
        """
        # All possible actions, and amount of times each one was chosen for each state
        num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
        N = np.zeros(num_box + (env.action_space.n,))
        # State-Action-Values is also referred to as "Q" often.
        state_action_values = np.zeros(num_box + (env.action_space.n,))

        # # All possible actions, and amount of times each one was chosen for each state
        # N = np.zeros((env.observation_space.n, env.action_space.n))
        # # State-Action-Values is also referred to as "Q" often.
        # state_action_values = np.zeros((env.observation_space.n, env.action_space.n))

        # Reset before fitting
        observation = env.reset()
        previous_observation = observation
        step_count = 4
        step_counter = 4
        wins = []
        losses = []
        passed = []

        steps = len(env.crash.data_set) / 2
        # steps = 200000
        while step_count < steps:
            if step_counter % 1000 == 0 and step_counter != 0:
                win_percent = (env.crash.win_counter / 1000) * 100
                loss_percent = (env.crash.loss_counter / 1000) * 100
                passed_percent = (env.crash.passed_counter / 1000) * 100
                wins.append(win_percent)
                losses.append(loss_percent)
                passed.append(passed_percent)
                env.crash.win_counter = 0
                env.crash.loss_counter = 0
                env.crash.passed_counter = 0
            not_first_step = True
            epsilon = self._get_epsilon(step_count / steps)
            # Decides whether to explore/exploit with epsilon probability
            choice = np.random.choice(['to exploit', 'to explore'], 1, p=[1.0 - epsilon, epsilon])[0]
            if state_action_values[observation].min == 0 and state_action_values[observation].max == 0:
                not_first_step = False

            # Maximized Action (Exploit)
            if choice == 'to exploit' and not_first_step:
                r = state_action_values[observation]
                chosen_action = np.random.choice(np.where(r == r.max())[0])

            # Random Action (Explore)
            else:
                chosen_action = np.random.randint(0, env.action_space.n)

            env.crash.current_step = step_count

            # Collects the reward and observes the state
            observation, reward, done, info = env.step(chosen_action)

            # Update the step count for the chosen action
            N[previous_observation][chosen_action] += 1

            step_count += 1
            step_counter += 1

            # The crux of the algorithm: Updating the rewards estimate
            state_action_values[previous_observation][chosen_action] = \
                state_action_values[previous_observation][chosen_action] + \
                (1.0 / N[previous_observation][chosen_action]) * \
                (reward + self.discount * np.max(state_action_values[observation][:]) -
                 state_action_values[previous_observation][chosen_action])

            previous_observation = observation

            # Reset environment when it is done
            if done:
                previous_observation = env.reset()

        self.plot_percentages(wins, losses, passed)
        percentage_lossed = np.mean(losses)
        percentage_won = np.mean(wins)
        percentage_passed = np.mean(passed)
        print("Loss Average Percentage ", percentage_lossed)
        print("Win Average Percentage ", percentage_won)
        print("Passed Average Percentage ", percentage_passed)

        return state_action_values, observation, N

    def predict(self, env, state_action_values, observation, N):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. Any mechanisms used for
            exploration in the training phase should not be used in prediction.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        # Reset before predicting
        previous_observation = observation
        # done = False
        total_steps = len(env.crash.data_set)
        step_count = env.crash.current_step
        wins = []
        losses = []
        passed = []
        # step_counter = 0

        while step_count < total_steps:
            # if step_counter % 1000 == 0 and step_counter != 0:
            #     win_percent = (env.crash.win_counter / 1000) * 100
            #     loss_percent = (env.crash.loss_counter / 1000) * 100
            #     passed_percent = (env.crash.passed_counter / 1000) * 100
            #     wins.append(win_percent)
            #     losses.append(loss_percent)
            #     passed.append(passed_percent)
            #     env.crash.win_counter = 0
            #     env.crash.loss_counter = 0
            #     env.crash.passed_counter = 0
            # Chooses best action
            chosen_action = np.argmax(state_action_values[observation])

            env.crash.current_step = step_count

            # Collects the reward and observes the state
            observation, reward, done, info = env.step(chosen_action)

            # # Update the step count for the chosen action
            # N[previous_observation][chosen_action] += 1
            #
            # # The crux of the algorithm: Updating the rewards estimate
            # state_action_values[previous_observation][chosen_action] = \
            #     state_action_values[previous_observation][chosen_action] + \
            #     (1.0 / N[previous_observation][chosen_action]) * \
            #     (reward + self.discount * np.max(state_action_values[observation][:]) -
            #      state_action_values[previous_observation][chosen_action])
            #
            # previous_observation = observation

            step_count += 1
            # step_counter += 1

        print("Profit Made: {profit}".format(profit=env.crash.money_made[-1]))
        print("Most Profit: {profit} at index: {idx}".format(profit=np.max(env.crash.money_made),
                                                             idx=np.argmax(env.crash.money_made)))
        print("Lowest Profit: {profit} at index: {idx}".format(profit=np.min(env.crash.money_made),
                                                               idx=np.argmin(env.crash.money_made)))
        for k, v in env.crash.action_counter.items():
            print("Action: {k}, Chosen {v}%".format(k=k, v=(v / (len(env.crash.data_set) / 2))))

        self.plot_percentages(wins=wins, losses=losses, passed=passed)
        self.plot_profit(env.crash.money_made)

    def _get_epsilon(self, progress):
        """
        Retrieves the current value of epsilon. Should be called by the fit
        function during each step.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        An adaptive policy for epsilon-greedy reinforcement learning. Returns
        the current epsilon value given the learner's progress. This allows for
        the amount of exploratory vs exploitatory behavior to change over time.

        See free response question 3 for instructions on how to implement this
        function.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return (1 - progress) * self.epsilon

    @staticmethod
    def plot_percentages(wins, losses, passed):
        data_file = "Crash-4states10Actions-Half-Train"
        wins_plt = plt.plot(wins, label="Win Percentage Every 1k Samples")
        losses_plt = plt.plot(losses, label="Loss Percentage Every 1k Samples")
        passed_plt = plt.plot(passed, label="Passed Percentage Every 1k Samples")
        plt.legend(handles=[wins_plt[0], losses_plt[0], passed_plt[0]])
        plt.ylabel("Percentage")
        plt.xlabel("# of 1k Samples")
        plt.savefig(f'images/{data_file}.png')

    @staticmethod
    def plot_profit(profit):
        data_file = "Profit-Crash-4states10Actions-Half-Train"
        profit_plt = plt.plot(profit, label="Profit Every 1k Samples")
        plt.legend(handles=[profit_plt[0]])
        plt.ylabel("Profit")
        plt.xlabel("# of 1k Samples")
        plt.savefig(f'images/{data_file}.png')
