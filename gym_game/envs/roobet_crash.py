import hashlib
import hmac
import numpy as np
import math


class RoobetCrash:
    """
        Roobet Crash Game

        Arguments:
            cash_out - (float) The multiplier we will auto cash on.
            game_hash - (hash) The hash code of the most current crash point

        Notes:
            We reverse the data_set with "data_set[::-1]" within "def build_data_set"
            in order to to get the chronological order of the data,
            starting from the very first crash point and ending at the most current crash point to date.
    """

    def __init__(self, cash_out=2.00, game_hash="3310b1da0c9d87b7197dec6166679fb5e099e7266d273452c97da93c9facaffb"):
        # Used to build the crash data_set
        self.game_hash = game_hash
        self.e = 2 ** 52
        self.salt = "0000000000000000000fa3b65e43e4240d71762a5bf397d5304b2596d116859c"
        self.first_game_hash = "77b271fe12fca03c618f63dfb79d4105726ba9d4a25bb3f1964e435ccf9cb209"

        self.train_or_test = "train"
        self.current_step = 4
        self.data_set = None
        self.cash_out = cash_out
        self.play_or_not = False
        self.chosen_action = None
        self.correct_counter = 0
        self.win_counter = 0
        self.passed_counter = 0
        self.loss_counter = 0
        self.money_made = [1.00]
        self.action_counter = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        }
        # Split into even distributions of 10% each
        self.action_to_multiplier = {
            0: [1.00, 1.06],
            1: [1.07, 1.2],
            2: [1.21, 1.38],
            3: [1.39, 1.62],
            4: [1.63, 1.96],
            5: [1.97, 2.48],
            6: [2.49, 3.36],
            7: [3.37, 5.2],
            8: [5.21, 11.47],
            9: [11.48, float('inf')]
        }

        # self.action_to_multiplier = {
        #     0: [1.00, 1.62],
        #     1: [1.63, 1.96],
        #     2: [1.97, 2.48],
        #     3: [2.49, 3.36],
        #     4: [3.37, 5.2],
        #     5: [5.21, 11.47],
        #     6: [11.48, float('inf')]
        # }
        # self.action_to_multiplier = {
        #     0: [1.00, 1.27],
        #     1: [1.28, 1.91],
        #     2: [1.92, 3.83],
        #     3: [3.84, float('inf')]
        # }
        # self.action_to_multiplier = {
        #     0: [1.00, 1.19],
        #     1: [1.2, 1.59],
        #     2: [1.6, 2.4],
        #     3: [2.41, 4.84],
        #     4: [4.85, float('inf')]
        # }
        # self.action_to_multiplier = {
        #     0: [1.00, 1.99],
        #     1: [2.00, float('inf')]
        # }

    # The action determines if we play or not
    # 0 means we do not play
    # 1 means we play
    def action(self, action):
        self.chosen_action = action
        if self.train_or_test == 'test':
            self.action_counter[action] += 1
        # if self.train_or_test == 'test':
        #     print("Predicted Cash Out Point: {}".format(self.action_to_multiplier[self.chosen_action][0]))

    # We evaluate if we won or not
    # If train_or_test='train', then we evaluate from the data we have right away
    # If train_or_test='test', then we evaluate from user input (live data)
    # Returns the reward
    def evaluate(self, idx):
        reward = 0
        predicted_crash_point_range = self.action_to_multiplier[idx]
        current_crash_point = self.data_set[self.current_step]
        # if self.train_or_test == 'train':
        #     if predicted_crash_point_range[0] == 1.00:
        #         reward = 0
        #         self.passed_counter += 1
        #     elif predicted_crash_point_range[0] <= current_crash_point:
        #         reward = (current_crash_point - predicted_crash_point_range[0]) * predicted_crash_point_range[0]
        #         self.win_counter += 1
        #     else:
        #         reward = -(predicted_crash_point_range[0] - current_crash_point)
        #         self.loss_counter += 1
        # elif self.train_or_test == 'test':
        #     if predicted_crash_point_range[0] == 1.00:
        #         reward = 0
        #     elif predicted_crash_point_range[0] <= current_crash_point:
        #         reward = (current_crash_point - predicted_crash_point_range[0]) * predicted_crash_point_range[0]
        #         last_profit = round(self.money_made[-1], 2)
        #         current_profit = last_profit + (predicted_crash_point_range[0] - 1.00)
        #         self.money_made.append(round(current_profit, 2))
        #     else:
        #         reward = -(predicted_crash_point_range[0] - current_crash_point)
        #         last_profit = round(self.money_made[-1], 2)
        #         current_profit = last_profit - 1.00
        #         self.money_made.append(round(current_profit, 2))
        # return reward

        if self.train_or_test == 'train':
            if predicted_crash_point_range[0] == 1.00:
                reward = 0
                self.passed_counter += 1

            elif predicted_crash_point_range[0] <= current_crash_point:
                reward = (predicted_crash_point_range[0] * 100) - 100
                self.win_counter += 1

            else:
                reward = -100
                self.loss_counter += 1

            return reward

        elif self.train_or_test == 'test':
            if predicted_crash_point_range[0] == 1.00:
                last_profit = round(self.money_made[-1], 2)
                self.money_made.append(last_profit)
                reward = 0

            elif predicted_crash_point_range[0] <= current_crash_point:
                reward = ((predicted_crash_point_range[0] * 100) - 100)
                last_profit = round(self.money_made[-1], 2)
                self.money_made.append(last_profit + round((reward/100), 2))

            else:
                reward = -100
                last_profit = self.money_made[-1]
                self.money_made.append(last_profit-1)

            return reward

    # Always return False because the game is never ending
    def is_done(self):
        return False

    # Returns the state which is the last "N" crashes
    def observe(self):
        # if self.train_or_test == "test":
            # Takes user input for testing data (Live Roobet Crash Point)
            # current_crash_point = float(input("Please enter the exact crash point value from Roobet:\n"))
            # print(f'You entered: {current_crash_point}')
            # self.data_set.append(current_crash_point)
            # self.current_step += 1

        starting_point = self.current_step - 4
        n_crashes = self.data_set[starting_point:self.current_step]
        state = []
        for crash in n_crashes:
            for k, v in self.action_to_multiplier.items():
                if v[0] <= crash <= v[1]:
                    state.append(k)
                    break
        return tuple(state)
        # return 0

    def build_data_set(self):
        results = []
        counter = 0
        while self.game_hash != self.first_game_hash:
            crash_point = self.get_result()
            results.append(crash_point)
            self.game_hash = self.get_prev_game(self.game_hash)
            counter += 1
        self.data_set = results

    def get_result(self):
        hm = hmac.new(str.encode(self.game_hash), b'', hashlib.sha256)
        hm.update(self.salt.encode("utf-8"))
        h = hm.hexdigest()
        # This is the house edge (4%)
        if int(h, 16) % 25 == 0:
            return 1
        h = int(h[:13], 16)
        e = 2 ** 52
        return (((100 * e - h) / (e - h) // 1)) / 100.0

    @staticmethod
    def get_prev_game(hash_code):
        m = hashlib.sha256()
        m.update(hash_code.encode("utf-8"))
        return m.hexdigest()
