import matplotlib.pyplot as plt
import numpy as np
import hashlib
import random
import string
import hmac
import time
import random

start_time = time.time()

game_hash = '88caa6d27bc387126d68bed1564154cecc307ddcb0f4bb8259f473f9551bad5d' # Update to latest game's hash for more results
first_game = "77b271fe12fca03c618f63dfb79d4105726ba9d4a25bb3f1964e435ccf9cb209"

e = 2**52
salt = "0000000000000000000fa3b65e43e4240d71762a5bf397d5304b2596d116859c"


# Check to see what the profit would be for each multiplier
def win_or_loss_per_multiplier(maximum_multiplier: float, crash_point: float, win_loss_table: dict):
    multiplier = 1.01
    while multiplier <= maximum_multiplier:
        # Lose
        if multiplier < crash_point:
            win_loss_table[multiplier] -= 1
        # Win
        elif multiplier >= crash_point:
            win_loss_table[multiplier] += (1 * multiplier)
        multiplier += .01


# Initialize the entire table for all possible multipliers up to the maximum multiplier
# The value is the amount made which starts off at zero
def initialize_win_loss_table(maximum_multiplier: float):
    win_loss_table = {}
    multiplier = 1.00
    while multiplier <= maximum_multiplier:
        multiplier = round(multiplier, 2)
        win_loss_table[multiplier] = 0
        multiplier += .01
    return win_loss_table


def get_result(game_hash):
    hm = hmac.new(str.encode(game_hash), b'', hashlib.sha256)
    hm.update(salt.encode("utf-8"))
    h = hm.hexdigest()
    # This is the house edge (4%)
    if int(h, 16) % 25 == 0:
        return 1
    h = int(h[:13], 16)
    e = 2**52
    return (((100 * e - h) / (e-h) // 1)) / 100.0


def get_prev_game(hash_code):
    m = hashlib.sha256()
    m.update(hash_code.encode("utf-8"))
    return m.hexdigest()


# This yields 2.09x but in reality it came out as 1.00x due to house edge.
# print(get_result('07c7ba5fb49664f7e2ef99b1a615e6b6cb38d2f6169434d807103d280d1c7de8'))
# print(get_result('9fb71a7386f3f193997c95dcedaf5c154691b099548a15d66eb499cfaa29f51c'))

# Total amounts of money we have is $1000
# total_pool = 1000

results = []
# frequency_table = initialize_win_loss_table(maximum_multiplier=100.00)

maximum_multiplier = 100.00
win_loss_table = initialize_win_loss_table(maximum_multiplier=maximum_multiplier)
while game_hash != first_game:
    crash_point = get_result(game_hash)
    results.append(crash_point)
    if crash_point > 100.00:
        crash_point = 100.00
    win_loss_table[crash_point] += 1
    game_hash = get_prev_game(game_hash)
    # # Calculate how much we would've lossed or gained with this crash point for each multiplier
    # # Up to the maximum multiplier (incremented by .01)
    # win_or_loss_per_multiplier(win_loss_table=win_loss_table,
    #                            crash_point=crash_point, maximum_multiplier=maximum_multiplier)

results = np.array(results)
total_samples = len(results)

print("All Multipliers and Profit Made Over {total} Samples".format(total=total_samples))

chance_four = 0
distributions = []
chances = []
for k, v in win_loss_table.items():
    chance = (v/total_samples) * 100
    chance_four += chance
    if chance_four >= 32:
        distributions.append(k)
        chances.append(chance_four)
        chance_four = 0
    print("Multiplier: {multiplier} = Frequency: {frequency} = Chance: {chance}%".format(multiplier=k, frequency=v,
                                                                                         chance=chance))

print("Distributions: {}".format(distributions))
print("Chances: {}".format(chances))


highest_multiplier_gain = max(win_loss_table, key=win_loss_table.get)
print("Highest Multiplier Gainer: {}".format(highest_multiplier_gain))


# print(freq_count)
# print("Odds of Losing: {}".format((freq_count[1] / total_amount) * 100))
# print("Odds of Winning: {}".format((freq_count[2] / total_amount) * 100))
# print("Total Made: {}".format(total_made))
# print("Total Amount of Games Played: {}".format(total_amount))
# print("Profit for 100 games for first 10 times: {}".format(profit_per_hundred[:100]))
# print("Biggest Loss: {}".format(min(profit_per_hundred)))
# print("Biggest Gain: {}".format(max(profit_per_hundred)))


# TODO: Create a Crash simulator with given percentage chances
# Split into 10 even distributions
# Choose a random number between the range
# action_to_multiplier = {
#     0: [1.00, 1.06],
#     1: [1.07, 1.2],
#     2: [1.21, 1.38],
#     3: [1.39, 1.62],
#     4: [1.63, 1.96],
#     5: [1.97, 2.48],
#     6: [2.49, 3.36],
#     7: [3.37, 5.2],
#     8: [5.21, 11.47],
#     9: [11.48, 100.00]
# }
#
# crash_count = 0
# crash_data = []
# while crash_count < 10000000:
#     random_idx = random.randint(0, 9)
#     chosen_range = action_to_multiplier[random_idx]
#     random_multiplier = round(random.uniform(chosen_range[0], chosen_range[1]), 2)
#     crash_data.append(random_multiplier)
#     crash_count += 1

elapsed_time = time.time() - start_time
print("Time elapsed: ", elapsed_time)