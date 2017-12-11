# encoding=utf-8

import random
import matplotlib.pyplot as plt
import numpy as np
import math

SAMPLING_NUM = 10000
CANDIDATE_NUM = 20


def check_success(candidates, stop_time):
    max_in_observation = max(candidates[:stop_time])
    chosen = 0
    for i in range(stop_time, len(candidates)):
        if candidates[i] > max_in_observation:
            chosen = candidates[i]
            break

    max_in_all = max(candidates)
    if math.isclose(chosen, max_in_all):
        return True

    return False


def main():
    lifes = [[random.uniform(0, 1) for i in range(CANDIDATE_NUM)] for j in range(SAMPLING_NUM)]
    success_count = [0] * CANDIDATE_NUM
    for stop_time in range(1, CANDIDATE_NUM):
        for life in lifes:
            if check_success(life, stop_time):
                success_count[stop_time] += 1

    cherry = np.argmax(success_count) + 1
    print('choose between the %d and %d candidates after %d life simulation' % (cherry, CANDIDATE_NUM, SAMPLING_NUM))

    print('standard answer is:', int(CANDIDATE_NUM / math.e) + 1)

    plt.plot(range(1, CANDIDATE_NUM), success_count[1:CANDIDATE_NUM], 'ro-', linewidth=2.0)
    plt.show()  # Could not display in Linux


if __name__ == '__main__':
    main()
