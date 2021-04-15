# ========================================
# Hidden Markov Model 隐马尔可夫模型
# ========================================

import numpy as np


class HMM:
    OUTPUT_COUNT = 65536

    def __init__(self):
        self.S = {0: "B", 1: "M", 2: "E", 3: "S"}
        self.__reset_probabilities()  # 初始化频率
        self.__reset_frequencies()  # 初始化频数
        self.N = 0

    def train(self, states, observations):
        self.__count_frequencies(states, observations)
        self.__update_probabilities()
        pass

    def predict(self, observations):
        states = []
        for ob in observations:
            states.append(self.__perform_viterbi(ob))
        return states

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def __count_frequencies(self, states, observations):
        if len(states) != len(observations):
            raise ValueError

        # TODO: 共用循环
        # Pi
        for s in states:
            self.Pi_f[s[0]] += 1

        # A
        for i in range(len(states)):
            for j in range(len(states[i]) - 1):
                self.A_f[states[i][j]][states[i][j + 1]] += 1

        # B
        for i in range(len(observations)):
            for j in range(len(observations[i])):
                self.B_f[states[i][j]][observations[i][j]] += 1

        self.N += len(states)  # TODO: 去掉N

    def __update_probabilities(self):
        # Pi
        self.Pi = self.Pi_f / self.N

        # A
        A_f_sum = self.A_f.sum(axis=1).reshape(-1, 1)
        A_f_sum[A_f_sum == 0] = 1
        self.A = self.A_f / A_f_sum

        # B
        B_f_sum = self.B_f.sum(axis=1).reshape(-1, 1)
        B_f_sum[B_f_sum == 0] = 1
        self.B = self.B_f / B_f_sum

    def __reset_probabilities(self):
        self.Pi = np.zeros(4)
        self.A = np.zeros((4, 4))
        self.B = np.zeros((4, HMM.OUTPUT_COUNT))

    def __reset_frequencies(self):
        self.Pi_f = np.zeros(4, dtype=int)
        self.A_f = np.zeros((4, 4), dtype=int)
        self.B_f = np.zeros((4, HMM.OUTPUT_COUNT), dtype=int)

    def __perform_viterbi(self, observation):
        ob_len = len(observation)
        if ob_len == 0:
            return []

        psi = np.zeros((ob_len - 1, 4), dtype=int)

        delta = np.zeros(4)
        delta = (self.Pi * self.B[:, observation[0]]).reshape(-1, 1)
        for t in range(1, ob_len):
            p = delta * self.A
            psi[t - 1] = p.argmax(axis=0)
            delta = (p.max(axis=0) * self.B[:, observation[t]]).reshape(-1, 1)

        s = delta.argmax()
        states = [s]
        for t in range(ob_len - 1, 0, -1):
            s = psi[t - 1][s]
            states.append(s)

        states.reverse()
        return states
