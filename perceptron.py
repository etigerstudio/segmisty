# ========================================
# Structured Averaged Perceptron 结构化平均感知机
# ========================================

from utils import NA
import utils

class Perceptron:
    def __init__(self):
        self.w = {}
        self.current_step = 0

    def train(self, sentences, tag_set):
        for i in range(10):
            converged = True
            # if i % 100 == 0:
            #     print(f"iter {i} {self.predict('“吃屎的东西，连一捆麦也铡不动呀？')[0]}")
            for sen, tags in zip(sentences, tag_set):
                y_predict, features = self.predict(sen)
                if y_predict != tags:
                    converged = False
                    # if i % 100 == 0:
                    # print(f"iter {i}: optimizing y_predict={y_predict} y_truth={tags}")  # features={features}")
                    self.__optimize(features, tags, y_predict)

            if converged:
                print(f"converged after iter {i}")
                break

    def predict(self, sentence):
        features = self.__extract_features(sentence)
        y_predict = self.__perform_viterbi(sentence, features)
        return y_predict, features

    def __optimize(self, features, y_truth, y_predict):
        self.current_step += 1
        for y, delta in [(y_truth, 1), (y_predict, -1)]:  # 奖励正确序列，惩罚错误序列
            for i in range(len(features)):
                current_tag = y[i]
                last_tag = y[i - 1] if i >= 1 else -1  # BOS: -1
                self.__update_weight(self.__embed_tag(last_tag, current_tag, features[i].copy()), delta)

    def __update_weight(self, feature, delta):
        for k in feature:
            self.__update_w_k(k, delta)  # TODO: Is learning rate necessary?
            # print(f"updated:{k} averaged:{self.__get_w_k(k)} value:{self.w[k].value} delta:{delta} accumulated:{self.w[k].accumulated}")

    @staticmethod
    def __extract_features(sentence):
        features = []

        for i in range(len(sentence)):
            l2 = sentence[i - 2] if i >= 2 else NA
            l1 = sentence[i - 1] if i >= 1 else NA
            mid = sentence[i]
            r1 = sentence[i + 1] if i < len(sentence) - 1 else NA
            r2 = sentence[i + 2] if i < len(sentence) - 2 else NA

            features.append([
                f"1 {l1}",
                f"2 {mid}",
                f"3 {r1}",
                f"4 {l2 + l1}",
                f"5 {l1 + mid}",
                f"6 {mid + r1}",
                f"7 {r1 + r2}"
            ])

        return features

    def __perform_viterbi(self, sentence, features):
        delta = [[0 for _ in range(4)] for _ in range(len(sentence))]
        psi = [[0 for _ in range(4)] for _ in range(len(sentence))]  # 0 not being used
        y = []

        # 1 初始化 Initialization
        for i in range(4):
            delta[0][i] = self.__calc_score(-1, i, features[0])  # BOS: -1

        # 2 递推 Iteration
        for t in range(1, len(sentence)):
            for i in range(4):
                max_score = float("-inf")
                for j in range(4):
                    score = self.__calc_score(j, i, features[t]) + delta[t - 1][j]  # if t >= 1 else self.__calc_score(j, i, features[t])
                    if score > max_score:
                        max_score = score
                        max_index = j

                # print(f"t:{t} i:{i} max_index:{max_index} max_score:{max_score}")
                delta[t][i] = max_score
                psi[t][i] = max_index

        # 3 终止 Ending
        max_score = float("-inf")
        for i in range(4):
            score = delta[-1][i]
            if score > max_score:
                max_score = score
                y_t = i
        y.append(y_t)

        # 4 回溯 Backtracking
        for t in reversed(range(1, len(sentence))):
            y_t = psi[t][y_t]
            y.append(y_t)

        y.reverse()
        return y

    def __embed_tag(self, last_tag, current_tag, feature):
        for i in range(len(feature)):
            feature[i] += f" {current_tag}"
        # if last_tag != -1:
        feature.append(f"8 {last_tag} {current_tag}")
        # feature.append("0")  # Bias
        return feature

    def __calc_score(self, last_tag, current_tag, feature):
        return self.__calc_dot_product(
            self.__embed_tag(last_tag, current_tag, feature.copy()))

    def __calc_dot_product(self, feature):
        result = 0
        for k in feature:
            result += self.__get_w_k(k)
        return result

    class Weight:
        def __init__(self, current_step):
            self.value = 0
            self.accumulated = 0
            self.last_step = current_step
            self.averaged_value = 0

    def __update_w_k(self, k, delta):
        if k not in self.w:
            self.w[k] = Perceptron.Weight(self.current_step)

        self.w[k].accumulated += self.w[k].value * (self.current_step - self.w[k].last_step)
        self.w[k].last_step = self.current_step
        self.w[k].value += delta
        self.w[k].averaged_value = None

    def __get_w_k(self, k):
        if k not in self.w:
            return 0

        if self.w[k].last_step != self.current_step \
                or self.w[k].averaged_value is None:
            self.w[k].accumulated += self.w[k].value * (self.current_step - self.w[k].last_step)
            self.w[k].last_step = self.current_step
            self.w[k].averaged_value = (self.w[k].accumulated + self.w[k].value) / self.current_step

        return self.w[k].averaged_value

        # return self.w[k].value

perceptron = Perceptron()
tags, _ = utils.read_hmm_tagged_sentences("small_msra_test.txt")
sentences = utils.read_plain_sentences("small_msra_test.txt")
perceptron.train(sentences, tags)
print(perceptron.predict("“吃屎的东西，连一捆麦也铡不动呀？"))
# perceptron.train(["我想吃饭", "运动真好"], [[3, 3, 0, 2], [0, 2, 3, 3]])
# perceptron.train(["我想吃饭", "运动真好"], [[3, 3, 0, 2], [0, 1, 2, 3]])
# perceptron.train(["我想吃饭"], [[3, 3, 0, 2]])
# perceptron.train(["运动真好"], [[0, 2, 3, 3]])
# perceptron.train(["运动员"], [[0, 2, 3]])
# perceptron.train(["运动真好"], [[0, 2, 3, 3]])
# print(perceptron.predict("我想吃饭")[0])
# print(perceptron.predict("运动真好")[0])
# print(perceptron.predict("运动员")[0])
# print(perceptron.predict("我想运动")[0])
# print(Perceptron.extract_features("迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）"))