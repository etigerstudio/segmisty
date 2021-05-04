# ========================================
# Fast Structured Averaged Perceptron 快速结构化平均感知机
# ========================================

from utils import NA_STR, Evaluator
import json
import utils
import time
from collections import deque
import os


class Perceptron:
    __slots__ = "weights", "current_step", "name", "best_f1", "saved_models"

    def __init__(self, name=None):
        self.weights = self.__generate_weight_structure(7)
        self.current_step = 0
        self.name = name
        self.best_f1 = 0
        self.saved_models = deque()

    def train(self, sentences, tag_set, max_epoch=250, evaluate_sentences=None, evaluate_tag_set=None, save_models=True):
        for i in range(1, max_epoch + 1):
            converged = True
            evaluator = Evaluator()
            for sen, tags in zip(sentences, tag_set):
                y_predict, features = self.predict(sen)

                predict_sen = utils.join_sequential_tagged_sentences([y_predict], [sen])[0]
                truth_sen = utils.join_sequential_tagged_sentences([tags], [sen])[0]
                evaluator.count(truth_sen, predict_sen)

                if y_predict != tags:
                    converged = False
                    self.__optimize(features, tags, y_predict)

            if converged:
                print(f"converged after iter {i}")
                break

            # print(f"iter:{i}")
            _, _, f1, _, formatted_string = evaluator.get_statistics()
            print(f"training:{i} {formatted_string}")

            if save_models:
                if evaluate_sentences is None or evaluate_tag_set is None:
                    if i % 20 == 0:
                        self.export(f"{self.name}-autosave-{time.time()}-{i}.perceptron")
                else:  # 动态保存在验证集上表现好的模型 Dynamically save models based on performance on validation set
                    p, r, f1, elapsed_time, formatted_string = self.evaluate(evaluate_sentences, evaluate_tag_set)
                    print(f"evaluation:{i} {formatted_string}")
                    if f1 > self.best_f1:
                        self.best_f1 = f1
                        save_filename = f"{self.name}-f1{f1:.4}-{time.time()}-{i}.perceptron"
                        self.saved_models.append(save_filename)
                        self.export(save_filename)
                        print(f"saved model f1:{f1} name:{save_filename}")
                        if len(self.saved_models) > 5:
                            purge_filename = self.saved_models.popleft()
                            os.remove(purge_filename)
                            print(f"purged model name:{purge_filename}")

        if save_models:
            self.export(f"{self.name}-{time.time()}-{i}.perceptron")
        print(f"training concluded. total iters: {i}.")

    def predict(self, sentence):
        features = self.__extract_features(sentence)
        y_predict = self.__perform_viterbi(sentence, features)
        return y_predict, features

    def __optimize(self, features, y_truth, y_predict):
        self.current_step += 1
        for i in range(len(features)):
            if y_truth[i] != y_predict[i]:  # 奖励正确序列，惩罚错误序列 BOS: -1
                self.__update_weight(y_truth[i] if i >= 1 else -1, y_truth[i], features[i], 1)
                self.__update_weight(y_predict[i] if i >= 1 else -1, y_predict[i], features[i], -1)

    def __update_weight(self, last_tag, current_tag, feature, delta):
        for i in range(len(feature)):
            self.__update_w_k(self.weights[i][current_tag], feature[i], delta)
        self.__update_w_k(self.weights[7], (last_tag + 1) + current_tag * 5, delta)
        # print(f"updated:{k} averaged:{self.__get_w_k(k)} value:{self.w[k].value} delta:{delta} accumulated:{self.w[k].accumulated}")

    @staticmethod
    def __extract_features(sentence):
        features = []

        for i in range(len(sentence)):
            l2 = sentence[i - 2] if i >= 2 else NA_STR
            l1 = sentence[i - 1] if i >= 1 else NA_STR
            mid = sentence[i]
            r1 = sentence[i + 1] if i < len(sentence) - 1 else NA_STR
            r2 = sentence[i + 2] if i < len(sentence) - 2 else NA_STR

            features.append([
                l1,
                mid,
                r1,
                l2 + l1,
                l1 + mid,
                mid + r1,
                r1 + r2
            ])

        return features

    def __perform_viterbi(self, sentence, features):
        if len(sentence) == 0:
            return []

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

    def __calc_score(self, last_tag, current_tag, feature):
        result = 0
        for i in range(len(feature)):
            result += self.__get_w_k(self.weights[i][current_tag], feature[i])
        result += self.__get_w_k(self.weights[7], (last_tag + 1) + current_tag * 5)
        return result

    class Weight:
        __slots__ = "value", "accumulated", "last_step", "averaged_value"

        def __init__(self, current_step):
            self.value = 0
            self.accumulated = 0
            self.last_step = current_step
            self.averaged_value = 0

        def export(self):
            return {
                "value": self.value,
                "accumulated": self.accumulated,
                "last_step": self.last_step,
                "averaged_value": self.averaged_value,
            }

        @classmethod
        def load(cls, dict):
            w = cls(dict["last_step"])
            w.value = dict["value"]
            w.accumulated = dict["accumulated"]
            w.averaged_value = dict["averaged_value"]
            return w

    def __update_w_k(self, w, k, delta):
        if k not in w:
            w[k] = Perceptron.Weight(self.current_step)

        wk = w[k]
        wk.accumulated += wk.value * (self.current_step - wk.last_step)
        wk.last_step = self.current_step
        wk.value += delta
        wk.averaged_value = None

    def __get_w_k(self, w, k):
        if k not in w:
            return 0

        wk = w[k]
        if wk.last_step != self.current_step \
                or wk.averaged_value is None:
            wk.accumulated += wk.value * (self.current_step - wk.last_step)
            wk.last_step = self.current_step
            wk.averaged_value = (wk.accumulated + wk.value) / self.current_step

        return wk.averaged_value

    def export(self, filename):
        obj = {"weights": {k: w.export() for k, w in self.w.items()}, "current_step": self.current_step}
        with open(filename, "w") as f:
            json.dump(obj, f, ensure_ascii=False)
        print(f"saved to {filename}")

    def evaluate(self, sentences, tag_set, export_results=False, export_filename=None):
        predicted_states = []
        evaluator = utils.Evaluator()
        for sen, tags in zip(sentences, tag_set):
            y_predict = self.predict(sen)[0]
            predict_sen = utils.join_sequential_tagged_sentences([y_predict], [sen])[0]
            truth_sen = utils.join_sequential_tagged_sentences([tags], [sen])[0]
            predicted_states.append(y_predict)
            evaluator.count(truth_sen, predict_sen)
        p, r, f1, elapsed_time, formatted_string = evaluator.get_statistics()
        print(f"eval: {formatted_string}")

        if export_results:
            utils.export_sequential_tagged_sentences(predicted_states, sentences, export_filename)

        return p, r, f1, elapsed_time, formatted_string

    @classmethod
    def load(cls, filename, name=None):
        p = cls(name)

        with open(filename, "r") as f:
            obj = json.load(f)
        p.w = {k: Perceptron.Weight.load(w_dict) for k, w_dict in obj["weights"].items()}
        p.current_step = obj["current_step"]
        print(f"loaded from {filename}")
        return p

    def __generate_weight_structure(self, unigram_count):
        w = [[{} for _ in range(5)] for _ in range(unigram_count)]  # 5 = 4 states + BOS(-1)
        w.append([Perceptron.Weight(0) for _ in range(20)])  # 20 = 5(last_tag) * 4(current_tag)
        return w
