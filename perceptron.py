# ========================================
# Structured Averaged Perceptron 结构化平均感知机
# ========================================

from utils import NA, Evaluator
import json
import utils
import time
from collections import deque
import os


class Perceptron:
    def __init__(self, name=None):
        self.w = {}
        self.current_step = 0
        self.name = name
        self.best_f1 = 0
        self.saved_models = deque()

    def train(self, sentences, tag_set, max_epoch=250, evaluate_filename=None):
        # last_f1 = float("-inf")

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
            if evaluate_filename is None:
                if i % 20 == 0:
                    self.export(f"{self.name}-autosave-{time.time()}-{i}.perceptron")
            else:  # 动态保存在验证集上表现好的模型 Dynamically save models based on performance on validation set
                p, r, f1, elapsed_time, formatted_string = self.evaluate(evaluate_filename)
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

            # if i >= 20 and f1 <= last_f1:
            #     print(f"iter:{i} f1({f1}) is worsened, early-stopping triggered")
            #     break
            # else:
            #     last_f1 = f1

        self.export(f"{self.name}-{time.time()}-{i}.perceptron")

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
            self.__update_w_k(k, delta)
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

    def export(self, filename):
        obj = {"weights": {k: w.export() for k, w in self.w.items()}, "current_step": self.current_step}
        with open(filename, "w") as f:
            json.dump(obj, f, ensure_ascii=False)
        print(f"saved to {filename}")

    def evaluate(self, filename, export_results=False):
        tag_set, sentences = utils.read_sequential_tagged_sentences(filename)
        predicted_states = []
        evaluator = utils.Evaluator()
        for sen, tags in zip(sentences, tag_set):
            y_predict = self.predict(sen)[0]
            predict_sen = utils.join_sequential_tagged_sentences([y_predict], [sen])[0]
            truth_sen = utils.join_sequential_tagged_sentences([tags], [sen])[0]
            predicted_states.append(y_predict)
            evaluator.count(truth_sen, predict_sen)
        p, r, f1, elapsed_time, formatted_string = evaluator.get_statistics()
        print(f"eval {filename}: {formatted_string}")

        if export_results:
            utils.export_sequential_tagged_sentences(predicted_states, sentences, "perceptron_pku_61.result.txt")

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