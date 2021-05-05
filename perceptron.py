# ========================================
# Structured Averaged Perceptron 结构化平均感知机
# ========================================

from utils import NA_STR, Evaluator
import json
import utils
import time
from collections import deque
import os


class Perceptron:
    __slots__ = "weights", \
                "current_step", \
                "current_epoch", \
                "name", \
                "best_f1", \
                "saved_models", \
                "regularization", \
                "kernel", \
                "l1_decay", \
                "l2_decay", \
                "require_averaged_weights"

    def __init__(self, name=None):
        self.weights = self.__generate_weight_structure(7)
        self.current_step = 0
        self.current_epoch = 0
        self.name = name
        self.best_f1 = 0
        self.saved_models = deque()
        self.regularization = 0  # 0: none 1: L1 2: L2
        self.kernel = 0  # 0: linear 1: quadratic
        self.l1_decay = 0.0001
        self.l2_decay = 0.999
        self.require_averaged_weights = True

    def train(self, sentences, tag_set, max_iter=250, kernel=0, evaluate_sentences=None, evaluate_tag_set=None, save_models=True):
        print(f"training began. current_epoch:{self.current_epoch} max_iter:{max_iter}")
        self.require_averaged_weights = False
        for i in range(max_iter):
            converged = True
            self.current_epoch += 1
            evaluator = Evaluator()
            for sen, tags in zip(sentences, tag_set):
                y_predict, features = self.predict(sen)
                # print(f"truth:{tags} predict:{y_predict}")

                predict_sen = utils.join_sequential_tagged_sentences([y_predict], [sen])[0]
                truth_sen = utils.join_sequential_tagged_sentences([tags], [sen])[0]
                evaluator.count(truth_sen, predict_sen)

                if y_predict != tags:
                    converged = False
                    self.__optimize(features, tags, y_predict)

            if converged:
                print(f"converged after iter {self.current_epoch}")
                break

            # print(f"iter:{self.current_epoch}")
            _, _, f1, _, formatted_string = evaluator.get_statistics()
            print(f"training iter {self.current_epoch}: {formatted_string}")

            if save_models:
                if evaluate_sentences is None or evaluate_tag_set is None:
                    if self.current_epoch % 20 == 0:
                        self.export(f"{self.name}-autosave-{time.time()}-{self.current_epoch}.perceptron")
                else:  # 动态保存在验证集上表现好的模型 Dynamically save models based on performance on validation set
                    self.require_averaged_weights = True
                    p, r, f1, elapsed_time, formatted_string = self.evaluate(evaluate_sentences, evaluate_tag_set)
                    self.require_averaged_weights = False
                    print(f"evaluation {self.current_epoch}: {formatted_string}")
                    if f1 > self.best_f1:
                        self.best_f1 = f1
                        save_filename = f"{self.name}-f1{f1:.4}-{time.time()}-{self.current_epoch}.perceptron"
                        self.saved_models.append(save_filename)
                        self.export(save_filename)
                        print(f"saved model f1:{f1} name:{save_filename}")
                        if len(self.saved_models) > 5:
                            purge_filename = self.saved_models.popleft()
                            os.remove(purge_filename)
                            print(f"purged model name:{purge_filename}")

        if save_models:
            self.export(f"{self.name}-{time.time()}-{self.current_epoch}.perceptron")
        print(f"training concluded. total iters: {self.current_epoch}.")
        self.require_averaged_weights = True

    def predict(self, sentence):
        features = self.__extract_features(sentence)
        y_predict = self.__perform_viterbi(sentence, features)
        return y_predict, features

    def __optimize(self, features, y_truth, y_predict):
        self.current_step += 1
        for i in range(len(features)):
            # 奖励正确序列，惩罚错误序列 BOS: -1
            # 按需优化
            if y_truth[i] != y_predict[i]:
                self.__update_weight(y_truth[i - 1] if i >= 1 else -1, y_truth[i], features[i], 1, True)
                self.__update_weight(y_predict[i - 1] if i >= 1 else -1, y_predict[i], features[i], -1, True)
            else:
                if i >= 1 and y_truth[i - 1] != y_predict[i - 1]:
                    self.__update_weight(y_truth[i - 1] if i >= 1 else -1, y_truth[i], features[i], 1, False)
                    self.__update_weight(y_predict[i - 1] if i >= 1 else -1, y_predict[i], features[i], -1, False)

    def __update_weight(self, last_tag, current_tag, feature, delta, update_unigram):
        if update_unigram:
            for i in range(len(feature)):
                self.__update_w_k(self.weights[i][current_tag], feature[i], delta)
                # print(f"updated:{feature[i]} [{i}][{current_tag}] {self.__get_w_k(self.weights[i][current_tag], feature[i])} {self.weights[i][current_tag][feature[i]].value} {delta} {self.weights[i][current_tag][feature[i]].accumulated}")
        # else:
        #     print(f"unigram skipped: {current_tag} {feature} {delta}")
        self.__update_w_k(self.weights[7], (last_tag + 1) + current_tag * 5, delta, list_based_w=True)
        # print(f"updated:{last_tag}>{current_tag}:{last_tag + 1 + current_tag * 5} {self.__get_w_k(self.weights[7], last_tag + 1 + current_tag * 5)} {self.weights[7][(last_tag + 1) + current_tag * 5].value} {delta} {self.weights[7][(last_tag + 1) + current_tag * 5].accumulated}")

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
        if self.kernel == 0:  # Linear
            if self.require_averaged_weights:
                for i in range(len(feature)):
                    result += self.__get_w_k(self.weights[i][current_tag], feature[i])
                result += self.__get_w_k(self.weights[7], (last_tag + 1) + current_tag * 5, list_based_w=True)
            else:
                for i in range(len(feature)):
                    w = self.weights[i][current_tag]
                    result += w[feature[i]].value if feature[i] in w else 0
                result += self.weights[7][(last_tag + 1) + current_tag * 5].value
        else:  # Quadratic
            if self.require_averaged_weights:
                for i in range(len(feature)):
                    result += self.__get_w_k(self.weights[i][current_tag], feature[i]) ** 2
                result += self.__get_w_k(self.weights[7], (last_tag + 1) + current_tag * 5, list_based_w=True) ** 2
            else:
                for i in range(len(feature)):
                    w = self.weights[i][current_tag]
                    result += w[feature[i]].value if feature[i] ** 2 in w else 0
                result += self.weights[7][(last_tag + 1) + current_tag * 5].value ** 2

        return result

    class Weight:
        __slots__ = "value", "accumulated", "last_step", "averaged_value"

        def __init__(self, current_step):
            self.value = 0
            self.accumulated = 0
            self.last_step = current_step
            self.averaged_value = 0

        def __repr__(self):
            return str(self.value)  # show accumulated value when inspected

    def __regularize_wk(self, value, accumulated, steps, regularization):
        if steps == 0:
            return value, accumulated

        if regularization == 1:  # L1
            reg_value = max(value - steps * self.l1_decay, 0) if value > 0 else -min(value + steps * self.l1_decay, 0)
            reg_accumulated = (value + reg_value) * min(steps, value / self.l1_decay) / 2
        else:  # L2
            reg_value = value * self.l2_decay ** steps
            reg_accumulated = value * (1 - self.l2_decay ** steps) / 1 - self.l2_decay
        return reg_value, reg_accumulated

    def __update_w_k(self, w, k, delta, list_based_w=False):
        if not list_based_w and k not in w:
            w[k] = Perceptron.Weight(self.current_step)

        wk = w[k]
        if self.regularization == 0:
            wk.accumulated += wk.value * (self.current_step - wk.last_step)
        else:
            wk.accumulated, wk.value = self.__regularize_wk(wk.value, wk.accumulated, self.current_step - wk.last_step, self.regularization)
        wk.value += delta
        wk.averaged_value = None
        wk.last_step = self.current_step

    def __get_w_k(self, w, k, list_based_w=False):
        if not list_based_w and k not in w:
            return 0

        wk = w[k]
        if wk.last_step != self.current_step \
                or wk.averaged_value is None:
            if self.regularization == 0:
                wk.accumulated += wk.value * (self.current_step - wk.last_step)
            else:
                wk.accumulated, wk.value = self.__regularize_wk(wk.value, wk.accumulated, self.current_step - wk.last_step, self.regularization)
            wk.averaged_value = (wk.accumulated + wk.value) / self.current_step
            wk.last_step = self.current_step

        return wk.averaged_value

    def export(self, filename):
        weights = [[{k: self.__weight_to_list(w, k) for k in w} for w in ws] for ws in self.weights[0:-1]]  # unigram
        weights.append([self.__weight_to_list(self.weights[-1], i, list_based_w=True) for i in range(len(self.weights[-1]))])  # bigram

        obj = {
            "name": self.name,
            "weights": weights,
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "regularization": self.regularization,
            "kernel": self.kernel
        }

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
        # print(f"eval: {formatted_string}")

        if export_results:
            utils.export_sequential_tagged_sentences(predicted_states, sentences, export_filename)

        return p, r, f1, elapsed_time, formatted_string

    @staticmethod
    def load(filename, name=None):
        with open(filename, "r") as f:
            obj = json.load(f)

        p = Perceptron(name)
        p.name = obj["name"]
        p.weights = [[{k: Perceptron.__weight_from_list(wk) for k, wk in w.items()} for w in ws] for ws in obj["weights"][0:-1]]  # unigram
        p.weights.append([Perceptron.__weight_from_list(obj["weights"][-1][i]) for i in range(len(obj["weights"][-1]))])  # bigram
        p.current_step = obj["current_step"]
        p.current_epoch = obj["current_epoch"]
        p.regularization = obj["regularization"]
        p.kernel = obj["kernel"]

        print(f"loaded from {filename}")
        return p

    @staticmethod
    def __weight_from_dict(dict):
        w = Perceptron.Weight(dict["last_step"])
        w.value = dict["value"]
        w.accumulated = dict["accumulated"]
        w.averaged_value = dict["averaged_value"]
        return w

    def __weight_to_dict(self, w, k, list_based_w=False):
        _ = self.__get_w_k(w, k, list_based_w)  # trigger averaged weight value update
        wk = w[k]
        return {
            "value": wk.value,
            "accumulated": wk.accumulated,
            "last_step": wk.last_step,
            "averaged_value": wk.averaged_value,
        }

    @staticmethod
    def __weight_from_list(list):
        w = Perceptron.Weight(list[2])
        w.value = list[0]
        w.accumulated = list[1]
        w.averaged_value = list[3]
        return w

    def __weight_to_list(self, w, k, list_based_w=False):
        _ = self.__get_w_k(w, k, list_based_w)  # trigger averaged weight value update
        wk = w[k]
        return [
            wk.value,
            wk.accumulated,
            wk.last_step,
            wk.averaged_value,
        ]

    def __generate_weight_structure(self, unigram_count):
        w = [[{} for _ in range(4)] for _ in range(unigram_count)]  # 4 states
        w.append([Perceptron.Weight(0) for _ in range(20)])  # 20 = 5(last_tag, including BOS) * 4(current_tag)
        return w
