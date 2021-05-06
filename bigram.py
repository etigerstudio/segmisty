# ========================================
# Bi-gram 二元语法分词
# ========================================

from utils import BOS, EOS
from math import log
from sys import maxsize
import utils


class BiGram:
    def __init__(self, uni_freq, bi_freq, enable_atomic_segmentation=True):
        self.uni_freq = uni_freq
        self.bi_freq = bi_freq
        self.vocabulary = set(uni_freq.keys())
        self.vocab_max_word_len = max([len(w) for w in self.vocabulary])
        self.total_word_count = len(uni_freq)
        self.total_word_frequency = sum(uni_freq.values())
        self.enable_atomic_segmentation = enable_atomic_segmentation

        self.SMOOTHING_LAMBDA = 0.1
        self.SMOOTHING_MU = 1 / self.total_word_frequency + 0.00001

    def __coarse_word_net_segment(self, sentence, vocabulary):
        sen_len = len(sentence)
        word_net = [{0} for _ in range(sen_len)]
        i = 0

        while i < sen_len:
            # 原子切分，目前包含数字、英文规则
            if self.enable_atomic_segmentation:
                seg = utils.try_atomic_segmentation(sentence[i:])
                if seg:
                    word_net[i].add(seg)  # 成功切分数字
                    i += seg
                    continue
                else:
                    word_net[i].add(1)  # 未成功切分
            else:
                word_net[i].add(1)  # 默认切分单字词

            for j in range(2, min(self.vocab_max_word_len, sen_len - i) + 1):
                if sentence[i:i + j] in vocabulary:
                    word_net[i].add(j)

            i += 1

        word_net.insert(0, {1})
        word_net.append({1})
        return word_net

    def __calc_smoothed_probability(self, w1, w2):
        """
        P_smoothed(w2|w1) = (1 - l) * ((1 - m) * c(w1w2) / c(w1) + m) + l * ((c(w2) + 1) / N)
        P经验平滑公式；P的总和等于1
        """
        return \
            -log((1 - self.SMOOTHING_LAMBDA) * (
                        (1 - self.SMOOTHING_MU) * self.bi_freq[w1][w2] / (self.uni_freq[w1] + self.SMOOTHING_MU)) \
                 + self.SMOOTHING_LAMBDA * (
                             (self.uni_freq[w2] + 1) / (self.total_word_frequency)))  # Should + |V| or not? Not yet.

    @staticmethod
    def __dijkstra_shortest_path(graph):
        """
        通过Dijkstra算法求解最短路径

        :param graph: 有向图权重矩阵
        :return: 最短路径分割点下标
        """
        length = len(graph[0])
        dist = [maxsize] * length
        dist[0] = 0

        visited = set()
        prev = [-1] * length

        while len(visited) < length:
            # 寻找下一个节点
            min = maxsize
            for u in set(range(length)) - visited:
                if dist[u] < min:
                    min = dist[u]
                    min_index = u

            # 判断是否已经到达终点
            if min_index == length - 1:
                # print("到达终点")
                break

            # 松弛
            u = min_index
            for v in set(range(length)) - visited:
                if dist[v] > dist[u] + graph[u][v]:
                    # print(f"发现新极短路径 u:{u} v:{v} 原距离:{dist[v]} 新距离:{dist[u] + graph[u][v]}")
                    dist[v] = dist[u] + graph[u][v]
                    prev[v] = u

            visited.add(u)

        # 回溯路径
        path = [length - 1]
        u = length - 1
        while prev[u] != -1:
            path.append(prev[u])
            u = prev[u]
        path.reverse()
        # print("最终求解路径：", path)

        return path

    def __construct_graph(self, sentence, word_net):
        if len(sentence) != len(word_net) - 2:
            raise ValueError

        length = len(word_net)
        graph = [[maxsize for _ in range(length)] for _ in range(length)]
        net_max_word_len = max([max(s) for s in word_net])

        for i in range(1, length):
            for l2 in word_net[i]:  # 后词的长度
                for l1 in range(1, min(net_max_word_len, i) + 1):  # 前词的长度
                    if l1 in word_net[i - l1]:
                        if i == 1:  # 句首特殊情况
                            graph[i - l1][i] = self.__calc_smoothed_probability(BOS, sentence[i - 1:i + l2 - 1])
                            # print(f"写入可能分割 w1:{BOS} {i - l1} w2:{sentence[i - 1:i + l2 - 1]} {i} 概率负对数:{graph[i - l1][i]}")
                        elif i == length - 1:  # 句尾特殊情况
                            graph[i - l1][i] = self.__calc_smoothed_probability(sentence[i - l1 - 1:i - 1], EOS)
                            # print(f"写入可能分割 w1:{sentence[i - l1 - 1:i - 1]} {i - l1} w2:{EOS} {i} 概率负对数:{graph[i - l1][i]}")
                        else:  # 句中标准情况
                            graph[i - l1][i] = self.__calc_smoothed_probability(sentence[i - l1 - 1:i - 1], sentence[i - 1:i + l2 - 1])
                            # print(f"写入可能分割 w1:{sentence[i - l1 - 1:i - 1]} {i - l1} w2:{sentence[i - 1:i + l2 - 1]} {i} 概率负对数:{graph[i - l1][i]}")

        return graph

    @staticmethod
    def __convert_path_to_segmentation(sentence, path):
        seg = []

        for i in range(2, len(path)):
            seg.append(sentence[path[i - 1] - 1:path[i] - 1])

        return seg

    def segment_sentences(self, sentences):
        results = []

        for i in range(len(sentences)):
            if i % 20 == 0:
                print(f"{i} / {len(sentences)}", end='\r')
            word_net = self.__coarse_word_net_segment(sentences[i], self.vocabulary)
            graph = self.__construct_graph(sentences[i], word_net)
            path = self.__dijkstra_shortest_path(graph)
            seg = self.__convert_path_to_segmentation(sentences[i], path)
            results.append(seg)

        return results
