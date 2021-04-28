# ========================================
# Bi-gram 二元语法分词
# ========================================

from utils import BOS, EOS
from math import log
from sys import maxsize


class BiGram:
    def __init__(self, uni_freq, bi_freq):
        self.uni_freq = uni_freq
        self.bi_freq = bi_freq
        self.vocabulary = set(uni_freq.keys())
        self.max_word_len = max([len(w) for w in self.vocabulary])
        self.total_word_count = len(uni_freq)
        self.total_word_frequency = sum(uni_freq.values())

        self.SMOOTHING_LAMBDA = 0.1
        self.SMOOTHING_MU = 1 / self.total_word_frequency + 0.00001

    def coarse_word_net_segment(self, sentence, vocabulary):
        sen_len = len(sentence)
        word_net = []
        for i in range(0, len(sentence)):
            word_net.append([])
            word_net[i].append(sentence[i])  # 默认切分单字词
            for j in range(2, min(self.max_word_len, sen_len - i) + 1):
                if sentence[i:i + j] in vocabulary:
                    word_net[i].append(sentence[i:i + j])
        word_net.insert(0, [BOS])
        word_net.append([EOS])
        return word_net

    def calc_plain_probability(self, w1, w2):
        if self.bi_freq[w1][w2] == 0:
            return -log(0.0001)  # 将0替换为较小概率

        p = -log(self.bi_freq[w1][w2] / self.uni_freq[w1])
        # print(w1, w2, p)
        return p

    def calc_smoothed_probability(self, w1, w2):
        """
        P_smoothed(w2|w1) = (1 - l) * ((1 - m) * c(w1w2) / c(w1) + m) + l * ((c(w2) + 1) / N + |V|)
        P经验平滑公式；P的总和等于1
        """
        return \
            -log((1 - self.SMOOTHING_LAMBDA) * ((1 - self.SMOOTHING_MU) * self.bi_freq[w1][w2] / (self.uni_freq[w1] + self.SMOOTHING_MU)) \
                 + self.SMOOTHING_LAMBDA * ((self.uni_freq[w2] + 1) / (self.total_word_frequency + self.total_word_count)))

    @staticmethod
    def dijkstra_shortest_path(graph):
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

    def construct_graph(self, word_net):
        length = len(word_net)
        graph = [[maxsize for _ in range(length)] for _ in range(length)]

        for i in range(1, length):
            for w2 in word_net[i]:
                for j in range(1, min(self.max_word_len, i) + 1):
                    for w1 in word_net[i - j]:
                        if (len(w1) == j and w1 != BOS) or (j == 1 and w1 == BOS):
                            # graph[i - j][i] = calc_plain_probability(w1, w2, uni_freq, bi_freq)
                            graph[i - j][i] = self.calc_smoothed_probability(w1, w2)
                            # print(f"写入可能分割 w1:{w1} {i - j} w2:{w2} {i} 概率负对数:{graph[i - j][i]}")

        return graph

    def convert_path_to_segmentation(self, sentence, path):
        seg = []

        for i in range(2, len(path)):
            seg.append(sentence[path[i - 1] - 1:path[i] - 1])

        return seg

    def segment_sentences(self, sentences):
        results = []

        for sen in sentences:
            word_net = self.coarse_word_net_segment(sen, self.vocabulary)
            graph = self.construct_graph(word_net)
            path = self.dijkstra_shortest_path(graph)
            seg = self.convert_path_to_segmentation(sen, path)
            results.append(seg)

        return results
