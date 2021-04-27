# ========================================
# Bi-gram 二元语法分词
# ========================================

import utils
from utils import BOS, EOS
from math import log
from sys import maxsize

"""
总体步骤:
    1. 统计一元词频
    2. 统计二元词频
    3. 切词生成词网
    4. 计算最大概率路径
"""


# 1. 统计一元词频
# 2. 统计二元词频
uni_freq, bi_freq = utils.read_bigram_words("short.txt")
vocabulary = set(uni_freq.keys())
max_word_len = max([len(w) for w in vocabulary])

# 3. 切词生成词网
# test_sen = "共同创造美好的新世纪"
# test_sen = "一起共同创造美好的二十一世纪"
# test_sen = "共同创造美好的新世纪——二○○一年新年贺词"
test_sen = "（二○○○年十二月三十一日）（附图片1张）"
test_sen_len = len(test_sen)
word_net = []
for i in range(0, len(test_sen)):
    word_net.append([])
    for j in range(1, min(max_word_len, test_sen_len - i) + 1):
        if test_sen[i:i + j] in vocabulary:
            word_net[i].append(test_sen[i:i + j])
    # if len(word_net[i]) == 0:  # 避免断路
    #     word_net[i].append(test_sen[i])  # TODO: 精确判断是否断路
word_net.insert(0, [BOS])
word_net.append([EOS])
print(word_net)

# 4. 计算最大概率路径
def calc_segmentation_probability(w1, w2, uni_freq, bi_freq):
    if bi_freq[w1][w2] == 0:
        return -log(0.0001)

    p = -log(bi_freq[w1][w2] / uni_freq[w1])
    # print(w1, w2, p)
    return p

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
            print("到达终点")
            break

        # 松弛
        u = min_index
        for v in set(range(length)) - visited:
            if dist[v] > dist[u] + graph[u][v]:
                print(f"发现新极短路径 u:{u} v:{v} 原距离:{dist[v]} 新距离:{dist[u] + graph[u][v]}")
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
    print("最终求解路径：", path)

    return path

def construct_graph(word_net, uni_freq, bi_freq):
    length = len(word_net)
    graph = [[maxsize for _ in range(length)] for _ in range(length)]

    for i in range(1, length):
        for w2 in word_net[i]:
            for j in range(1, min(max_word_len, i) + 1):
                for w1 in word_net[i - j]:
                    if (len(w1) == j and w1 != BOS) or (j == 1 and w1 == BOS):
                        graph[i - j][i] = calc_segmentation_probability(w1, w2, uni_freq, bi_freq)
                        print(f"写入可能分割 w1:{w1} {i-j} w2:{w2} {i} 概率负对数:{graph[i - j][i]}")

    return graph

def convert_path_to_segmentation(sentence, path):
    seg = []

    for i in range(2, len(path)):
        seg.append(sentence[path[i - 1] - 1:path[i] - 1])

    return seg

# w1, w2 = "共同", "创造"
# w1, w2 = "，", "先生"
# p = calc_segmentation_probability(w1, w2, uni_freq, bi_freq)
# print(p)
graph = construct_graph(word_net, uni_freq, bi_freq)
# print(graph)
path = dijkstra_shortest_path(graph)
seg = convert_path_to_segmentation(test_sen, path)
print(seg)
