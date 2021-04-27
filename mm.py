# ========================================
# Maximum Matching 最大匹配分词
# ========================================

from enum import Enum


class MMMode(Enum):
    Forward = 1
    Reverse = 2
    Bidirectional = 3

def tokenize(vocabulary, sentences, mode=MMMode.Forward):
    max_word_len = max([len(w) for w in vocabulary])  # 词汇表单词最大长度
    all_words = []  # 整个篇章的分词结果数组

    for s in sentences:
        if mode == MMMode.Forward:
            s_words = __tokenize_one_pass(vocabulary, max_word_len, s, True)
        elif mode == MMMode.Reverse:
            s_words = __tokenize_one_pass(vocabulary, max_word_len, s, False)
        elif mode == MMMode.Bidirectional:
            f_words = __tokenize_one_pass(vocabulary, max_word_len, s, True)
            r_words = __tokenize_one_pass(vocabulary, max_word_len, s, False)
            s_words = __choose_better_tokenization(f_words, r_words)  # 选择前向和后向中效果更好的分词结果，默认后向
        else:
            raise NotImplementedError

        all_words.append(s_words)

    return all_words


def __tokenize_one_pass(vocabulary, max_word_len, sentence, is_forward):
    cur_pos, s_len = 0, len(sentence)  # 当前匹配位置、当前句长度
    s_words = []  # 当前句的分词结果数组

    while cur_pos < s_len:  # 沿一方向扫描该句
        cur_word_len = min(max_word_len, s_len - cur_pos)  # 当前匹配单词长度

        while True:
            cur_word = sentence[cur_pos:cur_pos + cur_word_len] if is_forward \
                else sentence[s_len - cur_pos - cur_word_len:s_len - cur_pos]  # 当前待匹配单词

            if cur_word_len == 1 or cur_word in vocabulary:  # 单字词 或 词汇表中存在当前单词
                s_words.append(cur_word)  # 匹配结束，记录单词
                cur_pos += cur_word_len
                break
            else:
                cur_word_len -= 1

    if not is_forward:
        s_words.reverse()
    return s_words


def __choose_better_tokenization(t1, t2):
    l1, l2 = len(t1), len(t2)
    if l1 > l2:
        return t2
    elif l1 < l2:
        return t1
    else:  # l1 == l2
        s_count1 = len([None for s in t1 if len(t1) == 1])  # t1中单字词数量
        s_count2 = len([None for s in t2 if len(t2) == 1])
        return t1 if s_count1 < s_count2 else t2  # 默认返回后者