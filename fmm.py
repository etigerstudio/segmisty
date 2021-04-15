# ========================================
# Forward Maximum Matching 前向最大匹配分词
# ========================================

def tokenize(vocabulary, sentences):
    max_word_len = max([len(w) for w in vocabulary])  # 词汇表单词最大长度
    all_words = []  # 整个篇章的分词结果数组

    for s in sentences:
        cur_pos, s_len = 0, len(s)  # 当前匹配起始下标、当前句长度
        s_words = []  # 当前句的分词结果数组

        while cur_pos < s_len:
            cur_word_len = min(max_word_len, s_len - cur_pos)  # 当前匹配单词长度

            while True:
                cur_word = s[cur_pos:cur_pos + cur_word_len] # 当前匹配单词

                if cur_word_len == 1 or cur_word in vocabulary:  # 单字词 或 词汇表中存在当前单词
                    s_words.append(cur_word)
                    cur_pos += cur_word_len
                    break
                else:
                    cur_word_len -= 1

        all_words.append(s_words)

    return all_words