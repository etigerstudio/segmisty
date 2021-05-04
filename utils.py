# ========================================
# Utilities 工具函数
# ========================================

from collections import defaultdict
import re
import time

BOS = " BOS "
EOS = " EOS "
NA_STR = "#"
NA_CODE = 0xA1AA  # in user-defined 1 of GBK encoding
# 正则规则测试见https://regexr.com/5rqrb
numeric_re = re.compile("^[十百千万亿]分之[零一二三四五六七八九十百千万亿]+(点[零一二三四五六七八九十])?|^[0-9零○〇一二两三四五六七八九十廿百千万亿壹贰叁肆伍陆柒捌玖拾佰仟]+[年月日时分秒]|^[-－]?\\d+(.\\d)?[%％]?")
NUMBER_CHARS = set("0123456789零○〇一二两三四五六七八九十廿百千万亿壹贰叁肆伍陆柒捌玖拾佰仟")
# 正则规则测试见https://regexr.com/5rr5u
english_re = re.compile("^[\w－\-．.／/:;：；<=>?＜＝＞？@＠_＿\\\\]+", flags=re.ASCII)
ENGLISH_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
TAGS_MAP = {0: "B", 1: "M", 2: "E", 3: "S"}
STATES_MAP = {"B": 0, "M": 1, "E": 2, "S": 3}

def read_plain_sentences(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        lines = [l.replace('  ', '') for l in lines]
        return lines


def read_vocabulary_dataset(filename):
    with open(filename, "r") as f:
        return set(f.read().splitlines())


def read_bigram_words(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        lines = [l.strip().split('  ') for l in lines]

        uni_freq = defaultdict(int)
        for l in lines:
            for w in l:
                uni_freq[w] += 1
        uni_freq[BOS] = len(lines)
        uni_freq[EOS] = len(lines)
                
        bi_freq = defaultdict(lambda: defaultdict(int))
        for l in lines:
            if len(l) > 0:
                bi_freq[BOS][l[0]] += 1
                bi_freq[l[-1]][EOS] += 1
                for i in range(1, len(l)):
                    bi_freq[l[i - 1]][l[i]] += 1

        return uni_freq, bi_freq


def export_plain_sentences(sentences, filename):
    content = '\n'.join(['  '.join(s) for s in sentences])
    with open(filename, "w") as f:
        f.write(content)


def read_sequential_tagged_sentences(filename, convert_character_to_int=False):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        lines = [l.strip().split('  ') for l in lines]

        states = []
        observations = []
        for l in lines:
            s, ob = [], []
            for w in l:
                w_len = len(w)
                if w_len == 0:
                    continue
                if w_len == 1:
                    s.append(3)  # S
                else:
                    s.extend([0, *([1] * (w_len - 2)), 2])  # B, M, ... , E
                if convert_character_to_int:
                    ob.extend(string_to_ints(w))

            states.append(s)
            if convert_character_to_int:
                observations.append(ob)

        if convert_character_to_int:
            return states, observations
        else:
            return states, [''.join(l) for l in lines]


def join_sequential_tagged_sentences(states, observations, convert_int_to_character=False):
    if len(states) != len(observations):
        raise ValueError

    sentences = []
    for i in range(len(states)):
        sentence = ''
        if convert_int_to_character:
            string = ints_to_string(observations[i])
        else:
            string = observations[i]
        for j in range(len(states[i])):
            tag, char = states[i][j], string[j]
            sentence += char if tag == 0 or tag == 1 or j == len(states[i]) - 1 else char + '  '
        sentences.append(sentence)

    return sentences


def export_sequential_tagged_sentences(states, observations, filename, convert_int_to_character=False):
    sentences = join_sequential_tagged_sentences(states, observations, convert_int_to_character)

    with open(filename, "w") as f:
        f.write('\n'.join(sentences))


def string_to_ints(string, encoding='gbk'):
    ints = []
    for c in string:
        ints.append(int.from_bytes(c.encode(encoding), byteorder='little'))
    return ints


def ints_to_string(ints, encoding='gbk'):
    string = ''
    for i in ints:
        string += i.to_bytes((i.bit_length() + 7) // 8, byteorder='little').decode(encoding)
    return string


def try_atomic_segmentation(sentence):
    if sentence[0] in NUMBER_CHARS:
        exp = numeric_re
    elif sentence[0] in ENGLISH_CHARS:
        exp = english_re
    else:
        return

    match = exp.match(sentence)
    if match:
        return match.span()[1]
    else:
        return


def generate_crfpp_compatible_file(input_filename, output_filename, with_tags=True):
    tag_set, sentences = read_sequential_tagged_sentences(input_filename)
    contents = ''
    for tags, sen in zip(tag_set, sentences):
        for i in range(len(tags)):
            contents += f"{sen[i]} {TAGS_MAP[tags[i]]}\n" if with_tags else f"{sen[i]}\n"
        contents += "\n"
    with open(output_filename, "w") as f:
        f.write(contents)


def transform_crfpp_results_to_sentences(input_filename, output_filename):
    with open(input_filename, "r") as f:
        sections = f.read().split("\n\n")

    sections = [[(c[0], c[-1]) for c in [l.split("\t") for l in s.split("\n")]] for s in sections]

    states, observations = [], []
    for s in sections:
        states_i, observations_i = [], []
        for observation, state in s:
            if observation == "" or state == "":
                continue
            states_i.append(STATES_MAP[state])
            observations_i.append(observation)
        states.append(states_i)
        observations.append(observations_i)

    export_sequential_tagged_sentences(states, observations, output_filename)


def evaluate_truth_and_predict(truth_filename, predict_filename):
    evaluator = Evaluator()
    with open(truth_filename, "r") as t, open(predict_filename, "r") as p:
        for tl, pl in zip(t.read().splitlines(), p.read().splitlines()):
            evaluator.count(tl, pl)
    return evaluator.get_statistics()


class Evaluator:
    def __init__(self):
        self.start_time = time.time()
        self.tp, self.truth, self.predict = 0, 0, 0  # True positive

    @staticmethod
    def __convert_sentence_to_set(sentence):
        word_set, offset = set(), 0
        words = sentence.split('  ')
        for w in words:
            word_set.add((offset, w))

        return word_set

    def count(self, truth_sen, predict_sen):
        truth_set = self.__convert_sentence_to_set(truth_sen)
        predict_set = self.__convert_sentence_to_set(predict_sen)
        tp_set = truth_set & predict_set

        self.truth += len(truth_set)
        self.predict += len(predict_set)
        self.tp += len(tp_set)

    def get_statistics(self):
        precision = self.tp / self.predict if self.predict != 0 else 0.0
        recall = self.tp / self.truth if self.truth != 0 else 0.0
        f1 = 2 * self.tp / (self.truth + self.predict) if self.truth + self.predict != 0 else 0.0
        elapsed_time = time.time() - self.start_time
        formatted_string = f"P:{precision:.4} R:{recall:.4} F1:{f1:.4} {elapsed_time:.4}s"
        return precision, recall, f1, elapsed_time, formatted_string
