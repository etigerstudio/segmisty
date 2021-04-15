# ========================================
# Utilities 工具函数
# ========================================

def read_plain_sentences(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        lines = [l.replace('  ', '') for l in lines]
        return lines


def read_vocabulary_dataset(filename):
    with open(filename, "r") as f:
        return set(f.read().splitlines())


def export_plain_sentences(sentences, filename):
    content = '\n'.join(['  '.join(s) for s in sentences])
    with open(filename, "w") as f:
        f.write(content)


def read_hmm_tagged_sentences(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        lines = [l.split('  ') for l in lines]

        states = []
        observations = []
        for l in lines:
            s, ob = [], []
            for w in l:
                w_len = len(w)
                if w_len == 1:
                    s.append(3)  # S
                else:
                    s.extend([0, *([1] * (w_len - 2)), 2])  # B, M, ... , E
                ob.extend(string_to_ints(w))

            states.append(s)
            observations.append(ob)

        return states, observations


def export_hmm_tagged_sentences(states, observations, filename):
    if len(states) != len(observations):
        raise ValueError

    sentences = []
    for i in range(len(states)):
        sentence = ''
        string = ints_to_string(observations[i])
        for j in range(len(states[i])):
            tag, char = states[i][j], string[j]
            sentence += char if tag == 0 or tag == 1 else char + '  '
        sentences.append(sentence)

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