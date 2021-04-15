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


def export_sentences(sentences, filename):
    content = '\n'.join(['  '.join(s) for s in sentences])
    with open(filename, "w") as f:
        f.write(content)
