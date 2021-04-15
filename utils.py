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
        return f.read().splitlines()


# def set_item(obj, key, value):
#     if obj[key] is not None:
#         obj[key] =