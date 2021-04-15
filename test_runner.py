import utils
import mm

if __name__ == '__main__':
    v = utils.read_vocabulary_dataset("training_vocab.txt")
    # # s = utils.read_plain_sentences("test.txt")
    s = utils.read_plain_sentences("short.txt")

    # v = ["武汉", "市长", "长江大桥"]
    # s = ["武汉市长江大桥"]

    results = mm.tokenize(v, s, mode=mm.MMMode.Forward)
    print("F", results)

    results = mm.tokenize(v, s, mode=mm.MMMode.Reverse)
    print("R", results)

    results = mm.tokenize(v, s, mode=mm.MMMode.Bidirectional)
    print("Bi", results)