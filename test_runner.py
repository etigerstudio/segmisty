import utils
import mm
import hmm


if __name__ == '__main__':
    unit = "hmm"

    if unit == "mm-short":
        v = utils.read_vocabulary_dataset("training_vocab.txt")
        # # s = utils.read_plain_sentences("test.txt")
        s = utils.read_plain_sentences("short.txt")

        results = mm.tokenize(v, s, mode=mm.MMMode.Forward)
        print("F", results)

        results = mm.tokenize(v, s, mode=mm.MMMode.Reverse)
        print("R", results)

        results = mm.tokenize(v, s, mode=mm.MMMode.Bidirectional)
        print("Bi", results)


    elif unit == "mm-changjiang":
        v = ["武汉", "市长", "长江大桥"]
        s = ["武汉市长江大桥"]

        results = mm.tokenize(v, s, mode=mm.MMMode.Forward)
        print("F", results)

        results = mm.tokenize(v, s, mode=mm.MMMode.Reverse)
        print("R", results)

        results = mm.tokenize(v, s, mode=mm.MMMode.Bidirectional)
        print("Bi", results)


    elif unit == "hmm":
        states = [[0, 2], [0, 2], [0, 1, 1, 2], [3]]
        observations = [[53966, 23], [5966, 43], [43, 42, 41, 40], [99]]

        model = hmm.HMM()
        model.train(states, observations)