import utils
import mm
import hmm
import bgram

if __name__ == '__main__':
    unit = "bigram-export"

    if unit == "mm-short":
        v = utils.read_vocabulary_dataset("training_vocab.txt")
        s = utils.read_plain_sentences("short.txt")

        results = mm.tokenize(v, s, mode=mm.MMMode.Forward)
        print("F", results)

        results = mm.tokenize(v, s, mode=mm.MMMode.Reverse)
        print("R", results)

        results = mm.tokenize(v, s, mode=mm.MMMode.Bidirectional)
        print("Bi", results)

    elif unit == "mm-export":
        v = utils.read_vocabulary_dataset("training_vocab.txt")
        s = utils.read_plain_sentences("test.txt")

        results = mm.tokenize(v, s, mode=mm.MMMode.Forward)
        utils.export_plain_sentences(results, "fmm_result.txt")

        results = mm.tokenize(v, s, mode=mm.MMMode.Reverse)
        utils.export_plain_sentences(results, "rmm_result.txt")

        results = mm.tokenize(v, s, mode=mm.MMMode.Bidirectional)
        utils.export_plain_sentences(results, "bimm_result.txt")


    elif unit == "mm-changjiang":
        v = ["武汉", "市长", "长江大桥"]
        s = ["武汉市长江大桥"]

        results = mm.tokenize(v, s, mode=mm.MMMode.Forward)
        print("F", results)

        results = mm.tokenize(v, s, mode=mm.MMMode.Reverse)
        print("R", results)

        results = mm.tokenize(v, s, mode=mm.MMMode.Bidirectional)
        print("Bi", results)


    elif unit == "hmm-dummy":
        states = [[0, 2], [0, 2], [0, 1, 1, 2], [3]]
        observations = [[53966, 23], [5966, 43], [43, 42, 41, 40], [99]]

        model = hmm.HMM()
        model.train(states, observations)
        print(model.predict(observations))
        print(model.predict([[53966, 23, 5966, 43, 42, 41, 40]]))


    elif unit == "hmm-short":
        model = hmm.HMM()
        model.train(*utils.read_hmm_tagged_sentences("training.txt"))
        print(model.predict([utils.string_to_ints("迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）")]))
        # print(list(reversed(model.predict([utils.string_to_ints(reversed("武汉市长江大桥"))]))))


    elif unit == "hmm-train":
        model = hmm.HMM()
        model.train(*utils.read_hmm_tagged_sentences("training.txt"))
        _, observations = utils.read_hmm_tagged_sentences("training.txt")
        predicted_states = model.predict(observations)
        utils.export_hmm_tagged_sentences(predicted_states, observations, "hmm_training_result.txt")


    elif unit == "hmm-export":
        model = hmm.HMM()
        model.train(*utils.read_hmm_tagged_sentences("training.txt"))
        _, observations = utils.read_hmm_tagged_sentences("test.txt")
        predicted_states = model.predict(observations)
        utils.export_hmm_tagged_sentences(predicted_states, observations, "hmm_result.txt")


    elif unit == "bigram-test":
        test_sen1 = "在这个激动人心的时刻，我很高兴通过中国国际广播电台、中央人民广播电台和中央电视台，向全国各族人民，向香港特别行政区同胞、澳门特别行政区同胞和台湾同胞、海外侨胞，向世界各国的朋友们，致以新世纪第一个新年的祝贺！"
        test_sen2 = "女士们，先生们，同志们，朋友们："

        uni_freq, bi_freq = utils.read_bigram_words("training.txt")
        bi_gram = bgram.BiGram(uni_freq, bi_freq)
        print(bi_gram.segment_sentences([test_sen1, test_sen2]))


    elif unit == "bigram-export":
        s = utils.read_plain_sentences("test.txt")

        uni_freq, bi_freq = utils.read_bigram_words("training.txt")
        bi_gram = bgram.BiGram(uni_freq, bi_freq)
        results = bi_gram.segment_sentences(s)
        utils.export_plain_sentences(results, "bgram_result.txt")