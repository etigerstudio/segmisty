import utils
import mm
import hmm
import bigram
import time

if __name__ == '__main__':

    start_time = time.time()
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
        model.train(*utils.read_hmm_tagged_sentences("msr_training.utf8"))
        _, observations = utils.read_hmm_tagged_sentences("msr_test.utf8")
        predicted_states = model.predict(observations)
        utils.export_hmm_tagged_sentences(predicted_states, observations, "msr_hmm_result.txt")


    elif unit == "bigram-test":
        test_sen1 = "在这个激动人心的时刻，我很高兴通过中国国际广播电台、中央人民广播电台和中央电视台，向全国各族人民，向香港特别行政区同胞、澳门特别行政区同胞和台湾同胞、海外侨胞，向世界各国的朋友们，致以新世纪第一个新年的祝贺！"
        test_sen2 = "女士们，先生们，同志们，朋友们："
        test_sen3 = "（二○○○年十二月三十一日）（附图片1张）"

        uni_freq, bi_freq = utils.read_bigram_words("training.txt")
        bi_gram = bigram.BiGram(uni_freq, bi_freq)
        print(bi_gram.segment_sentences([test_sen1, test_sen2, test_sen3]))


    elif unit == "bigram-export":
        s = utils.read_plain_sentences("msr_test.utf8")

        uni_freq, bi_freq = utils.read_bigram_words("msr_training.utf8")
        bi_gram = bigram.BiGram(uni_freq, bi_freq, enable_atomic_segmentation=True)
        results = bi_gram.segment_sentences(s)
        utils.export_plain_sentences(results, "msr_bigram_result.txt")


    elif unit == "atom-segmentation-test":
        print(utils.try_atomic_segmentation("12") == 2)
        print(utils.try_atomic_segmentation("1.2") == 3)
        print(utils.try_atomic_segmentation(".2") is None)
        print(utils.try_atomic_segmentation("二○○一年") == 5)
        print(utils.try_atomic_segmentation("2003年") == 5)
        print(utils.try_atomic_segmentation("九月") == 2)
        print(utils.try_atomic_segmentation("9月") == 2)
        print(utils.try_atomic_segmentation("到2003年") is None)
        print(utils.try_atomic_segmentation("到9月") is None)
        print(utils.try_atomic_segmentation("12%") == 3)
        print(utils.try_atomic_segmentation("1.2％") == 4)
        print(utils.try_atomic_segmentation("百分之十六") == 5)
        print(utils.try_atomic_segmentation("百分之一百零三点六") == 9)
        print(utils.try_atomic_segmentation("www.in－paku.go.jp") == 17)
        print(utils.try_atomic_segmentation("caibian3＠peopledaily．com．cn") == 27)
        print(utils.try_atomic_segmentation("happynewyear.txt.vbs") == 20)
        print(utils.try_atomic_segmentation("AM21B") == 5)

    print("unit: %s - %s seconds" % (unit, time.time() - start_time))