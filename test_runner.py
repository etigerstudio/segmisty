import utils
import mm
import hmm
import bigram
import time
import perceptron

if __name__ == '__main__':

    start_time = time.time()
    unit = "perceptron-pku"

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
        model.train(*utils.read_sequential_tagged_sentences("training.txt", convert_character_to_int=True))
        print(model.predict([utils.string_to_ints("迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）")]))
        # print(list(reversed(model.predict([utils.string_to_ints(reversed("武汉市长江大桥"))]))))


    elif unit == "hmm-train":
        model = hmm.HMM()
        model.train(*utils.read_sequential_tagged_sentences("training.txt", convert_character_to_int=True))
        _, observations = utils.read_sequential_tagged_sentences("training.txt", convert_character_to_int=True)
        predicted_states = model.predict(observations)
        utils.export_sequential_tagged_sentences(predicted_states, observations, "hmm_training_result.txt", convert_int_to_character=True)


    elif unit == "hmm-export":
        model = hmm.HMM()
        model.train(*utils.read_sequential_tagged_sentences("msr_training.utf8", convert_character_to_int=True))
        _, observations = utils.read_sequential_tagged_sentences("msr_test.utf8", convert_character_to_int=True)
        predicted_states = model.predict(observations)
        utils.export_sequential_tagged_sentences(predicted_states, observations, "msr_hmm_result.txt", convert_int_to_character=True)


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

    elif unit == "perceptron-test":
        p = perceptron.Perceptron("dummy")
        p.train(["我想吃饭", "运动真好"], [[3, 3, 0, 2], [0, 2, 3, 3]])
        print(p.predict("我想吃饭")[0])
        print(p.predict("运动真好")[0])

        p = perceptron.Perceptron("small_msra")
        tags, sentences = utils.read_sequential_tagged_sentences("small_msra_test.txt")
        p.train(sentences, tags)
        print(p.predict("“吃屎的东西，连一捆麦也铡不动呀？")[0])

    elif unit == "perceptron-load":
        p = perceptron.Perceptron.load("mini_test.perceptron")
        print(p.predict("“吃屎的东西，连一捆麦也铡不动呀？")[0])

    elif unit == "perceptron-export-load":
        p = perceptron.Perceptron("dummy-export")
        p.train(["我想吃饭", "运动真好"], [[3, 3, 0, 2], [0, 2, 3, 3]])
        print(p.predict("我想吃饭")[0])
        print(p.predict("运动真好")[0])
        p.export("dummy-export.perceptron")

        p = None
        p = perceptron.Perceptron.load("dummy-export.perceptron")
        print(p.predict("我想吃饭")[0])
        print(p.predict("运动真好")[0])

    elif unit == "perceptron-pku":
        p = perceptron.Perceptron()
        tags, sentences = utils.read_sequential_tagged_sentences("training.txt")
        e_tags, e_sentences = utils.read_sequential_tagged_sentences("test.txt")
        p.train(sentences, tags, evaluate_sentences=e_sentences, evaluate_tag_set=e_tags)
        print(p.predict("共同创造美好的新世纪——二○○一年新年贺词")[0])

    elif unit == "perceptron-evaluate":
        p = perceptron.Perceptron.load("pku-train-f1-0.9923-250.perceptron")
        _, _, f1, _, formatted_string = p.evaluate("training.txt", export_results=True, export_filename="pku-train-f1-0.9923-250.result.txt")
        print(f"pku-train: {formatted_string}")

    elif unit == "perceptron-profile":
        p = perceptron.Perceptron.load("pku-f1-0.8864-14.perceptron")
        tags, sentences = utils.read_sequential_tagged_sentences("training.txt")
        p.train(sentences, tags, max_iter=1)

    elif unit == "crfpp-train-generate":
        # utils.generate_crfpp_compatible_file("training.txt", "training_crfpp.txt")
        utils.generate_crfpp_compatible_file("pku_test.utf8", "testing_crfpp.txt", with_tags=False)

    elif unit == "crfpp-result-transform":
        utils.transform_crfpp_results_to_sentences("crfpp_crf_test_result.txt", "crfpp_result.txt")

    elif unit == "evaluate_prediction":
        print(utils.evaluate_truth_and_predict("test.txt", "crfpp_result.txt")[4])



    print("unit: %s - %s seconds" % (unit, time.time() - start_time))