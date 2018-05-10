# encoding: utf-8

# author: liaochangzeng
# github: https://github.com/changzeng

import pickle
import thulac
import argparse
from random import randint

def preprocess():
    min_len = 10

    with open("data/words.txt", encoding="utf-8") as fd:
        txt = fd.read().strip()
        word_set = set(txt.split("\n"))

    # 获得要滤除的行号
    remove_list = []
    with open("data/raw.en", encoding="utf-8") as fd:
        index = 0
        for line in fd:
            line = line.strip()
            word_list = line.split(" ")
            valid_list = [word for word in word_list if word in word_set]
            rate = len(valid_list) * 1.0 / len(word_list)
            if rate < 0.5 or len(word_list)<min_len:
                remove_list.append(index)
            index += 1

    with open("data/raw.en", encoding="utf-8") as fd:
        en_txt = fd.read().strip()
        en_txt = en_txt.split("\n")
        en_txt = [en_txt[i] for i in range(len(en_txt)) if i not in remove_list]
        en_txt = [sen for sen in en_txt if len(sen) >= min_len]
    thu = thulac.thulac(seg_only=True)
    with open("data/raw.zh", encoding="utf-8") as fd:
        zh_txt = fd.read().strip()
        zh_txt = zh_txt.split("\n")
        zh_txt = [zh_txt[i] for i in range(len(zh_txt)) if i not in remove_list]
        zh_txt = [" ".join([item[0] for item in thu.cut(sen)]) for sen in zh_txt]

    with open("data/valid.en", "w", encoding="utf-8") as fd:
        fd.write("\n".join(en_txt))
    with open("data/valid.zh", "w", encoding="utf-8") as fd:
        fd.write("\n".join(zh_txt))


def pos_samples():
    with open("data/valid.en", encoding="utf-8") as fd:
        en_txt = fd.read().strip().split("\n")
    with open("data/valid.zh", encoding="utf-8") as fd:
        zh_txt = fd.read().strip().split("\n")
    return zip(en_txt, zh_txt, ['1']*len(en_txt))


def neg_samples(num=1):
    with open("data/valid.en", encoding="utf-8") as fd:
        en_txt = fd.read().strip().split("\n")
    with open("data/valid.zh", encoding="utf-8") as fd:
        zh_txt = fd.read().strip().split("\n")

    gen_en_txt = []
    gen_zh_txt = []
    for i in range(len(en_txt)):
        for _ in range(num):
            z = randint(0, len(en_txt)-1)
            while i == z:
                z = randint(0, len(en_txt)-1)
            gen_en_txt.append(en_txt[i])
            gen_zh_txt.append(zh_txt[z])
    return zip(gen_en_txt, gen_zh_txt, ['0']*len(en_txt)*num)

def gen_samples(neg_sample_num):
    pos = list(pos_samples())
    neg = list(neg_samples(neg_sample_num))
    with open("data/train.data", "w", encoding="utf-8") as fd:
        samples = pos + neg
        samples = ["\t".join(item) for item in samples]
        fd.write("\n".join(samples))

# 生成字典文件
def gen_dictionary():
    from batch_loader import Vocabulary
    with open("data/train.data", encoding="utf-8") as fd:
        txt = fd.read().strip().split("\n")
        txt = [item.split("\t") for item in txt]
        en = [item[0] for item in txt]
        zh = [item[1] for item in txt]
    vocab_en = Vocabulary(100, 5)
    vocab_zh = Vocabulary(100, 5)
    vocab_en.fit_transform(en)
    vocab_zh.fit_transform(zh)
    vocab_en.save("data/vocab_en.data")
    vocab_zh.save("data/vocab_zh.data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocess")
    parser.add_argument("--mode", type=str, default="preprocess", help="running mode")
    parser.add_argument("--neg_sample_num", type=int, default=1, help="Neg-Sample number")
    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess()
    elif args.mode == "gen_samples":
        gen_samples(args.neg_sample_num)
    elif args.mode == "gen_dic":
        gen_dictionary()
    else:
        print("Error! No such mode!")
