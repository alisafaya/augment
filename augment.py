import argparse 
import nltk
import itertools
import sys

from tqdm import tqdm
import gensim.downloader as api
from nltk.corpus import wordnet as wn

import nltk; nltk.download("wordnet")

def get_words(infile):
    return open(infile).read().splitlines()

def apply_wordnet(word_list, args):

    print("Applying augmentation with wordnet, max words for each word =", args.word_net_threshold)
    augmentations = {}
    for w in tqdm(word_list, desc="Extracting similar words"):
        augmentations[w] = list(l for l in set(itertools.chain(*[[str(lemma.name()) for lemma in syn.lemmas() ] for syn in wn.synsets(w)  ])) if "_" not in l)[:args.word_net_threshold]

    return augmentations

def apply_wordvectors(word_list, args):

    print("Applying augmentation with word vectors, threshold =", args.word_vectors_threshold)
    print("Loading vector set...")
    model = api.load(args.vector_set)

    augmentations = {}
    for w in tqdm(word_list, desc="Extracting similar words"):
        if w in model.vocab:
            augmentations[w] = list(map(lambda x:x[0], model.most_similar(w, topn=args.word_vectors_threshold)))
        else:
            augmentations[w] = []

    return augmentations

def write_output(words, args):
    print("Writing output..")
    
    if args.output_file == "stdout":
        fo = sys.stdout
        print("\n".join(words), file=fo)
    else:
        fo = open(args.output_file, "w")
        print("\n".join(words), file=fo)
        fo.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="stdout")

    parser.add_argument("--word-vectors-threshold", type=int, default=10)
    parser.add_argument("--word-net-threshold", type=int, default=10)
    
    parser.add_argument("--vector-set", type=str, default="glove-twitter-25")
    parser.add_argument("--only-word-net", action="store_true")
    parser.add_argument("--only-word-vectors", action="store_true")
    parser.add_argument("--aggregate", action="store_true", help="after augmenting with wordnet, give the augmented list as input to word-vectors augmentation")

    args = parser.parse_args()

    print("-"*20)
    print("Word augmenting tool")
    print("-"*20)

    word_list = get_words(args.input_file)
    word_augmentations = { w : [] for w in word_list }

    if not args.only_word_vectors:
        augmentations = apply_wordnet(word_list, args)
        for w in augmentations:
            word_augmentations[w] = list(set(word_augmentations.get(w, []) + augmentations[w]))

    if not args.only_word_net:
        if args.aggregate:
            print("Extracting word-vectors from augmented list")
            word_list = word_augmentations.values()

        augmentations = apply_wordvectors(word_list, args)
        for w in augmentations:
            word_augmentations[w] = list(set(word_augmentations.get(w, []) + augmentations[w]))

    words = list(set( itertools.chain(*[ [w,] + word_augmentations[w] for w in word_augmentations ])))
    write_output(words, args)
    
    print("-"*100)
    print("Finished.")



