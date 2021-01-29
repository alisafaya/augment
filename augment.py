import argparse 
import nltk
import itertools
from tqdm import tqdm
import gensim.downloader as api
from nltk.corpus import wordnet as wn

# pip install gensim
# pip install nltk

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
    print("Loading vector set")

    model = api.load('word2vec-google-news-300')

    augmentations = {}
    for w in tqdm(word_list, desc="Extracting similar words"):
        if w in model.vocab:
            augmentations[w] = list(map(lambda x:x[0], model.most_similar(w, topn=args.word_vectors_threshold)))
        else:
            augmentations[w] = []

    return augmentations

def write_output(word_dict, args):
    print("Writing output..")
    with open(args.output_file, "w") as fo:
        for w in word_dict:
            print(w + "," + ",".join(word_dict[w]), file=fo)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--word-vectors-threshold", type=int, default=10)
    parser.add_argument("--word-net-threshold", type=int, default=10)

    parser.add_argument("--input-file", type=str, default="input_sample.txt")
    parser.add_argument("--output-file", type=str, default="augment_output.txt")
    
    parser.add_argument("--only-word-net", action="store_true")
    parser.add_argument("--only-word-vectors", action="store_true")
    parser.add_argument("--aggregate", action="store_true", help="after augmenting with wordnet, give the augmented list as input to word-vectors augmentation")

    args = parser.parse_args()

    word_list = get_words(args.input_file)
    word_augmentations = { w : [] for w in word_list }

    if not args.only_word_vectors:
        augmentations = apply_wordnet(word_list, args)
        for w in augmentations:
            word_augmentations[w] = list(set(word_augmentations[w] + augmentations[w]))

    if not args.only_word_net:
        if args.aggregate:
            print("Extracting word-vectors from augmented list")
            augmentations = apply_wordvectors(word_list, args)
        else:
            augmentations = apply_wordvectors(word_list, args)

        for w in augmentations:
            word_augmentations[w] = list(set(word_augmentations[w] + augmentations[w]))

    write_output(word_augmentations, args)
    print("Finished.")



