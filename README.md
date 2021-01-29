
## Augment

- Take a text file that contains a list of words (one word per line), and augment this list using wordnet and word-vectors. 
- If `--output-file` is provided, write that list into the file

__Example usage__

`python augment.py --input-file input_sample.txt --output-file output_sample.txt`

__Options__

```bash
usage: augment.py [-h] [--word-vectors-threshold WORD_VECTORS_THRESHOLD] [--word-net-threshold WORD_NET_THRESHOLD] [--input-file INPUT_FILE] [--output-file OUTPUT_FILE]
                  [--vector-set VECTOR_SET] [--only-word-net] [--only-word-vectors] [--aggregate]

optional arguments:
  -h, --help            show this help message and exit
  --word-vectors-threshold WORD_VECTORS_THRESHOLD
  --word-net-threshold WORD_NET_THRESHOLD
  --input-file INPUT_FILE
  --output-file OUTPUT_FILE
  --vector-set VECTOR_SET
  --only-word-net
  --only-word-vectors
  --aggregate           after augmenting with wordnet, give the augmented list as input to word-vectors augmentation

```