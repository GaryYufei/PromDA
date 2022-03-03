from pathlib import Path
import re
import random
import os
import argparse
from fast_bleu import SelfBLEU

def read_sentence_classification(file_path):
    token_docs = []
    tag_docs = []
    with open(file_path) as out:
        for l in out:
            l = l.strip()
            items = l.split('\t')
            if not len(items) == 2: continue
            token_docs.append(items[0])
            tag_docs.append(items[1])

    return token_docs, tag_docs

def read_pair_sentence_label_data(data_path):
    token_docs = []
    output_labels = []
    with open(data_path) as out:
        for l in out:
            l = l.strip()
            items = l.split('\t')
            if not len(items) == 3: continue
            token_docs.append("%s\t%s" % (items[0], items[1]))
            output_labels.append(items[2])
    return token_docs, output_labels

parser = argparse.ArgumentParser("Filtering LM-generated data")
parser.add_argument(
    "--data-path", nargs='+', required=True, help="Path to the main training files."
)
parser.add_argument(
    "--output-path", nargs='+', required=True, help="Path to the output files."
)
parser.add_argument(
    "--enable-sentence-pair",
    action='store_true',
    help="Read Sentence Pair",
)

if __name__ == "__main__":
	_A = parser.parse_args()
	token_docs, tag_docs = [], []

	for path_ in _A.data_path:
		if not _A.enable_sentence_pair:
			words, tags = read_sentence_classification(path_)
		else:
			words, tags = read_pair_sentence_label_data(path_)
		token_docs += words
		tag_docs += tags

	print("input data size %d" % len(token_docs))
	str_set = set()
	for word, tag in zip(token_docs, tag_docs):
		str_ = word + ' [wordtag] ' + tag
		str_set.add(str_)
	print("after filtering duplicated sample %d" % len(str_set))

	word2tag_dict = {}
	for str_ in str_set:
		word, tag = str_.split(' [wordtag] ')
		if word not in word2tag_dict:
			word2tag_dict[word] = []
		word2tag_dict[word].append(tag)
	word2tag_dict = {k: v for (k, v) in word2tag_dict.items() if len(v) == 1}
	print("after filtering sample with multiple tag sequence %d" % len(word2tag_dict))

	output_list_writer = [open(o_path, 'w') for o_path in _A.output_path]
	index = 0
	bio_words = []
	for word_seq in word2tag_dict:
		if len(word2tag_dict[word_seq]) == 1:
			tags = word2tag_dict[word_seq][0]
			out = output_list_writer[index % len(output_list_writer)]
			bio_words.append(word_seq.split())
			out.write("%s\t%s\n" % (word_seq, tags))
			index += 1
	for out in output_list_writer:
		out.close()

	ave_length = sum([len(v) for v in bio_words]) / len(bio_words)
	print("Generated Length %.2f" % ave_length)
	weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
	self_bleu = SelfBLEU(bio_words, weights)
	score = self_bleu.get_score()
	self_b3 = score['trigram']
	self_b3_value = 100 * sum(self_b3) / len(self_b3) 
	print("Self BLEU %.2f" % self_b3_value)
