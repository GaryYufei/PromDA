from pathlib import Path
import re
import random
import argparse
import os
import argparse
from fast_bleu import SelfBLEU

def read_conll(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            items = line.split()
            if len(items) == 2:
                token, tag = items
                tokens.append(token)
                tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

parser = argparse.ArgumentParser("Filtering LM-generated data")
parser.add_argument(
    "--data-path", nargs='+', required=True, help="Path to the main training files."
)
parser.add_argument(
    "--output-path", nargs='+', required=True, help="Path to the output files."
)

if __name__ == "__main__":
	_A = parser.parse_args()
	token_docs, tag_docs = [], []

	for path_ in _A.data_path:
		words, tags = read_conll(path_)
		token_docs += words
		tag_docs += tags

	print("input data size %d" % len(token_docs))
	str_set = set()
	for word, tag in zip(token_docs, tag_docs):
		str_ = ' '.join(word) + '\t' + ' '.join(tag)
		str_set.add(str_)
	print("after filtering duplicated sample %d" % len(str_set))

	word2tag_dict = {}
	for str_ in str_set:
		word, tag = str_.split('\t')
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
			words = word_seq.split()
			tags = word2tag_dict[word_seq][0].split()
			if len(words) == len(tags):
				out = output_list_writer[index % len(output_list_writer)]
				bio_words.append(words)
				for g, l in zip(words, tags):
					out.write("%s %s\n" % (g, l))
				out.write("\n")
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
