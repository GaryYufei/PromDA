from datasets import load_dataset
import nltk
from rake_nltk import Rake
from tqdm import tqdm

train_out = open('C4_train_page.txt', 'w')
dev_out = open('C4_dev_page.txt', 'w')


r = Rake()
c4_data = load_dataset('c4', "realnewslike", split="train", streaming=True)
count = 0
for ins in tqdm(c4_data):
	text = ins['text']
	text = text.strip().replace('\n', ' ')
	text = text.strip().replace('\t', ' ')
	used_sentence = []
	for sent in nltk.sent_tokenize(text):
		if len(sent) > 0 and len(sent.split()) > 10:
			r.extract_keywords_from_text(sent)
			if len(r.get_ranked_phrases()) > 0:
				used_sentence.append(sent)

	if len(used_sentence) >= 5:
		if count < 10000:
			dev_out.write('\t'.join(used_sentence) + '\n')
		else:
			train_out.write('\t'.join(used_sentence) + '\n')

		if count == 10000:
			dev_out.close()

		count += 1
train_out.close()


