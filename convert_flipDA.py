import json

with open(input_file) as out:
	with open("datasets/boolq/fewglue_train.tsv", 'w') as writer:
		for l in out:
			item = json.loads(l)
			writer.write("%s\t%s\t%s\t\n" % (item['question'], item['passage'], item['label'].capitalize()))
