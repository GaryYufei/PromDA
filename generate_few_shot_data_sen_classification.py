import re
import random
import argparse
import os

def read_sentence_classification(file_path):
    token_docs = []
    tag_docs = []
    with open(file_path) as out:
        for l in out:
            l = l.strip()
            items = l.split('\t')
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

def extract_data(_A, data_path, split_name, save_unlabel_data=True):
    if not _A.enable_sentence_pair:
        token_docs, tag_docs = read_sentence_classification(data_path)
    else:
        token_docs, tag_docs = read_pair_sentence_label_data(data_path)

    random.seed(_A.random_seed)
    os.makedirs(_A.output_path, exist_ok=True)

    tag_dict = {}
    all_data = []
    for tokens, tag in zip(token_docs, tag_docs):
        all_data.append((tokens, tag))
        if tag not in tag_dict:
            tag_dict[tag] = []
        tag_dict[tag].append((tokens, tag))

    print("found %d tag slots" % len(tag_dict))
    print("found %d instances" % len(all_data))

    
    selected_sen = []
    selected_tag = []
    used_sen = set()
    if _A.few_shot_k > 0:
        for chunk in tag_dict:
            random.shuffle(tag_dict[chunk])
            count = 0
            for (sen, tags) in tag_dict[chunk]:
                if (sen not in used_sen) and len(sen.split()) > _A.min_length:
                    used_sen.add(sen)
                    selected_sen.append(sen)
                    selected_tag.append(tags)
                    count += 1
                if count == _A.few_shot_k: break
    elif _A.total_training_num > 0:
        random.shuffle(all_data)
        for (sen, tags) in all_data:
            if (sen not in used_sen) and len(sen.split()) > _A.min_length:
                used_sen.add(sen)
                selected_sen.append(sen)
                selected_tag.append(tags)

            if len(selected_sen) == _A.total_training_num: break
    else:
        raise ValueError("few_shot_k and total_training_num cannot be non-positive!")


    print("select %d labeled instances" % len(selected_tag))
    num_ = _A.few_shot_k if _A.few_shot_k > 0 else _A.total_training_num

    output_path = os.path.join(_A.output_path, '%s_whole_%d.txt' % (split_name, num_))
    with open(output_path, 'w') as out:
        for (gen, labels) in zip(selected_sen, selected_tag):
            out.write("%s\t%s\n" % (gen, labels))

    if save_unlabel_data:
        unlabeled_output_path = os.path.join(_A.output_path, 'unlabeled_train_whole_%d.txt' % num_)
        unlabeled_count = 0
        with open(unlabeled_output_path, 'w') as out:
            for (sen, labels) in zip(token_docs, tag_docs):
                if sen not in used_sen:
                    unlabeled_count += 1
                    out.write("%s\t%s\n" % (sen, labels))
        print("select %d unlabeled instances" % unlabeled_count)

parser = argparse.ArgumentParser("Generate Few-shot Data")
parser.add_argument(
    "--data-path", required=True, help="Path to the main training files."
)
parser.add_argument(
    "--output-path", required=True, help="Path to the output files."
)
parser.add_argument(
    "--few-shot-k", type=int, default=-1, help="The number of instances for each labels"
)
parser.add_argument(
    "--total-training-num", type=int, default=-1, help="Total number of training data"
)
parser.add_argument(
    "--min-length", type=int, default=0, help="Minimal Length"
)
parser.add_argument(
    "--random-seed", type=int, default=2, help="Random Seed"
)
parser.add_argument(
    "--enable-sentence-pair",
    action='store_true',
    help="Read Sentence Pair",
)

if __name__ == "__main__":
    _A = parser.parse_args()
    train_path = os.path.join(_A.data_path, 'train_whole.txt')
    valid_path = os.path.join(_A.data_path, 'valid_whole.txt')

    print("training data")
    extract_data(_A, train_path, 'train')

    print("valid data")
    extract_data(_A, valid_path, 'valid', save_unlabel_data=False)
    