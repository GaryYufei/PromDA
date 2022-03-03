from pathlib import Path
import re
import random
import argparse
import os

def get_chunk_type(tag_name):
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq):
    default = "O"
    chunks = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

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
            token, tag = line.split()
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

def extract_data(_A, data_path, split_name, save_unlabel_data=True):
    token_docs, tag_docs = read_conll(data_path)

    random.seed(_A.random_seed)
    os.makedirs(_A.output_path, exist_ok=True)

    tag_dict = {}
    data_count = 0
    all_data = []
    for tokens, tags in zip(token_docs, tag_docs):
        all_data.append((tokens, tags))
        data_count += 1
        chunks = get_chunks(tags)
        for chunk in chunks:
            if chunk[0] not in tag_dict:
                tag_dict[chunk[0]] = []
            tag_dict[chunk[0]].append((tokens, tags))

    print("found %d tag slots" % len(tag_dict))
    print("found %d instances" % data_count)

    selected_sen = []
    selected_tag = []
    used_sen = set()
    if _A.few_shot_k > 0:
        for chunk in tag_dict:
            random.shuffle(tag_dict[chunk])
            count = 0
            for (tokens, tags) in tag_dict[chunk]:
                sen = ' '.join(tokens)
                if (sen not in used_sen) and len(tokens) > _A.min_length and len(tokens) < _A.max_length:
                    used_sen.add(sen)
                    selected_sen.append(tokens)
                    selected_tag.append(tags)
                    count += 1
                if count == _A.few_shot_k: break
    elif _A.total_training_num > 0:
        random.shuffle(all_data)
        for (tokens, tags) in all_data:
            sen = ' '.join(tokens)
            if (sen not in used_sen) and len(tokens) > _A.min_length and len(tokens) < _A.max_length:
                used_sen.add(sen)
                selected_sen.append(tokens)
                selected_tag.append(tags)

            if len(selected_sen) == _A.total_training_num: break
    else:
        raise ValueError("few_shot_k and total_training_num cannot be non-positive!")


    print("select %d labeled instances" % len(selected_tag))
    num_ = _A.few_shot_k if _A.few_shot_k > 0 else _A.total_training_num

    output_path = os.path.join(_A.output_path, '%s_whole_%d.txt' % (split_name, num_))
    with open(output_path, 'w') as out:
        for (gen, labels) in zip(selected_sen, selected_tag):
            for g, l in zip(gen, labels):
                out.write("%s %s\n" % (g, l))
            out.write("\n")

    if save_unlabel_data:
        unlabeled_output_path = os.path.join(_A.output_path, 'unlabeled_train_whole_%d.txt' % num_)
        unlabeled_count = 0
        with open(unlabeled_output_path, 'w') as out:
            for (gen, labels) in zip(token_docs, tag_docs):
                sen = ' '.join(gen)
                if sen not in used_sen and len(gen) > _A.min_length and len(gen) < _A.max_length:
                    unlabeled_count += 1
                    for g, l in zip(gen, labels):
                        out.write("%s %s\n" % (g, l))
                    out.write("\n")
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
    "--max-length", type=int, default=10000, help="Minimal Length"
)
parser.add_argument(
    "--random-seed", type=int, default=2, help="Random Seed"
)

if __name__ == "__main__":
    _A = parser.parse_args()
    train_path = os.path.join(_A.data_path, 'train_whole.txt')
    valid_path = os.path.join(_A.data_path, 'valid_whole.txt')

    print("training data")
    extract_data(_A, train_path, 'train')

    print("valid data")
    extract_data(_A, valid_path, 'valid', save_unlabel_data=False)