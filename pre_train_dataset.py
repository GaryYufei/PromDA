from rake_nltk import Rake
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from pathlib import Path
import re
import torch
import numpy as np
import random
import copy
from tqdm import tqdm
from nltk.corpus import wordnet as wn

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
            items = line.split()
            if len(items) == 2:
                token, tag = items
                tokens.append(token)
                tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

def sorting(lst):
    lst2 = sorted(lst, key=len)
    return lst2

def clean_top_features(keywords, top=5):
    keywords = sorting(keywords)
    newkeys = []
    newkeys.append(keywords[len(keywords)-1])
    for i in range(len(keywords)-2,-1,-1):
        if newkeys[len(newkeys)-1].startswith(keywords[i]):
            continue
        newkeys.append(keywords[i])

    if len(newkeys) > top:
        return newkeys[:top]
    return newkeys

def process_tensor(tensor_list, last_dim, output_mask=False):
    tensor_len = [d.shape[0] for d in tensor_list]
    tensor_max_lenth = max(tensor_len)
    d_type = tensor_list[0].dtype
    if last_dim > 0:
        tensor_np = np.zeros((len(tensor_list), tensor_max_lenth, last_dim), dtype=d_type)
    else:
        tensor_np = np.zeros((len(tensor_list), tensor_max_lenth), dtype=d_type)
    mask_np = np.zeros((len(tensor_list), tensor_max_lenth), dtype=np.float32)
    for i, (d, l) in enumerate(zip(tensor_list, tensor_len)):
        if l > 0:
            tensor_np[i, :l] = d
            mask_np[i, :l] = 1
    if output_mask:
        return torch.from_numpy(tensor_np), torch.from_numpy(mask_np)
    else:
        return torch.from_numpy(tensor_np)

def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100 #tag2id['O']
        arr_offset = np.array(doc_offset)

        preversed_doc_enc_labels = doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)]
        if preversed_doc_enc_labels.shape[0] == len(doc_labels):
            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

class SeqLabelConsistencyDataset(torch.utils.data.Dataset):

    def __init__(self, token_docs, tag_docs, label_dict, tokenizer):
        self.label_dict = label_dict 
        self.tokenizer = tokenizer

        self.words = token_docs
        self.encodings = self.tokenizer(token_docs, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        self.encoded_labels = encode_tags(tag_docs, self.encodings, self.label_dict)

        self.encodings.pop('offset_mapping')
        self.encodings.pop('token_type_ids')

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.encoded_labels[idx])
        item['labels'][item['attention_mask'] == 0] = -100
        item['gt_x'] = self.words[idx]
        return item

    def __len__(self):
        return len(self.encoded_labels)

class PairSenLabelDataset(torch.utils.data.Dataset):

    def __init__(self, config, data_path, label_dict, tokenizer, is_training=False):
        self.label_dict = label_dict 
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.config = config

        token_docs, output_labels = self.read_pair_sentence_label_data(data_path, enable_oversample=True)
        if is_training:
            for d_path in config.lm_gen_train_path_list:
                sub_token_docs, sub_output_labels = self.read_pair_sentence_label_data(d_path)
                token_docs += sub_token_docs
                output_labels += sub_output_labels

        print("Total instances %d" % len(token_docs))
        self.words = [s.replace(' [SEP] ', '\t') for s in token_docs]
        self.encoded_labels = output_labels
        self.encodings = self.tokenizer(token_docs, padding=True, truncation=True)

    def read_pair_sentence_label_data(self, data_path, enable_oversample=False):
        token_docs = []
        output_labels = []
        with open(data_path) as out:
            for l in out:
                l = l.strip()
                items = l.split('\t')
                if not len(items) == 3: continue
                if items[2] not in self.label_dict: continue
                for _ in range(self.config.oversample if (self.is_training or self.config.enable_eval_oversample) and enable_oversample else 1):
                    token_docs.append("%s [SEP] %s" % (items[0], items[1]))
                    output_labels.append(self.label_dict[items[2]])
        return token_docs, output_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.encoded_labels[idx]])
        item['gt_x'] = self.words[idx]
        return item

    def __len__(self):
        return len(self.encoded_labels)


class SenLabelDataset(torch.utils.data.Dataset):

    def __init__(self, config, data_path, label_dict, tokenizer, is_training=False):
        self.label_dict = label_dict 
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.config = config

        token_docs, output_labels = self.read_sentence_label_data(data_path, enable_oversample=True)
        if is_training:
            for d_path in config.lm_gen_train_path_list:
                sub_token_docs, sub_output_labels = self.read_sentence_label_data(d_path)
                token_docs += sub_token_docs
                output_labels += sub_output_labels
                    
        print("Total instances %d" % len(token_docs))
        self.words = token_docs
        self.encoded_labels = output_labels
        self.encodings = self.tokenizer(token_docs, padding=True, truncation=True)

    def read_sentence_label_data(self, data_path, enable_oversample=False):
        token_docs = []
        output_labels = []
        with open(data_path) as out:
            for l in out:
                l = l.strip()
                items = l.split('\t')
                if not len(items) == 2: continue
                if items[1] not in self.label_dict: continue
                for _ in range(self.config.oversample if (self.is_training or self.config.enable_eval_oversample) and enable_oversample else 1):
                    token_docs.append(items[0])
                    output_labels.append(self.label_dict[items[1]])
        return token_docs, output_labels


    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.encoded_labels[idx]])
        item['gt_x'] = self.words[idx]
        return item

    def __len__(self):
        return len(self.encoded_labels)

def sen_data_wrapper(config, dataset):
    input_ids = torch.cat([d['input_ids'].unsqueeze(0) for d in dataset], dim=0)
    attention_mask = torch.cat([d['attention_mask'].unsqueeze(0) for d in dataset], dim=0)
    labels = torch.cat([d['labels'].unsqueeze(0) for d in dataset], dim=0).long()
    gt_x = [d['gt_x'] for d in dataset]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "gt_x": gt_x}

def get_sen_data_loader(config, dataset, batch_size, shuffle=False):
    collate_fn = lambda d: sen_data_wrapper(config, d)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=shuffle
    )



class SeqLabelDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path, label_dict, tokenizer, is_training=False):
        self.label_dict = label_dict 
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.config = config

        token_docs = []
        tag_docs = []

        tokens, tags = read_conll(data_path)
        for token, tag in zip(tokens, tags):
            if all([t in self.label_dict for t in tag]) and (len(token) < 200 or not self.is_training):
                for _ in range(config.oversample if (self.is_training or self.config.enable_eval_oversample) else 1):
                    token_docs.append(token)
                    tag_docs.append(tag)

        if is_training:
            for d_path in config.lm_gen_train_path_list:
                tokens, tags = read_conll(d_path)
                for token, tag in zip(tokens, tags):
                    if all([t in self.label_dict for t in tag]) and len(token) < 200:
                        token_docs.append(token)
                        tag_docs.append(tag)

        print("Total instances %d" % len(tag_docs))

        self.words = token_docs
        self.encodings = self.tokenizer(token_docs, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        self.encoded_labels = encode_tags(tag_docs, self.encodings, self.label_dict)

        self.encodings.pop('offset_mapping')
        self.encodings.pop('token_type_ids')

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.encoded_labels[idx])
        item['labels'][item['attention_mask'] == 0] = -100
        item['gt_x'] = self.words[idx]
        return item

    def __len__(self):
        return len(self.encoded_labels)

def seq_data_wrapper(config, dataset):
    input_ids = torch.cat([d['input_ids'].unsqueeze(0) for d in dataset], dim=0)
    attention_mask = torch.cat([d['attention_mask'].unsqueeze(0) for d in dataset], dim=0)
    labels = torch.cat([d['labels'].unsqueeze(0) for d in dataset], dim=0)
    gt_x = [d['gt_x'] for d in dataset]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "gt_x": gt_x}

def get_seq_data_loader(config, dataset, batch_size, shuffle=False):
    collate_fn = lambda d: seq_data_wrapper(config, d)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=shuffle
    )

class PreTrainDataset(Dataset):

    def __init__(self, config, tokenizer, data_path):
        self.config = config
        self.tokenizer = tokenizer
        self.pre_training_modes = config.pre_training_modes
        self.mode_func = {
            "webpage_syn_keyword": self.webpage_syn_keyword,
        }
        self.r = Rake()
        self.record = []
        with open(data_path) as out:
            for l in tqdm(out):
                l = l.strip()
                if len(l) > 0:
                    self.record.append(l)

    def webpage_syn_keyword(self, text):
        sen_list = text.split('\t')
        sen_count = random.randint(1, 3)
        start_index = random.randint(0, len(sen_list) - sen_count)
        selected_sens = sen_list[start_index: start_index + sen_count]
        self.r.extract_keywords_from_sentences(selected_sens)
        keyword_count = random.randint(1, 5)
        keywords = clean_top_features(self.r.get_ranked_phrases(), top=keyword_count)
        syn_keyword = []
        for keyword in keywords:
            if random.random() < 0.5:
                syn_keyword.append(keyword)
            else:
                nw_list = []
                for w in keyword.split():
                    w = w.strip()
                    w_syn = [ws.lemma_names()[0] for ws in wn.synsets(w)]
                    if len(w_syn) > 0:
                        w = random.choice(w_syn)
                    nw_list.append(w)
                syn_keyword.append(' '.join(nw_list))
        s_doc_token = self.tokenizer(' and '.join(syn_keyword), return_tensors="np")['input_ids'][0, :self.config.max_length]
        t_doc_token = self.tokenizer(' '.join(selected_sens), return_tensors="np")['input_ids'][0, :self.config.max_length]
        return s_doc_token, t_doc_token

    def __len__(self):
        return len(self.record)

    def __getitem__(self, index):
        method = random.choice(self.pre_training_modes)
        return self.mode_func[method](self.record[index])

def data_wrapper(config, dataset):
    encoder_input_ids, encoder_mask = process_tensor([d[0] for d in dataset], 0, output_mask=True)
    decoder_input_ids, decoder_mask = process_tensor([d[1] for d in dataset], 0, output_mask=True)
    decoder_input_ids[decoder_mask == 0] = -100

    return {"encoder_input_ids": encoder_input_ids, "encoder_mask": encoder_mask, "decoder_input_ids": decoder_input_ids}

def get_data_loader(config, dataset, batch_size, shuffle=False):
    collate_fn = lambda d: data_wrapper(config, d)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=shuffle
    )

class NLGMixSenClsDataset(Dataset):

    SKIP_ATTRIBUTES = ['gt_x', 'gt_y']

    def __init__(self, config, data_path, tokenizer, label_index, is_training=False):
        self.tokenizer = tokenizer
        self.label_index = label_index
        self.is_training = is_training
        self.config = config
        self.r = Rake()
        self.data_list = []

        if is_training:
            assert len(self.config.training_da_mode) > 0
            self.da_mode = self.config.training_da_mode
        else:
            assert len(self.config.eval_da_mode) > 0
            self.da_mode = self.config.eval_da_mode

        self.mode_func = {
            "keyword": self.gen_from_keyword_sequence,
            "tag": self.gen_from_tag_sequence,
        }

        self.task_index = {
            "keyword": 0,
            "tag": 1 if config.prefix_set_number == 2 else 0
        }  

        self.data_list = []
        vocab2doc = {}

        if config.enable_sentence_classification:
            with open(data_path) as out:
                doc_id = 0
                for l in out:
                    l = l.strip()
                    items = l.split('\t')
                    if not len(items) == 2: continue
                    for _ in range(config.eval_data_replication if not is_training else 1):
                        self.data_list.append((items[0], items[1]))
                    for w in set(items[0].split()):
                        w = w.lower()
                        if w not in vocab2doc:
                            vocab2doc[w] = []
                        vocab2doc[w].append(doc_id)
                    doc_id += 1
        elif config.enable_pair_sentence_classification:
            with open(data_path) as out:
                doc_id = 0
                for l in out:
                    l = l.strip()
                    items = l.split('\t')
                    if not len(items) == 3: continue
                    sentence = "%s [SEP] %s" % (items[0], items[1])
                    for _ in range(config.eval_data_replication if not is_training else 1):
                        self.data_list.append((sentence, items[2]))
                    combined_sen = "%s %s" % (items[0], items[1])
                    for w in set(combined_sen.split()):
                        w = w.lower()
                        if w not in vocab2doc:
                            vocab2doc[w] = []
                        vocab2doc[w].append(doc_id)
                    doc_id += 1
                
        self.idf_value = {}
        for w in vocab2doc:
            self.idf_value[w] = doc_id / len(vocab2doc[w])

        print("Data Size %d" % len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        (word, tag) = self.data_list[idx]
        current_mode = random.choice(self.da_mode)
        task_index = self.task_index[current_mode]
        x_np, y_np, input_y, input_x = self.mode_func[current_mode](word, tag)
        return x_np, y_np, input_y, input_x, task_index

    def gen_from_tag_sequence(self, token, tag):
        input_y = tag + " " + token
        input_x = tag
        y_np = self.tokenizer(input_y, return_tensors="np")['input_ids'][0, :self.config.max_length]
        x_np = self.tokenizer(input_x, return_tensors="np")['input_ids'][0, :self.config.max_length]
        return x_np, y_np, input_y, input_x

    def gen_from_keyword_sequence(self, token, tag):
        input_y = tag + " " + token
        self.r.extract_keywords_from_text(token)
        raw_keyword_list = self.r.get_ranked_phrases()
        if len(raw_keyword_list) > 0:
            mention_list = clean_top_features(raw_keyword_list, top=5)
        else:
            current_tokens = token.split()
            w_idf = [(w, self.idf_value[w.lower()]) for w in current_tokens if w.lower() in self.idf_value]
            w_idf = sorted(w_idf, key=lambda x: x[1], reverse=True)
            mention_list = [k[0] for k in w_idf[:6]]
        random.shuffle(mention_list)

        input_x = " and ".join(mention_list)  # "generate with keywords: " + 
        y_np = self.tokenizer(input_y, return_tensors="np")['input_ids'][0, :self.config.max_length]
        x_np = self.tokenizer(input_x, return_tensors="np")['input_ids'][0, :self.config.max_length]
        return x_np, y_np, input_y, input_x


class NLGMixDataset(Dataset):

    SKIP_ATTRIBUTES = ['gt_x', 'gt_y']

    def __init__(self, config, data_path, tokenizer, label_index, is_training=False):
        self.tokenizer = tokenizer
        self.label_index = label_index
        self.is_training = is_training
        self.config = config
        self.r = Rake()

        self.data_list = []

        if is_training:
            assert len(self.config.training_da_mode) > 0
            self.da_mode = self.config.training_da_mode
        else:
            assert len(self.config.eval_da_mode) > 0
            self.da_mode = self.config.eval_da_mode

        self.mode_func = {
            "keyword": self.gen_from_keyword_sequence,
            "tag": self.gen_from_tag_sequence,
            "keyword_tag_mixture": self.gen_from_tag_keyword_mix_sequence
        }

        self.task_index = {
            "keyword": 0,
            "tag": 1 if config.prefix_set_number == 2 else 0
        }  


        tokens, tags = read_conll(data_path)
        assert len(tokens) == len(tags)

        doc_count = len(tokens)
        vocab2doc = {}
        for doc_id, words in enumerate(tokens):
            for w in set(words):
                w = w.lower()
                if w not in vocab2doc:
                    vocab2doc[w] = []
                vocab2doc[w].append(doc_id)
        self.idf_value = {}
        for w in vocab2doc:
            self.idf_value[w] = doc_count / len(vocab2doc[w])

        for (word, tag) in zip(tokens, tags):
            for _ in range(config.eval_data_replication if not is_training else 1):
                self.data_list.append((word, tag))

        print("Data Size %d" % len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        (word, tag) = self.data_list[idx]
        current_mode = random.choice(self.da_mode)
        task_index = self.task_index[current_mode]
        x_np, y_np, input_y, input_x = self.mode_func[current_mode](word, tag)
        return x_np, y_np, input_y, input_x, task_index

    def add_annotation(self, tokens, chunk_info):
        entity_label = "B-%s" % chunk_info[0]
        if chunk_info[1] + 1 == chunk_info[2]:
            tokens[chunk_info[1]] = "%s %s &&" % (entity_label, tokens[chunk_info[1]])
        else:
            tokens[chunk_info[1]] = "%s %s" % (entity_label, tokens[chunk_info[1]])
            tokens[chunk_info[2] - 1] = "%s &&" % (tokens[chunk_info[2] - 1])
        return tokens

    def gen_from_tag_sequence(self, token, tag):
        label_list = []
        chunks = get_chunks(tag)
        copied_token = copy.deepcopy(token)
        for v in chunks:
            entity_label = "B-%s" % v[0]
            copied_token = self.add_annotation(copied_token, v)
            label_list.append(entity_label)
        input_y = ' '.join(copied_token)
        random.shuffle(label_list)
        input_x = " and ".join(label_list)  # "generate with tags: " + 
        y_np = self.tokenizer(input_y, return_tensors="np")['input_ids'][0, :self.config.max_length]
        x_np = self.tokenizer(input_x, return_tensors="np")['input_ids'][0, :self.config.max_length]

        return x_np, y_np, input_y, input_x

    def gen_from_keyword_sequence(self, token, tag):
        mention_list = []
        chunks = get_chunks(tag)
        copied_token = copy.deepcopy(token)
        for v in chunks:
            entity_mention = ' '.join(token[v[1]: v[2]])
            copied_token = self.add_annotation(copied_token, v)
            mention_list += entity_mention.split()
        input_y = ' '.join(copied_token)
        current_tokens = input_y.split()
        w_idf = [(w, self.idf_value[w.lower()]) for w in current_tokens if w.lower() in self.idf_value]
        w_idf = sorted(w_idf, key=lambda x: x[1], reverse=True)
        mention_list += [k[0] for k in w_idf[:3]]
        mention_list = list(set(mention_list))

        random.shuffle(mention_list)
        mention_list = mention_list[:3]

        input_x = " and ".join(mention_list)  # "generate with keywords: " + 
        y_np = self.tokenizer(input_y, return_tensors="np")['input_ids'][0, :self.config.max_length]
        x_np = self.tokenizer(input_x, return_tensors="np")['input_ids'][0, :self.config.max_length]

        return x_np, y_np, input_y, input_x

    def gen_from_tag_keyword_mix_sequence(self, token, tag):
        mention_list = []
        chunks = get_chunks(tag)
        copied_token = copy.deepcopy(token)
        for v in chunks:
            entity_mention = ' '.join(token[v[1]: v[2]])
            entity_label = "B-%s" % v[0]
            copied_token = self.add_annotation(copied_token, v)
            mention_list += entity_mention.split()
            mention_list.append(entity_label)
        input_y = ' '.join(copied_token)
        current_tokens = copied_token
        w_idf = [(w, self.idf_value[w.lower()]) for w in current_tokens if w.lower() in self.idf_value]
        w_idf = sorted(w_idf, key=lambda x: x[1], reverse=True)
        mention_list += [k[0] for k in w_idf[:3]]
        mention_list = list(set(mention_list))
        random.shuffle(mention_list)
        mention_list = mention_list[:4]

        input_x = " and ".join(mention_list) # "generate with mixture of tags and keywords: " + 
        y_np = self.tokenizer(input_y, return_tensors="np")['input_ids'][0, :self.config.max_length]
        x_np = self.tokenizer(input_x, return_tensors="np")['input_ids'][0, :self.config.max_length]

        return x_np, y_np, input_y, input_x

def nlg_data_wrapper(config, dataset):
    encoder_input_ids, encoder_mask = process_tensor([d[0] for d in dataset], 0, output_mask=True)
    decoder_input_ids, decoder_mask = process_tensor([d[1] for d in dataset], 0, output_mask=True)
    decoder_input_ids[decoder_mask == 0] = -100
    gt_y = [d[2] for d in dataset]
    gt_x = [d[3] for d in dataset]
    if len(dataset[0]) == 5:
        task_index = torch.tensor([d[4] for d in dataset]).long()
    else:
        task_index = torch.tensor([0 for _ in range(len(dataset))]).long()

    return {"task_index": task_index, "encoder_input_ids": encoder_input_ids, "encoder_mask": encoder_mask, "decoder_input_ids": decoder_input_ids, "gt_y": gt_y, "gt_x": gt_x}

def nlg_get_data_loader(config, dataset, batch_size, shuffle=False):
    collate_fn = lambda d: nlg_data_wrapper(config, d)
    return DataLoader(dataset, 
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
