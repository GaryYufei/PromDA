import argparse
import torch
import torch.nn as nn
from t5_model import *
from config import Config
import os, sys, math
from checkpointing import CheckpointManager
import random
from pre_train_dataset import *
import utils
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

def get_label_dict(label_path):
    label_dict = {}
    with open(label_path) as out:
        for l in out.readlines():
            l = l.strip()
            label_dict[l] = len(label_dict)
    return label_dict

def evaluation(config, eval_data, model, label_map, device, show_detail=False, output_path=None):
	model.eval()
	model = model.to(device)
	preds = None
	pad_token_label_id = -100
	input_words = []
	scores = []
	softmax = torch.nn.Softmax(dim=-1)
	with torch.no_grad():
		for batch in tqdm(eval_data):
			for n in batch:
				if batch[n] is not None and n not in ['gt_x']:
					batch[n] = batch[n].to(device)

			outputs = model(
			    input_ids=batch['input_ids'], 
			    attention_mask=batch['attention_mask']
			)

			logits = outputs[0]
			logits = softmax(logits)
			max_score, _ = torch.max(logits, dim=-1)
			mean_score = torch.sum(max_score * batch['attention_mask'], dim=1) / torch.sum(batch['attention_mask'], dim=1)

			if preds is None:
				preds = logits.detach().cpu().numpy()
				out_label_ids = batch["labels"].detach().cpu().numpy()
			else:
				preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
				out_label_ids = np.append(out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)

			input_words += batch['gt_x']
			scores += mean_score.detach().cpu().numpy().tolist()

	preds = np.argmax(preds, axis=2)

	preds_list = [[] for _ in range(out_label_ids.shape[0])]
	out_label_list = [[] for _ in range(out_label_ids.shape[0])]

	for i in range(out_label_ids.shape[0]):
		for j in range(out_label_ids.shape[1]):
			if out_label_ids[i, j] != pad_token_label_id:
				preds_list[i].append(label_map[preds[i][j]])
				out_label_list[i].append(label_map[out_label_ids[i][j]])

	new_F = f1_score(out_label_list, preds_list) * 100
	print("all data Performance %.2f" % new_F)
	selected_words, selected_tags = [], []

	if config.score_top_ratio > 0:
		selected_gt, selected_pred = [], []
		rank_list = [(w, t, gt, s) for (w, t, gt, s) in zip(input_words, preds_list, out_label_list,  scores)] 
		sorted_rank_list = sorted(rank_list, key=lambda x: x[3], reverse=True)
		select_num = int(len(sorted_rank_list) * config.score_top_ratio)
		for input_word, tag, gt_tag, _ in sorted_rank_list[:select_num]:
			selected_words.append(input_word)
			selected_tags.append(tag)
			selected_pred.append(tag)
			selected_gt.append(gt_tag)
		F = f1_score(selected_gt, selected_pred) * 100
		print("Selected Performance %.2f" % F)
	else:
		for input_word, gt_label, pred_label in zip(input_words, out_label_list, preds_list):
			if config.enable_consistency_filtering:
				if len(input_word) > config.filter_by_min_length and ' '.join(gt_label) == ' '.join(pred_label):
					selected_words.append(input_word)
					selected_tags.append(pred_label)
			else:
				selected_words.append(input_word)
				selected_tags.append(pred_label)

	if output_path is not None:
		with open(output_path, 'w') as out:
			for gen, labels in zip(selected_words, selected_tags):
				for g, l in zip(gen, labels):
					out.write("%s %s\n" % (g, l))
				out.write("\n")

	return new_F



parser = argparse.ArgumentParser("Train a MT5 for Machine Translation")
parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
)
parser.add_argument(
    "--serialization-dir",
    default=None,
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--start-from-checkpoint",
    default=None,
    help="Path to load checkpoint and continue training [only supported for module_training].",
)
parser.add_argument(
    "--output-path",
    default=None,
    help="Path to save output captions",
)
parser.add_argument(
    "--pre-compute",
    action='store_true',
    help="Pre Compute",
)
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', action='store_true')
group.add_argument('--validation', action='store_true')
group.add_argument('--test', action='store_true')

if __name__ == "__main__":
	_A = parser.parse_args()
	_C = Config(_A.config, _A.config_override)

	np.random.seed(_C.random_seed)
	random.seed(_C.random_seed)
	torch.manual_seed(_C.random_seed)
	torch.cuda.manual_seed_all(_C.random_seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	label_dict = get_label_dict(_C.label_path)
	label_map = {v:k for (k,v) in label_dict.items()}
	tokenizer, model = get_bert_model(_C, len(label_dict))
	model = model.to(device)

	total_parameter_count = 0
	trainable_parameter_count = 0
	for p in model.parameters():
		total_parameter_count += p.numel()
		if p.requires_grad:
			trainable_parameter_count += p.numel()
	print('Total Parameter Count %d' % total_parameter_count)
	print('Trainable Parameter Count %d' % trainable_parameter_count)

	print(_C)
	for arg in vars(_A):
		print("{:<20}: {}".format(arg, getattr(_A, arg)))

	dev_data = SeqLabelDataset(_C, _C.dev_path, label_dict, tokenizer)
	dev_loader = get_seq_data_loader(_C, dev_data, _C.batch_size if _C.val_batch_size < 0 else _C.val_batch_size)

	test_data = SeqLabelDataset(_C, _C.test_path, label_dict, tokenizer)
	test_loader = get_seq_data_loader(_C, test_data, _C.batch_size if _C.val_batch_size < 0 else _C.val_batch_size)

	if _A.validation or _A.test:
		if torch.cuda.is_available():
			model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'))['model'], strict=False)
		else:
			model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)
		selected_data = dev_loader if _A.validation else test_loader
		evaluation(_C, selected_data, model, label_map, device, show_detail=True, output_path=_A.output_path)

	if _A.train:
		train_data = SeqLabelDataset(_C, _C.train_path, label_dict, tokenizer, is_training=True)
		train_loader = get_seq_data_loader(_C, train_data, _C.batch_size, shuffle=True)
		train_iter = iter(train_loader)

		if _C.num_training_steps == 0:
			_C.num_training_steps = int(len(train_iter) * _C.max_epoch / _C.gradient_accumulation_steps)
		epoch_num = math.ceil(_C.num_training_steps / _C.checkpoint_every_step)

		if _C.warmup_step == 0 and _C.warmup_ratio == 0:
			optimizer = utils.build_optimizer(_C, model)
		else:
			optimizer = utils.build_warmup_optimizer(_C, model)

		os.makedirs(_A.serialization_dir, exist_ok=True)
		_C.dump(os.path.join(_A.serialization_dir, "config.yml"))

		checkpoint_manager = CheckpointManager(model, _A.serialization_dir, mode="max")

		eval_every = _C.checkpoint_every_step * _C.gradient_accumulation_steps
		total_step = 0
		best_test_performance = 0
		for epoch in range(epoch_num):
			print('EPOCH %d / %d' % (epoch + 1, epoch_num))
			run_step = eval_every if total_step + eval_every < _C.num_training_steps * _C.gradient_accumulation_steps else  _C.num_training_steps * _C.gradient_accumulation_steps - total_step
			model.train()

			with tqdm(total=math.ceil(run_step / _C.gradient_accumulation_steps), file=sys.stdout) as pbar:
				for step in range(run_step):
					try:
						batch = next(train_iter)
					except:
						train_iter = iter(train_loader)
						batch = next(train_iter)

					for n in batch:
						if batch[n] is not None and n not in ['gt_x']:
							batch[n] = batch[n].to(device)
					total_step += 1

					outputs = model(
					    input_ids=batch['input_ids'], 
					    attention_mask=batch['attention_mask'], 
					    labels=batch['labels']
					)
					loss = outputs.loss
					loss = loss / _C.gradient_accumulation_steps
					loss.backward()

					if (step + 1) % _C.gradient_accumulation_steps == 0:
						torch.nn.utils.clip_grad_norm_(model.parameters(), _C.max_grad_norm)
						optimizer.step()
						if torch.cuda.is_initialized():
							torch.cuda.synchronize()
						pbar.set_description("loss %.2f" % (loss.item() * _C.gradient_accumulation_steps))
						pbar.update(1)
						optimizer.zero_grad()

			_score = evaluation(_C, dev_loader, model, label_map, device)
			update_test = checkpoint_manager.step(_score)
			if update_test:
				best_test_performance = evaluation(_C, test_loader, model, label_map, device)
			print("best test performance %.2f" % best_test_performance)


	