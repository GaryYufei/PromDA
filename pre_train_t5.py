import argparse
import torch
import torch.nn as nn
from config import Config
import os, sys, math
from pre_train_dataset import PreTrainDataset, get_data_loader
import t5_model
import numpy as np
import random
import utils
from checkpointing import CheckpointManager
from tqdm import tqdm

def evaluation(_C, eval_data, model, device):
	model.eval()
	loss_list = []
	with torch.no_grad():
		for batch in tqdm(eval_data):
			for n in batch:
				batch[n] = batch[n].to(device)

			outputs = model(
			    input_ids=batch['encoder_input_ids'], 
			    attention_mask=batch['encoder_mask'], 
			    labels=batch['decoder_input_ids'],
			)
			loss = outputs.loss

			loss_list.append(loss.item())

	final_loss = sum(loss_list) / len(loss_list)
	print("EVAL LOSS %.2f" % final_loss)
	return -1 * final_loss

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

	if _C.enable_full_finetune:
		tokenizer, model = t5_model.get_full_finetune_t5_model(_C)
	elif _C.enable_full_pretrain:
		tokenizer, model = t5_model.get_full_pretrain_t5_model(_C)
	else:
		tokenizer, model = t5_model.get_t5_model(_C)

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

	dev_data = PreTrainDataset(_C, tokenizer, _C.dev_path)
	dev_loader = get_data_loader(_C, dev_data, _C.batch_size, shuffle=False)

	test_data = PreTrainDataset(_C, tokenizer, _C.dev_path)
	test_loader = get_data_loader(_C, test_data, _C.batch_size, shuffle=False)

	if _A.validation or _A.test:
		if torch.cuda.is_available():
			model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'))['model'], strict=False)
		else:
			model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))['model'], strict=False)
		selected_data = dev_loader if _A.validation else test_loader
		evaluation(_C, selected_data, model, device)

	if _A.train:
		train_data = PreTrainDataset(_C, tokenizer, _C.train_path)
		train_loader = get_data_loader(_C, train_data, _C.batch_size, shuffle=True)
		train_iter = iter(train_loader)

		if _C.num_training_steps == 0:
			_C.num_training_steps = int(len(train_iter) * _C.max_epoch / _C.gradient_accumulation_steps)
		epoch_num = math.ceil(_C.num_training_steps / _C.checkpoint_every_step)

		if _C.enable_full_finetune:
			optimizer = utils.build_t5_finetune_optimizer(_C, model)
		elif _C.enable_full_pretrain:
			optimizer = utils.build_t5_pretraining_optimizer(_C, model)
		elif _C.enable_adam_opt:
			optimizer = utils.build_optimizer(_C, model)
		else:
			optimizer = utils.build_t5_optimizer(_C, model)

		os.makedirs(_A.serialization_dir, exist_ok=True)
		_C.dump(os.path.join(_A.serialization_dir, "config.yml"))

		checkpoint_manager = CheckpointManager(model, _A.serialization_dir, mode="max")

		eval_every = _C.checkpoint_every_step * _C.gradient_accumulation_steps
		total_step = 0
		total_oom = 0
		best_test_performance = 0
		model.parallelize()

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
						batch[n] = batch[n].to(device)
					total_step += 1

					try:
						outputs = model(
						    input_ids=batch['encoder_input_ids'], 
						    attention_mask=batch['encoder_mask'], 
						    labels=batch['decoder_input_ids'],
						)
						loss = outputs.loss
						loss = loss / _C.gradient_accumulation_steps
						loss.backward()
					except RuntimeError:
						torch.cuda.empty_cache()
						total_oom += 1
						pbar.set_description("OOM %d loss -" % (total_oom))
						pbar.update(1)
						continue

					if (step + 1) % _C.gradient_accumulation_steps == 0:
						optimizer.step()
						if torch.cuda.is_initialized():
							torch.cuda.synchronize()
						pbar.set_description("OOM %d loss %.2f" % (total_oom, loss.item() * _C.gradient_accumulation_steps))
						pbar.update(1)
						optimizer.zero_grad()

			_score = evaluation(_C, dev_loader, model, device)
			checkpoint_manager.step(_score)
				