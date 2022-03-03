from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup, Adafactor


class AdamWOpt(object):
    def __init__(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

def build_t5_finetune_optimizer(opt, model):
	optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
	return optimizer

def build_t5_pretraining_optimizer(opt, model):
	optimizer = Adafactor(model.parameters(), beta1=0, scale_parameter=True, relative_step=True, warmup_init=False, lr=None)
	lr_scheduler = AdafactorSchedule(optimizer)
	return AdamWOpt(optimizer, lr_scheduler)

def build_t5_optimizer(opt, model):
	optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=0.3, weight_decay=1e-5)
	return optimizer

def build_optimizer(opt, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": opt.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, eps=opt.adam_epsilon, betas=(0.9, 0.98))
    scheduler = get_constant_schedule(optimizer) 

    return AdamWOpt(optimizer, scheduler)

def build_warmup_optimizer(opt, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": opt.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, eps=opt.adam_epsilon, betas=(0.9, 0.98))
    warmup_step = (opt.num_training_steps * opt.warmup_ratio) if opt.warmup_ratio > 0 else opt.warmup_step
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, opt.num_training_steps, -1) 

    return AdamWOpt(optimizer, scheduler)

