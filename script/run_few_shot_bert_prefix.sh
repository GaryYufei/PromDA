#!/bin/bash

GPUID=$1
RANDOM_SEED=$2
SHOT=$3
DATASET=$4
LM_STEP=$5
if [ $# -eq 6 ]
then
	MAX_LENGTH=$6
else
	MAX_LENGTH=256
fi

SAMPLE_WITH_LABELS=1
EVAL_REPEAT_NUM=40
EVAL_SAMPLE_NUM=5
OVERSAMPLE=100

DATA_ROOT=./NLU_training_dataset/$DATASET
CONFIG_ROOT=./model_config/$DATASET
OUTPUT_ROOT=$DATASET\_$RANDOM_SEED\_shot_$SHOT\_tag_keyword_two_prefix_$LM_STEP\_step
PRETRAINED_PT_LM=pretrain_web_page_keyword_t5_short

# generate few-shot training data & unlabeled data.
if [[ $SAMPLE_WITH_LABELS -eq 1 ]]
then
  python generate_few_shot_data.py --data-path $DATA_ROOT --output-path $OUTPUT_ROOT --few-shot-k $SHOT --random-seed $RANDOM_SEED --min-length 0
else
  python generate_few_shot_data.py --data-path $DATA_ROOT --output-path $OUTPUT_ROOT --total-training-num $SHOT --random-seed $RANDOM_SEED --min-length 0
fi

CUDA_VISIBLE_DEVICES=$GPUID python train_data_agumentation.py --config $CONFIG_ROOT/nlg_prefix.yml \
 				--config-override checkpoint_every_step $LM_STEP num_training_steps $LM_STEP load_from_pretrained True training_da_mode "['keyword','tag']" eval_da_mode "['tag']" max_length $MAX_LENGTH train_path $OUTPUT_ROOT/train_whole_$SHOT.txt random_seed $RANDOM_SEED dev_path $OUTPUT_ROOT/train_whole_$SHOT.txt \
 				--train --serialization-dir $OUTPUT_ROOT/nlg_model_mix --start-from-checkpoint $PRETRAINED_PT_LM
				
CUDA_VISIBLE_DEVICES=$GPUID python train_data_agumentation.py --config $CONFIG_ROOT/nlg_prefix.yml \
				--config-override eval_da_mode "['tag']" max_length $MAX_LENGTH train_path $OUTPUT_ROOT/train_whole_$SHOT.txt random_seed $RANDOM_SEED test_path $OUTPUT_ROOT/train_whole_$SHOT.txt eval_data_replication $EVAL_REPEAT_NUM sample_num $EVAL_SAMPLE_NUM enable_filtering_error True \
				--start-from-checkpoint $OUTPUT_ROOT/nlg_model_mix --test --output-path $OUTPUT_ROOT/nlg_model_mix_output_part1.txt 

CUDA_VISIBLE_DEVICES=$GPUID python train_data_agumentation.py --config $CONFIG_ROOT/nlg_prefix.yml \
				--config-override eval_da_mode "['keyword']" max_length $MAX_LENGTH train_path $OUTPUT_ROOT/train_whole_$SHOT.txt random_seed $RANDOM_SEED test_path $OUTPUT_ROOT/train_whole_$SHOT.txt eval_data_replication $EVAL_REPEAT_NUM sample_num $EVAL_SAMPLE_NUM enable_filtering_error True \
				--start-from-checkpoint $OUTPUT_ROOT/nlg_model_mix --test --output-path $OUTPUT_ROOT/nlg_model_mix_output_part2.txt 

CUDA_VISIBLE_DEVICES=$GPUID python train_data_agumentation.py --config $CONFIG_ROOT/nlg_prefix.yml \
				--config-override eval_da_mode "['tag']" max_length $MAX_LENGTH train_path $OUTPUT_ROOT/train_whole_$SHOT.txt random_seed $RANDOM_SEED test_path $OUTPUT_ROOT/nlg_model_mix_output_part2.txt eval_data_replication 1 sample_num 1 enable_filtering_error True \
				--start-from-checkpoint $OUTPUT_ROOT/nlg_model_mix --test --output-path $OUTPUT_ROOT/nlg_model_mix_output_part3.txt 

CUDA_VISIBLE_DEVICES=$GPUID python train_data_agumentation.py --config $CONFIG_ROOT/nlg_prefix.yml \
				--config-override eval_da_mode "['keyword']" max_length $MAX_LENGTH train_path $OUTPUT_ROOT/train_whole_$SHOT.txt random_seed $RANDOM_SEED test_path $OUTPUT_ROOT/nlg_model_mix_output_part1.txt eval_data_replication 1 sample_num 1 enable_filtering_error True \
				--start-from-checkpoint $OUTPUT_ROOT/nlg_model_mix --test --output-path $OUTPUT_ROOT/nlg_model_mix_output_part4.txt

python LM_data_filtering.py --output-path $OUTPUT_ROOT/clean_nlg_mix_tag_output.txt --data-path $OUTPUT_ROOT/nlg_model_mix_output_part1.txt $OUTPUT_ROOT/nlg_model_mix_output_part3.txt 
python LM_data_filtering.py --output-path $OUTPUT_ROOT/clean_nlg_mix_keyword_output.txt --data-path $OUTPUT_ROOT/nlg_model_mix_output_part2.txt $OUTPUT_ROOT/nlg_model_mix_output_part4.txt 			

# train NLU & NLG model only using few-shot training data
CUDA_VISIBLE_DEVICES=$GPUID python train_SeqLabel.py --config $CONFIG_ROOT/bert_nlu.yml --config-override train_path $OUTPUT_ROOT/train_whole_$SHOT.txt random_seed $RANDOM_SEED --serialization-dir $OUTPUT_ROOT/bert_nlu_model_1 --train
for i in 1 2 3
do
	# NLU model run for consistency filtering
	CUDA_VISIBLE_DEVICES=$GPUID python train_SeqLabel.py --config $CONFIG_ROOT/bert_nlu.yml --start-from-checkpoint $OUTPUT_ROOT/bert_nlu_model_$i --test \
					--config-override test_path $OUTPUT_ROOT/clean_nlg_mix_tag_output.txt enable_consistency_filtering True \
					--output-path $OUTPUT_ROOT/consistency_nlg_mix_tag_output_$i.txt

	CUDA_VISIBLE_DEVICES=$GPUID python train_SeqLabel.py --config $CONFIG_ROOT/bert_nlu.yml --start-from-checkpoint $OUTPUT_ROOT/bert_nlu_model_$i --test \
					--config-override test_path $OUTPUT_ROOT/clean_nlg_mix_keyword_output.txt enable_consistency_filtering True \
					--output-path $OUTPUT_ROOT/consistency_nlg_mix_keyword_output_$i.txt

	# Using consistency filtering data to train a new NLU model
	CUDA_VISIBLE_DEVICES=$GPUID python train_SeqLabel.py --config $CONFIG_ROOT/bert_nlu.yml --serialization-dir $OUTPUT_ROOT/bert_nlu_model_$((i+1)) --train \
					--config-override oversample $OVERSAMPLE random_seed $RANDOM_SEED train_path $OUTPUT_ROOT/train_whole_$SHOT.txt lm_gen_train_path_list "['$OUTPUT_ROOT/consistency_nlg_mix_tag_output_$i.txt','$OUTPUT_ROOT/consistency_nlg_mix_keyword_output_$i.txt']"
done
