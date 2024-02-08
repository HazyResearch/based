CHECKPOINT_NAME=hazy-research/attention/1nu4scd8
BASE_CHECKPOINT_DIR=/var/cr01_data/sim_data/checkpoints
TASKS=hellaswag,lambada_openai,piqa,arc_easy,arc_challenge,winogrande
DEVICE=cuda:0
BATCH_SIZE=32

python run_harness.py --model based_hf \
    --model_args checkpoint_name=$CHECKPOINT_NAME,base_checkpoint_dir=$BASE_CHECKPOINT_DIR \
    --tasks $TASKS \
    --device $DEVICE \
    --batch_size $BATCH_SIZE