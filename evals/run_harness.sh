CHECKPOINT_NAME=hazyresearch/based-1.3b
TASKS=hellaswag,lambada_openai,piqa,arc_easy,arc_challenge,winogrande
DEVICE=cuda:0
BATCH_SIZE=32

python run_harness.py --model based_hf \
    --model_args checkpoint_name=$CHECKPOINT_NAME \
    --tasks $TASKS \
    --device $DEVICE \
    --batch_size $BATCH_SIZE