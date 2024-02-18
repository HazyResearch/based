# TASKS=hellaswag,lambada_openai,piqa,arc_easy,arc_challenge,winogrande
TASKS=swde
DEVICE=cuda:0
BATCH_SIZE=32

# MAMBA
CHECKPOINT_NAME=hazyresearch/mamba-1.3b
MODEL=mamba
python run_harness.py --model based_hf \
    --model_args checkpoint_name=$CHECKPOINT_NAME,model=$MODEL \
    --tasks $TASKS \
    --device $DEVICE \
    --batch_size $BATCH_SIZE

# # BASED
# CHECKPOINT_NAME=hazyresearch/based-1.3b
# MODEL=based
# python run_harness.py --model based_hf \
#     --model_args checkpoint_name=$CHECKPOINT_NAME,model=$MODEL \
#     --tasks $TASKS \
#     --device $DEVICE \
#     --batch_size $BATCH_SIZE

# # TRANSFORMER ++
# CHECKPOINT_NAME=hazyresearch/transformer-pp-1.3b
# MODEL=transformer
# python run_harness.py --model based_hf \
#     --model_args checkpoint_name=$CHECKPOINT_NAME,model=$MODEL \
#     --tasks $TASKS \
#     --device $DEVICE \
#     --batch_size $BATCH_SIZE