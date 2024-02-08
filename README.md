# Based: Simple linear attention language models balance the recall-throughput tradeoff

Based is an efficient architecture that approximates attention with linear attention to model long-range dependencies in the sequence plus short (i.e. dimension 64) sliding window attention to model fine-grained local dependencies in the sequence. In this repo, we are including code to train new models and to eval existing checkpoints on downstream tasks.

### Installations

We recommend using a new conda environment:
```
conda create --name based python=3.8
conda activate based

git clone git@github.com:HazyResearch/based.git
cd based
```

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .

# third party; to install flash-attention and the lm-eval harness. 
causal-conv1d==1.1.1
mamba-ssm==1.1.1

cd evals/lm-eval-harness
git submodule init
git submodule update
```

### Train


### Pretrained Checkpoints

We are releasing the following checkpoints for research, trained at the 360M and 1.3Bn parameter scales. Each checkpoint is trained on the same 10Bn tokens of the Pile corpus, using the same data order. The checkpoints are trained using the same code and infrastructure.  
- 360M parameters
    - [Based 360M]
- 1.3Bn parameters
    - [Based 1.3Bn](https://huggingface.co/hazyresearch/based-1.3b)
    - [Mamba 1.3Bn](https://huggingface.co/hazyresearch/mamba-1.3b)
    - [Transformer++ 1.3Bn](https://huggingface.co/hazyresearch/transformer-pp-1.3b)


### Downstream Evals

The following instructions can be used to reproduce our results on the [LM-Eval harness](https://github.com/EleutherAI/lm-evaluation-harness) using the pretrained checkpoints.

```
cd evals/
bash run_harness.sh
```




