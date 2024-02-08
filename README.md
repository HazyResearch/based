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
python setup.py install

# third party; to install flash-attention and the lm-eval harness. 
causal-conv1d==1.1.1
mamba-ssm==1.1.1
```

### Train



### Evals

The following instructions can be used to reproduce our results on the [LM-Eval harness](https://github.com/EleutherAI/lm-evaluation-harness)

```
cd evals/

```



