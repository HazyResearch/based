# Based: Simple linear attention language models balance the recall-throughput tradeoff


<div align="center" >
    <!-- <img src="assets/banner.png" height=150 alt="" style="margin-bottom:px"/>  -->
    BASED

[![GitHub](https://img.shields.io/github/license/HazyResearch/meerkat)](https://img.shields.io/github/license/HazyResearch/meerkat)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Simple linear attention language models balance the recall-throughput tradeoff.**


</div>


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
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 # due to observed causal-conv1d dependency
pip install -e .

# third party installs
pip3 install causal-conv1d      # for mamba 
pip3 install mamba-ssm==1.1.1

cd evals/lm-eval-harness        # for lm-eval harness
git submodule init
git submodule update
```

### Train

To train a new model, construct a config.yaml file at ```based/configs/experiment/```. We are including the configs used to produce the pretrained checkpoints for the paper (released on HF below) at ```based/configs/experiment/reference/```.

You can launch a training job using the following command from the ```based/based/``` directory, where you can modify the config name and number of GPUs (```trainer.devices```):
```python run.py experiment=reference/based-1.3b trainer.devices=8```


### Pretrained Checkpoints

We are releasing the following checkpoints for research, trained at the 360M and 1.3Bn parameter scales. Each checkpoint is trained on the same 10Bn tokens of the Pile corpus, using the same data order. The checkpoints are trained using the same code and infrastructure.  
- 360M parameters
    - [Based 360M]
- 1.3Bn parameters
    - [Based 1.3Bn](https://huggingface.co/hazyresearch/based-1.3b)
    - [Mamba 1.3Bn](https://huggingface.co/hazyresearch/mamba-1.3b)
    - [Transformer++ 1.3Bn](https://huggingface.co/hazyresearch/transformer-pp-1.3b). Transformer++ refers to the modern [Llama Architecture](https://github.com/facebookresearch/llama), which uses SwiGLU, Rotary, RMSNorm. 


### Downstream Evals

The following instructions can be used to reproduce our results on the [LM-Eval harness](https://github.com/EleutherAI/lm-evaluation-harness) using the pretrained checkpoints.

```
cd evals/
bash run_harness.sh
```




