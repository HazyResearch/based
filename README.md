<div align="center" >
    <img src="assets/banner.png" height=120 alt="" style="margin-bottom:px"/> 

**Simple linear attention language models balance the recall-throughput tradeoff.**

[![arXiv](https://img.shields.io/badge/arXiv-2402.18668-b31b1b.svg)](https://arxiv.org/abs/2402.18668)
[![GitHub](https://img.shields.io/github/license/HazyResearch/meerkat)](https://img.shields.io/github/license/HazyResearch/meerkat)

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/collections/hazyresearch/based-65d77fb76f9c813c8b94339c) 
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/collections/hazyresearch/based-65d77fb76f9c813c8b94339c)
<!-- [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) -->
<!-- [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/models) -->


</div>


Based is an efficient architecture inspired by recovering attention-like capabilities (i.e., *recall*). We do so by combining 2 simple ideas:
1. Short sliding window attention (e.g., window size 64), to model fine-grained local dependencies
2. "Dense" and global *linear* attention, to model long-range dependencies

In this way, we aim to capture the same dependencies as Transformers in a 100% subquadratic model, with *exact* softmax attention locally and a softmax-approximating linear attention for all other tokens. 

We find this helps close many of the performance gaps between Transformers and recent subquadratic alternatives (matching perplexity is not all you need? [[1](https://arxiv.org/abs/2312.04927), [2](https://arxiv.org/abs/2402.01032), [3](https://arxiv.org/abs/2402.04248)]).

In this repo, please find code to (1) train new models and (2) evaluate existing checkpoints on downstream tasks.

## Installation

**Note.** The code in this repository is tested on `python=3.8.18` and `torch=2.1.2`. We recommend using these versions in a clean environment. 

```bash
# clone the repository
git clone git@github.com:HazyResearch/based.git
cd based

# install torch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 # due to observed causal-conv1d dependency

# install based package
pip install -e .
```

## Pretrained Checkpoints

We are releasing the following checkpoints for research, trained at the 360M and 1.3B parameter scales. Each checkpoint is trained on the same 10B tokens of the Pile corpus, using the same data order. The checkpoints are trained using the same code and infrastructure.  

Use the code below to load the Based checkpoints:
```python  
import torch
from transformers import AutoTokenizer
from based.models.gpt import GPTLMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPTLMHeadModel.from_pretrained_hf("hazyresearch/based-360m").to("cuda", dtype=torch.float16)
```


| *Architecture* | *Size* | *Tokens*| *WandB* | *HuggingFace* | *Config* |
| ---          | ---  | ---   | ---   | --- | --- |
| **Based**        | 360m | 10b   |[02-20-based-360m](https://wandb.ai/hazy-research/based/runs/02-20-based-360m) |[hazyresearch/based-360m](https://huggingface.co/hazyresearch/based-360m)     |reference/based-360m.yaml |
| **Based**        | 1.4b | 10b   |[02-21-based-1b](https://wandb.ai/hazy-research/based/runs/02-24-based-1b)     |[hazyresearch/based-1b](https://huggingface.co/hazyresearch/based-1b)      |reference/based-1b.yaml |
| **Attention**    | 360m | 10b   |[02-21-attn-360m](https://wandb.ai/hazy-research/based/runs/02-21-attn-360m-redo1) |[hazyresearch/attn-360m](https://huggingface.co/hazyresearch/attn-360m)     |reference/attn-360m.yaml |
| **Attention**    | 1.4b | 10b   |[02-25-attn-1b](https://wandb.ai/hazy-research/based/runs/02-25-attn-1b) |[hazyresearch/attn-1b](https://huggingface.co/hazyresearch/attn-1b)     |reference/attn-360m.yaml |
| **Mamba**        | 360m | 10b   |[02-21-mamba-360m](https://wandb.ai/hazy-research/based/runs/02-21-mamba-360m) |[hazyresearch/mamba-360m](https://huggingface.co/hazyresearch/mamba-360m)     |reference/mamba-360m.yaml |
| **Mamba**        | 1.4b | 10b   |[02-22-mamba-1b](https://wandb.ai/hazy-research/based/runs/02-22-mamba-1b) |[hazyresearch/mamba-1b](https://huggingface.co/hazyresearch/mamba-1b)     |reference/mamba-1b.yaml |



**Warning.** We are releasing these models for the purpose of efficient architecture research. Because they have not been instruction fine-tuned or audited, they are not intended for use in any downstream applications. 

The following code will run text generation for a prompt and print out the response. 
```python
input = tokenizer.encode("If I take one more step, it will be", return_tensors="pt").to("cuda")
output = model.generate(input, max_length=20)
print(tokenizer.decode(output[0]))
```

**Note.** For the checkpoints from other models, you will need to install other dependencies and use slightly different code. 

To load the Attention models, use the following code:

```python  
import torch
from transformers import AutoTokenizer
from based.models.transformer.gpt import GPTLMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPTLMHeadModel.from_pretrained_hf("hazyresearch/attn-360m").to("cuda")
```

To use the Mamba checkpoints, first run `pip install mamba-ssm` and then use the following code:

```python  
import torch
from transformers import AutoTokenizer
from based.models.mamba import MambaLMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = MambaLMHeadModel.from_pretrained_hf("hazyresearch/mamba-360m").to("cuda")
```


## Train
In order to train a new model with our code, you'll need to do a bit of additional setup: 

```python
# install train extra dependencies
pip install -e .[train]

# install apex (if you run into issues, likely candidates are torch or pip version issues; if using torch 2.0.1, this may help https://github.com/NVIDIA/apex/issues/1735)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..
```

### Launching Training
Kernels for other fused operations: The config defaults will use fused kernels from the Flash Attention repo, which can all be installed by cloning the repo and ```python setup.py install``` the relevant [kernels here](https://github.com/Dao-AILab/flash-attention/tree/main/csrc). In particular, the fused_dense_lib, layer_norm, rotary, and xentropy kernels. Alternatively, you can change the codepaths to avoid the use of these kernels -- for instance by specifying fused_dense False in the experiment config, or by replacing the RMSNorm import in ```based/models/gpt.py``` to import from ```based/ops/triton/layer_norm```. 

To train a new model, construct a config.yaml file at ```train/configs/experiment/```. We are including the configs used to produce the pretrained checkpoints for the paper (released on HF below) at ```train/configs/experiment/reference/```.

You can launch a training job using the following command from the ```train/``` directory, where you can modify the config name and number of GPUs (```trainer.devices```):
```
cd train/
python run.py experiment=reference/based-1b trainer.devices=8
```

In our paper, we evaluated on the Pile corpus, which is no longer available online, so the ```train/configs/experiment/reference/``` configs are unfortunately not directly runnable. For your use, we are including an example config that would train on the WikiText103 language modeling data. You can launch using the following script:
```
cd train/
python run.py experiment=example/based-360m trainer.devices=8
```

You can adapt the training dataset by adding a new dataset config file under ```train/configs/datamodule/```. Follow the examples in ```wikitext103.yaml```. Once you've constructed the yaml file for your new dataset, go to the experiment config (e.g. ```train/configs/experiment/example/based-360m.yaml```) and update the name of the datamodule under ```override datamodule``` to the filename of your new dataset yaml file. 

Be sure to update the checkpointing directory [in the config](https://github.com/HazyResearch/based/blob/3fb009b8216b41d14ea3a2ab9552a5c609ef0bf4/train/configs/experiment/example/based-360m.yaml#L39) prior to launching training.


### Fast Training
We support a few different training views in this repo. The choice of ```parallel_implementation``` in your training config determines which training view gets used (https://github.com/HazyResearch/based/blob/e86e21401ad26e38a46590e73af43868f4a98b2a/based/models/mixers/linear_attention.py#L73). The default, which requires installing no kernels, simply retains a quadratic O(n^2) view during training. We currently recommend using Option 2 below for drastically faster training. These will be replaced with our new custom kernels (from the Based paper), to be released soon. 

- Option 1 (```parallel_implementation = "quadratic"```): default, quadratic PyTorch view.  
- Option 2 (```parallel_implementation = "fla_parallel"```): Flash linear attention kernel (https://github.com/sustcsonglin/flash-linear-attention). Use the following to install:
```
pip install triton==2.2.0
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```
- Option 3 (```parallel_implementation = "linear"```): Fast transformers linear attention kernel. Use the following to install:
```
cd train/csrc/causal_dot_prod/
python setup.py install
```

We have provided benchmarking plots for different kernels in the ```benchmark/examples/linear_attention_forward/``` folder. We are providing [WandB training curves here](https://api.wandb.ai/links/simarora/ryv84b55) showing how training using the ```fla-parallel``` mode allows Based to faster than Mamba at the 360M parameter scale at strong quality!


### Additional notes: 
- If you want to explore the optional decay strategy discussed in the Based paper, you can checkout the ```notebooks/03-31-decay.ipynb``` notebook.
- Note that this training code is from: https://github.com/Dao-AILab/flash-attention/tree/main/training, the Flash Linear Attention kernel is from https://github.com/sustcsonglin/flash-linear-attention, and the Fast Transformers kernel is from https://github.com/idiap/fast-transformers. Please cite them if you use their work!


## Evaluate
In our paper, we evaluate pretrained language models on a standard suite of benchmarks from the LM Evaluation Harness, as well as a suite of three *recall-intensive* tasks:

- **SWDE** (Info. extraction). A popular information extraction benchmark for semi-structured data. SWDE includes raw HTML documents from 8 Movie and 5 University websites (e.g.IMDB, US News) and annotations for 8-274 attributes per website (e.g., Movie runtime). **HuggingFace: [hazyresearch/based-swde](https://huggingface.co/datasets/hazyresearch/based-swde)**
- **FDA** (Info. extraction). A popular information extraction benchmark for unstructured data. The FDA setting contains 16 gold attributes and 100 PDF documents, which are up to 20 pages long, randomly sampled from FDA 510(k). **HuggingFace: [hazyresearch/based-fda](https://huggingface.co/datasets/hazyresearch/based-fda)**
- **SQUAD-Completion** (Document-QA). We find that original SQUAD dataset is challenging for our models without instruction fine-tuning. So we introduce a modified version of SQUAD where questions are reworded as next-token prediction tasks. For example, "What is the capital of France?" becomes "The capital of France is". **HuggingFace: [hazyresearch/based-squad](https://huggingface.co/datasets/hazyresearch/based-swde)**

Under `evaluate`, we have a clone of EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) that includes these new tasks and provides scripts for running all the evaluations from the paper. The following instructions can be used to reproduce our results on the [LM-Eval harness](https://github.com/EleutherAI/lm-evaluation-harness) using the pretrained checkpoints.


### Setup.
```bash
cd evaluate 

# init the submodule and install
git submodule init
git submodule update
pip install -e . 
```


### Running Evaluations.
We provide a script `evaluate/launch.py` that launch evaluations on the checkpoints we've released. 

For example, running the following from the `evaluate` folder will evaluate the 360M Based, Mamba, and Attention models on the SWDE dataset.

You can set your huggingface cache directory to a location with sufficient space (```export TRANSFORMERS_CACHE```, ```export HF_HOME```).

```bash
python launch.py \
    --task swde  --task fda --task squad_completion \
    --model "hazyresearch/based-360m" \
    --model "hazyresearch/mamba-360m" \
    --model "hazyresearch/attn-360m" \
    --model "hazyresearch/based-1b" \
    --model "hazyresearch/mamba-1b" \
    --model "hazyresearch/attn-1b"
```
Optionally, if you have access to multiple GPUs, you can pass the `-p` flag to run each evaluation on a different GPU. 
To run a limited number of samples for each task (_e.g._ 100), use the `--limit=100` option.

Below we include the results produced from running the command above. Note: the results below are on the new models trained and evaluated with the cleaned-up code in this repository. As a result, the results reported in our paper differ slightly, however the trends and conclusions remain the same.
| *Architecture* | *Size* |*HuggingFace* | *SWDE*| *FDA* | *SQUAD* |
| ---            | ---    | ---          | ---   | ---   | ---     |
| **Based**      | 360m   |[hazyresearch/based-360m](https://huggingface.co/hazyresearch/based-360m)  |25.65  |14.34  |24.23    |
| **Mamba**      | 360m   |[hazyresearch/mamba-360m](https://huggingface.co/hazyresearch/mamba-360m)  |17.28  |5.90   |24.83    |
| **Attention**  | 360m   |[hazyresearch/attn-360m](https://huggingface.co/hazyresearch/attn-360m)    |56.26  |57.89  |27.85    |
| **Based**      | 1.4b   |[hazyresearch/attn-1b](https://huggingface.co/hazyresearch/based-1b)    |37.71  |19.06  |29.49    |
| **Mamba**      | 1.4b   |[hazyresearch/attn-1b](https://huggingface.co/hazyresearch/mamba-1b)    |28.35  |11.07  |29.42    |
| **Attention**  | 1.4b   |[hazyresearch/attn-1b](https://huggingface.co/hazyresearch/attn-1b)    |69.04  |68.87  |35.89    |

Note that the results shown may differ slightly if the Flash-Attention kernels are not used during inference.

## Experiments on Synthetic Data
In our paper, we demonstrate the recall-throughput tradeoff using a synthetic associative recall task (see Figure 2, below, and Figure 3 in the paper). 
<div align="center" >
    <img src="assets/tradeoff.png" height=200 alt="" style="margin-bottom:px"/> 
</div>

The code for reproducing these figures is provided in a separate repository: [HazyResearch/zoology](https://github.com/HazyResearch/zoology). Follow the setup instruction in the Zoology README. The instructions for reproducing the are provided in [zoology/experiments](https://github.com/HazyResearch/zoology/tree/main/zoology/experiments). For example, you can create the figure above using. 

```
python -m zoology.launch zoology/experiments/arxiv24_based_figure2/configs.py -p
```


## Benchmarking and Efficiency

We include the kernels evaluated in the Based paper under ```based/benchmarking/```. We provide additional details on the CUDA releases in the README in this folder. Stay tuned!


## Citation and Acknowledgements

This repo contains work based on the following papers. Please consider citing if you found the work or code useful:
```
# Based
@article{arora2024simple,
  title={Simple linear attention language models balance the recall-throughput tradeoff},
  author={Arora, Simran and Eyuboglu, Sabri and Zhang, Michael and Timalsina, Aman and Alberti, Silas and Zinsley, Dylan and Zou, James and Rudra, Atri and Ré, Christopher},
  journal={arXiv:2402.18668},
  year={2024}
}

# Hedgehog (Linear attention)
@article{zhang2024hedgehog,
  title={The Hedgehog \& the Porcupine: Expressive Linear Attentions with Softmax Mimicry},
  author={Zhang, Michael and Bhatia, Kush and Kumbong, Hermann and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2402.04347},
  year={2024}
}

# Zoology (BaseConv, Synthetics, Recall Problem)
@article{arora2023zoology,
  title={Zoology: Measuring and Improving Recall in Efficient Language Models},
  author={Arora, Simran and Eyuboglu, Sabri and Timalsina, Aman and Johnson, Isys and Poli, Michael and Zou, James and Rudra, Atri and Ré, Christopher},
  journal={arXiv:2312.04927},
  year={2023}
}
```


This project was made possible by a number of other open source projects; please cite if you use their work! Notably:
- Our training code and sliding window implementation are based on Tri Dao's [FlashAttention](https://github.com/Dao-AILab/flash-attention). 
- We use EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation. 
- We use the conv1d kernel from [Mamba](https://github.com/state-spaces/mamba/tree/main).
- We integrated the causal dot product kernel from [Fast Transformers](https://github.com/idiap/fast-transformers).
- We integrated the based kernels from [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention).


Models in this project were trained using compute provided by:  
- [Together.ai](https://www.together.ai/)
- Google Cloud Platform through [Stanford HAI](https://hai.stanford.edu/call-google-cloud-credit-proposals)


Please reach out with feedback and questions!
