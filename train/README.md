
# Training Based models

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

We breakdown this section into three parts: 1) how to set up a training config and launch; 2) how to set up fast training kernels, and 3) how to install extra optimizations for training.

### Launching Training
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
We support a few different training views in this repo. The choice of ```parallel_implementation``` in your training config determines which training view gets used: 
https://github.com/HazyResearch/based/blob/e86e21401ad26e38a46590e73af43868f4a98b2a/based/models/mixers/linear_attention.py#L73
The default, which requires installing no kernels, simply retains a quadratic O(n^2) view during training. We currently recommend using Option 2 below for drastically faster training. These will be replaced with our new custom kernels (from the Based paper), to be released soon. 

- Option 1 (```parallel_implementation = "quadratic"```): default, quadratic PyTorch view.  
- Option 2 (```parallel_implementation = "fla_parallel"```): Flash linear attention kernel. Use the following to install:
```
pip install triton==2.2.0
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```
- Option 3 (```parallel_implementation = "linear"```): Fast transformers linear attention kernel. Use the following to install:
```
cd train/csrc/causal_dot_prod/
python setup.py install
```

We have provided benchmarking plots for different kernels in the [benchmark/examples/linear_attention_forward/](https://github.com/HazyResearch/based/tree/main/benchmark) folder. We are providing [WandB training curves here](https://api.wandb.ai/links/simarora/ryv84b55) showing how training using the ```fla-parallel``` mode allows Based to train fast with strong quality!


### Additional notes: 
- **Kernels for other fused operations:** The config defaults will use fused kernels from the Flash Attention repo, which can all be installed by cloning the Flash Attention repo and ```python setup.py install``` the relevant [kernels here](https://github.com/Dao-AILab/flash-attention/tree/main/csrc). In particular, the fused_dense_lib, layer_norm, rotary, and xentropy kernels. Alternatively, you can change the codepaths to avoid the use of these kernels -- for instance by specifying fused_dense False in the experiment config, or by replacing the RMSNorm import in ```based/models/gpt.py``` to import from ```based/ops/triton/layer_norm```. 
- **Decay** If you want to explore the optional decay strategy discussed in the Based paper, you can checkout the [notebooks/03-31-decay.ipynb](https://github.com/HazyResearch/based/blob/main/notebooks/03-31-decay.ipynb) notebook.
- **References** Note that this training code is from: https://github.com/Dao-AILab/flash-attention/tree/main/training, the Flash Linear Attention kernel is from https://github.com/sustcsonglin/flash-linear-attention, and the Fast Transformers kernel is from https://github.com/idiap/fast-transformers. **Please cite them if you use their work!**
