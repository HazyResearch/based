from dataclasses import dataclass, field
import importlib
import os
import datetime
import time

import torch
import click
import pandas as pd 
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from config import ModelConfig

DEVICE = "cuda"
torch.random.manual_seed(0)


def benchmark(config: DictConfig):
    return _benchmark(ModelConfig(
        config=config.model.config,
        _target_=config.model._target_,
        generate_kwargs={
            "enable_timing": False,
            "cg": True
        },
    ))

def _benchmark(
    config: ModelConfig,
    fn: str = "generate",
    prefill_size: int=128,
    n_generated_tokens: int=100, 
    batch_size: int=1, 
    repeats: int=1,
    name: str="test",
    info: dict={}
):
    results = []

    # create model
    model_config = hydra.utils.instantiate(
        config.config, _recursive_=False, _convert_="object"
    )
    model = hydra.utils.instantiate(
        config={}, _target_=config._target_,  _args_=[model_config], _recursive_=False
    ).to(device=DEVICE, dtype=config.dtype)

    # get model size in params
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{num_params=:.2e}")

    # prepare inputs
    torch.random.manual_seed(0)
    input_ids = torch.randint(1, 1000, (batch_size, prefill_size), dtype=torch.long, device=DEVICE)

    if "attention_mask" in config.generate_kwargs:
        config.generate_kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long, device=DEVICE)
    model.eval()

    if fn == "generate":
        fn = lambda: model.generate(
            input_ids=input_ids, 
            max_length=prefill_size + n_generated_tokens, 
            return_dict_in_generate=False, 
            output_scores=False, 
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            **config.generate_kwargs
        )
    elif fn == "forward":
        fn = lambda: model(input_ids, **config.generate_kwargs)
    else:
        raise ValueError(f"Invalid fn: {fn}")

    torch.cuda.synchronize()
    for repeat in range(repeats + 1):
        torch.cuda.reset_peak_memory_stats()
        oom = False
        start = time.time()
        try:
            fn()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as e:
            print("OOM!")
            oom = True
        except RuntimeError as e:
            # SE: catch CUDNN_STATUS_INTERNAL_ERROR which is commonly thrown 
            # when there's an OOM error: 
            # https://stackoverflow.com/questions/62067849/pytorch-model-training-runtimeerror-cudnn-error-cudnn-status-internal-error
            print("RuntimeError!", f"{'CUDNN_STATUS_INTERNAL_ERROR' in str(e)=}, {str(e)}")
            if "CUDNN_STATUS_INTERNAL_ERROR" in str(e):
                oom = True
                print(e)
            else:
                raise e

        latency = (time.time() - start) * 1000
        peak_mem = torch.cuda.max_memory_allocated(device=DEVICE)
        
        if repeat == 0:
            # need to do one warmup run to cache the computation graph 
            # (since cg=True may be set)
            continue

        results.append(
            {
                "name": name,
                "generation_latency": latency,
                "n_generated_tokens": n_generated_tokens,
                "peak_mem": peak_mem,
                "params": num_params,
                "batch_size": batch_size,
                "prefill_size": prefill_size,
                "oom": oom,
                **info
            }
        )
    print(results)
    return results



@click.command()
@click.argument("python_file", type=click.Path(exists=True))
@click.option("--outdir", type=click.Path(exists=True, file_okay=False, writable=True), default=None,)
@click.option("--name", type=str, default="default")
@click.option("-p", "--parallelize", is_flag=True)
@click.option("--gpus", default=None, type=str)
def main(python_file, outdir, name: str, parallelize: bool, gpus: str):

    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


    # Load the given Python file as a module
    spec = importlib.util.spec_from_file_location("config_module", python_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    configs = config_module.configs
    use_ray = parallelize and len(configs) > 0

    results = []
    # Run each script in parallel using Ray
    if not use_ray:
        for config in tqdm(configs): 
            out = _benchmark(**config)
            results.extend(out)
    else:
        import ray
        # ray was killing workers due to OOM, but it didn't seem to be necessary 
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"
        ray.init(ignore_reinit_error=True, log_to_driver=True)

        completed = 0
        total = len(configs)
        print(f"Completed: {completed} ({completed / total:0.1%}) | Total: {total}")

        remote = ray.remote(num_gpus=1)(_benchmark)
        futures = [remote.remote(**config) for config in configs]
        
        while futures:
            complete, futures = ray.wait(futures)
            completed += len(complete)
            for out in ray.get(complete):
                results.extend(out)
            print(f"Completed: {completed} ({completed / total:0.1%}) | Total: {total}")

        ray.shutdown()

    # get a run_dir using the date and time
    now = datetime.datetime.now()
    run_dir = f"output/benchmark_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(run_dir)

    df = pd.DataFrame(results)
    path = f"{run_dir}/results.csv"
    df.to_csv(path)

    # filter out OOMs
    df = df[df["oom"] == False]

    # print the results by prefill_size
    for prefill_size, group in df.groupby("prefill_size"):
        print(f"Prefill size: {prefill_size}")
        print(group.groupby("name")["generation_latency"].mean().reset_index())    
    print(f"{path}")

if __name__ == "__main__":
    main()