
# Benchmarking

Updates Log:
- Release Feb 2024: Our preliminary release includes the sketches for the three kernels we introduce in the ```examples/``` folder. We will be releasing the source code -- an exciting new CUDA framework from our lab -- in an upcoming release! Additionally, note that the kernels in the paper / this initial release are used for benchmarking the algorithms and have not been used in training thus far (discussed in paper Appendix as well). Training kernels are also upcoming!


### Setup

Source code containing helper functions and data types for the kernels are in ```src/```. To setup the environment correctly, please run the commands provided in ```env.src```.
You will need to use gcc ++20 to compile the kernels.

If you run into an error ```nvcc fatal   : Value 'c++20' is not defined for option 'std'```, then confirm that you're pointing to an updated version of CUDA, and that yyou're pointing to a matching version of torch.
```
export CUDA_HOME=/usr/local/cuda/
export PATH=${CUDA_HOME}/bin:${PATH} 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```
** the source code will be released soon **


### Microbenchmarks

We discuss three IO-aware algorithms in the paper for: 1) linear attention forward pass, 2) linear attention recurrent state update in next token prediction , and 3) sliding window next token prediction. In the Appendix, we include plots with micro benchmarking results for the kernels. The following code is used to produce these plots:


To benchmark the linear attention forward pass algorithm:
```
cd examples/linear_attention_forward/
python linear_attend_causal_reg_setup.py install
python linear_attend_profile.py
```

To benchmark the linear attention state update algorithm:
```
cd examples/based_inference/
python based_inference_setup.py install
python based_inference_profile.py
```

To benchmark the sliding window algorithm use the following commands. Note that we include comparison to flash attention. Please either install flash attention 2 (2.5.x+ is best), or modify the code according to your needs.
```
cd examples/tc_attend/
python tc_attend_setup.py install
python tc_attend_test.py
```

