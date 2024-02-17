
# Benchmarking

Updates Log:
- Release Feb 2024: The kernels in the paper / this initial release are used for benchmarking and were not used in training thus far. Training kernels will be in the next release.


### Setup

Source code containing helper functions and data types for the kernels are in ```src/```. To setup the environment correctly, please run the commands provided in ```env.src```.

Next, you will need to use gcc +20 to compile the kernels. To set this up:


### Microbenchmarks

We discuss three IO-aware algorithms in the paper for: 1) linear attention forward pass, 2) linear attention recurrent state update in next token prediction , and 3) sliding window next token prediction.

To benchmark the linear attention forward pass algorithm:
```
```

To benchmark the linear attention state update algorithm:
```
```

To benchmark the sliding window algorithm use the following commands. Note that we include comparison to flash attention. Please either install flash attention 2 (2.5.x+ is best), or modify the code according to your needs.
```
cd examples/
python tc_attend_setup.py install
python tc_attend_test.py
```


