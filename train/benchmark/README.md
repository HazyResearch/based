
Code to end-to-end benchmark the prefill and generation speeds of models. 

```bash
python benchmark/launch.py benchmark/configs/01-29-forward-1b.py --gpus=4,5,6,7
```

To run BASED with TK mode:
```bash
git clone git@github.com:HazyResearch/ThunderKittens.git
# ensure that "based" is selected in ThunderKittens/config.py
cd ThunderKittens
python setup.py install 
```

