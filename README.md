# Can Neural Networks Undo ln(x+y) ==> (x, y)?

This repository contains code for training neural networks to predict `x` and `y` if `ln(x+y)` is given.

The code here is almost the same as [xozero/nn-multiply](https://github.com/xozero/nn-multiply).

To run the experiments yourself, use `main.py`:

```
usage: main.py [-h] [--gendata] [--train] [-c CFG_ID] [-n [NRUNS]]

optional arguments:
  -h, --help            show this help message and exit
  --gendata
  --train
  -c CFG_ID, --cfg_id CFG_ID
                        cfg_id
  -n [NRUNS], --nruns [NRUNS]
```

To configure experiments, edit `simulations.json`. The `CFG_ID` in the above command usage is the key in the config. To find all keys in the config, run `python3 config.py`.

Example commands for generating data for configuration `7_high2_2000` and for training a model afterwards:

```
python3 main.py --gendata -c 10_nodes
python3 main.py --train -c 10_nodes
```

The generated data for all experiments configured in the checked in `simulations.json` can be found in `datasets`, it is all checked in. Thus, the `--gendata` step can be omitted for those.

The results are stored in the `runs` directory and the some plots are generated inside the `plots` directory.

A blog post with details can be found [here](#).
