# How to Run

## GPU memory needed

All the experiments is able to run on a single graphic card. However, **if you want to get results on ImageNet, the memory on any single graphic card should be larger than 24 GB.** Around 12 GB is enough for other datasets. 


## How to Install
This code is built on top of the toolbox [Dassl.ProGrad.pytorch](https://github.com/BeierZhu/Prompt-align/tree/main/Dassl.ProGrad.pytorch). You can prepare the environment as follows:

```
# Create a conda environment
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
# Please make sure you have installed the gpu version due to the speed.
# For example:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, run `pip install -r requirements.txt` under `KgCoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.


## Generalization From Base to New Classes

You will need `base2new_train_main.sh`, `base2new_test_main.sh`, and `run.sh`. The scripts with the prefix `base2new_train` train a model on base classes while the ones with the prefix `base2new_test` evaluate the trained model on new classes. Both kinds of scripts have only one input argument, i.e., `DATASET`. `DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `CoOp/configs/datasets/`.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
bash base2new_train.sh stanford_cars 8.0
bash base2new_test.sh stanford_cars 8.0
```

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results. For instance, after you finish the evaluation using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– stanford_cars/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– rn50_ep100/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– stanford_cars/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– rn50_ep100/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/stanford_cars/shots_16/CoCoOp/rn50_ep100
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/stanford_cars/shots_16/CoCoOp/rn50_ep100 --test-log
```

