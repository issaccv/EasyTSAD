# Code for ICML 2025

**This project use python 3.10.**

Download dataset first:

```sh
git clone https://github.com/CSTCloudOps/datasets
```


# Option 1: Use conda and conda-forge channel

For how to configure conda-forge channel, please refer to [this link](https://conda-forge.org/docs/user/introduction/#how-can-i-install-packages-from-conda-forge).


Create environment:

```sh
conda env create --name tsm -f environment.yaml
conda activate tsm
```

Reproduce the results:

```sh
python run.py
```

# Option2: Use pip
Create environment:

```sh
python -m venv tsm
source tsm/bin/activate
pip install -r requirements.txt
```
Reproduce the results:

```sh
python run.py
```

# Option 3: Use docker
```sh
docker build -t tsm:v1 .
docker run -it --rm \
    --gpus all \
    -v $(pwd):/workspace \
    tsm:v1
```
