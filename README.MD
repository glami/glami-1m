# GLAMI-1M Dataset

Before using the dataset, make sure you have enough space to download and unzip corresponding version of the dataset.
You can either manually download the dataset zip file yourself or use the repository scripts to download, extract, and load into a dataframe.

To manually download the dataset ZIP file:
- [GLAMI-1M dataset 228x298 version here (11GB)](https://zenodo.org/record/7326406/files/GLAMI-1M-dataset.zip?download=1)
- (coming soon) 800x800 version here (62GB)


# Experiments and Helper Code
Together with the dataset with provide helper code and experiments described in the paper.

### Installation

```
conda create -n g1m python=3.9
conda activate g1m
pip install -r requirements.txt
```

### Dataset Download

The dataset can be downloaded programaticaly.
The destination directory to download and extract the dataset can be configured with environmental variable EXTRACT_DIR.
EXTRACT_DIR is by default configured as a Linux temporary directory, which is removed upon machine restarts.

```bash
EXTRACT_DIR="/tmp/GLAMI-1M/" python -c 'import load_dataset; load_dataset.download_dataset())'
```

After the download, we can load the dataset into a dataframe.

```bash
EXTRACT_DIR="/tmp/GLAMI-1M/" python -c 'import load_dataset; print(load_dataset.get_dataframe("test").head())'
```
