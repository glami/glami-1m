# GLAMI-1M: A Multilingual Image-Text Fashion Dataset

We introduce GLAMI-1M: the largest multilingual image-text classification dataset and benchmark. The dataset contains images of fashion products with item descriptions, each in 1 of 13 languages. Categorization into 191 classes has high-quality annotations: all 100k images in the test set and 75% of the 1M training set were human-labeled. The paper presents baselines for image-text classification showing that the dataset presents a challenging fine-grained classification problem: The best scoring EmbraceNet model using both visual and textual features achieves 69.7% accuracy. Experiments with a modified Imagen model show the dataset is also suitable for image generation conditioned on text.

![GLAMI-1M Dataset Examples](media/glami-1m-dataset-examples.png)


## GLAMI-1M Paper

The [paper](https://bmvc2022.mpi-inf.mpg.de/0607.pdf) ([supplementary]()) was accepted to [BMVC 2022](https://bmvc2022.mpi-inf.mpg.de/607/).
The paper will be accessible soon on Arxiv as well.


## How to Download GLAMI-1M Manually
You can either manually download the dataset zip file yourself or use the repository scripts to download, extract, and load into a dataframe.

To manually download the dataset ZIP file(s):
- [GLAMI-1M Dataset (the default 228x298 version) here (1 zip  file of 11GB)](https://zenodo.org/record/7326406/files/GLAMI-1M-dataset.zip?download=1)
- [GLAMI-1M Dataset 800x800 version here (11 zip files of 54GB)](https://zenodo.org/record/7338792)


## GLAMI-1M Code
Together with the dataset with provide helper code and experiments described in the paper.
The model weights will be uploaded soon.


### Installation

```
conda create -n g1m python=3.9
conda activate g1m
pip install -r requirements.txt
```

### How to Download GLAMI-1M Programmatically

The dataset can be downloaded programmatically with the default 228x298 version.
The destination directory to download and extract the dataset can be configured with environmental variable EXTRACT_DIR.
EXTRACT_DIR is by default configured as a Linux temporary directory, which is removed upon machine restarts.
Before downloading the dataset, make sure you have enough space to download and unzip corresponding version of the dataset.


```bash
EXTRACT_DIR="/tmp/GLAMI-1M/" python -c 'import load_dataset; load_dataset.download_dataset())'
```

After the download, we can load the dataset into a dataframe.

```bash
EXTRACT_DIR="/tmp/GLAMI-1M/" python -c 'import load_dataset; print(load_dataset.get_dataframe("test").head())'
```

### Baseline Weights

Weights for the baseline models described in the paper are available [here](https://zenodo.org/record/7353512).


# Examples

![GLAMI-1M Dataset Examples Table](media/glami-1m-dataset-examples-table.png)
