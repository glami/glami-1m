import os
import zipfile
from tempfile import TemporaryFile

from transformers.file_utils import http_get

import pandas as pd

DATASET_URL = os.environ['DATASET_URL']
EXTRACT_DIR = os.environ.get('EXTRACT_DIR', '/tmp/glami-fashion-2022')

COL_NAME_ITEM_ID = "item_id"
COL_NAME_IMAGE_ID = "image_id"
COL_NAME_IMAGE_FILE = "image_file"
COL_NAME_NAME = "name"
COL_NAME_DESCRIPTION = "description"
COL_NAME_GEO = "geo"
COL_NAME_CATEGORY = "category"
COL_NAME_CAT_NAME = "category_name"
COL_NAME_LABEL_SOURCE = "label_source"



def download_dataset(extract_dir, dataset_url):
    """
    WARNING extraction require double size of the dataset ~ 20GB
    Download unzip ideally while streaming, since the size is the same, on disk to a tmp folder or other folder selected.
    Open the test set, create image_path column with a file path.
    """

    if not os.path.exists(extract_dir):
        with TemporaryFile('wb') as zf:
            http_get(dataset_url, zf)
            with zipfile.ZipFile(zf, "r") as f:
                f.extractall(extract_dir)

    else:
        print('Extract dir already exists. Delete it to re-download.')


def get_dataframe(extract_dir: str, split_type: str):
    assert split_type in ('train', 'test')
    df = pd.read_csv(extract_dir + f'/glami-fashion-2022-{split_type}.csv')
    df[COL_NAME_IMAGE_FILE] = extract_dir + '/images/' + df[COL_NAME_IMAGE_ID].astype(str) + '.jpg'
    assert os.path.exists(test_df.loc[0, COL_NAME_IMAGE_FILE])
    return df


download_dataset(EXTRACT_DIR, DATASET_URL)
test_df = get_dataframe(EXTRACT_DIR, 'test')
train_df = get_dataframe(EXTRACT_DIR, 'train')
