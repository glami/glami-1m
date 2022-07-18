import os
import zipfile
from tempfile import TemporaryFile

from tqdm import tqdm
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
    WARNING extraction requires double size of the dataset ~ 20GB.
    Download unzip ideally while streaming, since the size is the same, on disk to a tmp folder or other folder selected.
    Open the test set, create image_path column with a file path.

    The dataset is by default stored in linux TMP folder, so it will be removed on restart.
    This at the same time prevets disk overflow.
    """

    if not os.path.exists(extract_dir):
        with TemporaryFile() as zf:
            http_get(dataset_url, zf)
            zf.seek(0)
            print('Unzipping')
            with zipfile.ZipFile(zf, "r") as f:
                members = f.namelist()
                for zipinfo in tqdm(members):
                    f._extract_member(zipinfo, extract_dir, None)

    else:
        print('Extract dir already exists. Delete it to re-download.')


def get_dataframe(extract_dir: str, split_type: str):
    assert split_type in ('train', 'test')
    df = pd.read_csv(extract_dir + f'/glami-fashion-2022-{split_type}.csv')
    df[COL_NAME_IMAGE_FILE] = extract_dir + '/images/' + df[COL_NAME_IMAGE_ID].astype(str) + '.jpg'
    assert os.path.exists(df.loc[0, COL_NAME_IMAGE_FILE])
    return df


if __name__ == "__main__":
    download_dataset(EXTRACT_DIR, DATASET_URL)
    dataset_dir = EXTRACT_DIR + '/glami-2022-dataset/'
    test_df = get_dataframe(dataset_dir, 'test')
    train_df = get_dataframe(dataset_dir, 'train')
