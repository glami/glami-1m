import os
import zipfile
from tempfile import TemporaryFile
from tqdm import tqdm
from transformers.file_utils import http_get
import pandas as pd

DATASET_URL = os.environ.get("DATASET_URL")
EXTRACT_DIR = os.environ.get("EXTRACT_DIR", "/tmp/GLAMI-1M")
DATASET_DIR = dataset_dir = EXTRACT_DIR + "/GLAMI-1M-dataset"
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/GLAMI-1M/models")
EMBS_DIR = EXTRACT_DIR + "/embs"
CLIP_VISUAL_EMBS_DIR = EXTRACT_DIR + "/embs-clip-visual"
CLIP_TEXTUAL_EMBS_DIR = EXTRACT_DIR + "/embs-clip-textual"
CLIP_EN_TEXTUAL_EMBS_DIR = EXTRACT_DIR + "/embs-clip-en-textual"
GENERATED_DIR = EXTRACT_DIR + "/generated_images"

COL_NAME_ITEM_ID = "item_id"
COL_NAME_IMAGE_ID = "image_id"
COL_NAME_IMAGE_FILE = "image_file"
COL_NAME_IMAGE_URL = "image_url"
COL_NAME_NAME = "name"
COL_NAME_DESCRIPTION = "description"
COL_NAME_GEO = "geo"
COL_NAME_CATEGORY = "category"
COL_NAME_CAT_NAME = "category_name"
COL_NAME_LABEL_SOURCE = "label_source"
COL_NAME_EMB_FILE = "emb_file"
COL_NAME_MASK_FILE = "mask_file"
DEFAULT_IMAGE_SIZE = (298, 228)


COUNTRY_CODE_TO_COUNTRY_NAME = {
    "cz": "Czechia",
    "sk": "Slovakia",
    "ro": "Romania",
    "gr": "Greece",
    "si": "Slovenia",
    "hu": "Hungary",
    "hr": "Croatia",
    "es": "Spain",
    "lt": "Lithuania",
    "lv": "Latvia",
    "tr": "Turkey",
    "ee": "Estonia",
    "bg": "Bulgaria",
}

COUNTRY_CODE_TO_COUNTRY_NAME_W_CC = {name + f' ({cc})' for cc, name in COUNTRY_CODE_TO_COUNTRY_NAME}


def download_dataset(extract_dir, dataset_url):
    """
    WARNING extraction requires double size of the dataset ~ 20GB.
    Download unzip ideally while streaming, since the size is the same, on disk to a tmp folder or other folder selected.
    Open the test set, create image_path column with a file path.

    The dataset is by default stored in linux TMP folder, so it will be removed on restart.
    This at the same time prevets disk overflow.
    """

    if not os.path.exists(extract_dir):
        assert dataset_url is not None, f"Dataset URL is required"
        with TemporaryFile() as zf:
            http_get(dataset_url, zf)
            zf.seek(0)
            print("Unzipping")
            with zipfile.ZipFile(zf, "r") as f:
                members = f.namelist()
                for zipinfo in tqdm(members):
                    f._extract_member(zipinfo, extract_dir, None)

    else:
        print("Extract dir already exists. Delete it to re-download.")


def get_dataframe(extract_dir: str, split_type: str):
    assert split_type in ("train", "test")
    df = pd.read_csv(extract_dir + f"/GLAMI-1M-{split_type}.csv")
    df[COL_NAME_IMAGE_FILE] = extract_dir + "/images/" + df[COL_NAME_IMAGE_ID].astype(str) + ".jpg"
    df[COL_NAME_DESCRIPTION] = df[COL_NAME_DESCRIPTION].fillna('')
    assert os.path.exists(df.loc[0, COL_NAME_IMAGE_FILE])
    return df


if __name__ == "__main__":
    download_dataset(EXTRACT_DIR, DATASET_URL)
    dataset_dir = EXTRACT_DIR + "/glami-2022-dataset"
    test_df = get_dataframe(dataset_dir, "test")
    train_df = get_dataframe(dataset_dir, "train")
