import os.path

import pandas as pd

from load_dataset import get_dataframe, DATASET_DIR, COL_NAME_IMAGE_FILE, download_dataset, EXTRACT_DIR, DATASET_URL

download_dataset()

train_df = get_dataframe("train")
test_df = get_dataframe("test")
df = pd.concat([train_df, test_df])

for file in df[COL_NAME_IMAGE_FILE]:
    assert os.path.exists(file)

print()