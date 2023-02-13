import os
import zipfile
from datetime import datetime

from tqdm import tqdm

from load_dataset import EXTRACT_DIR, COL_NAME_IMAGE_ID, get_dataframe

if __name__ == "__main__":
    start_time = datetime.now()

    os.chdir(EXTRACT_DIR)
    pub_test_df = get_dataframe('test')

    zip_file_name = f"GLAMI-1M-dataset--test-only.zip"
    print(f"Zipping to {zip_file_name} images from test set")
    with zipfile.ZipFile(zip_file_name, "w", compression=0) as zip_file:
        zip_file.write("GLAMI-1M-dataset/GLAMI-1M-test.csv")
        zip_file.write("GLAMI-1M-dataset/LICENSE")
        zip_file.write("GLAMI-1M-dataset/NOTICE")
        zip_file.write("GLAMI-1M-dataset/README.txt")

        for i in tqdm(pub_test_df[COL_NAME_IMAGE_ID]):
            image_file = f"GLAMI-1M-dataset/images/{i}.jpg"
            if os.path.exists(image_file):
                zip_file.write(image_file)

    print(f"Zipping took: {datetime.now() - start_time}")
