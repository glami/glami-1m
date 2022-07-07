import os
import zipfile
from urllib.request import urlretrieve, urlopen


DATASET_URL = os.environ['DATASET_URL']
EXTRACT_DIR = os.environ.get('EXTRACT_DIR', '/tmp/glami-fashion-2022')


def download_dataset(extract_dir, dataset_url):
    """
    WARNING extraction require double size of the dataset ~ 20GB
    Download unzip ideally while streaming, since the size is the same, on disk to a tmp folder or other folder selected.
    Open the test set, create image_path column with a file path.
    """

    if not os.path.exists(extract_dir):
        # hub.http_get()
        zip_path, _ = urlretrieve(dataset_url)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(extract_dir)

        os.remove(zip_path)

    else:
        print('Extract dir already exists. Delete it to re-download.')


download_dataset(EXTRACT_DIR, DATASET_URL)