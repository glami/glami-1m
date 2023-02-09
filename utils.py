import base64
from typing import Optional, List

import numpy as np
import pandas as pd
from IPython.core.display import display, HTML, Image
from tqdm import tqdm

from load_dataset import CLIP_VISUAL_EMBS_DIR, COL_NAME_ITEM_ID, COL_NAME_IMAGE_FILE, COL_NAME_NAME, \
    COL_NAME_DESCRIPTION, COL_NAME_CAT_NAME


def chunker(dftochunk, size):
    n = int(np.ceil(len(dftochunk) / size))
    for i in range(n):
        lower = i * size
        upper = min((i + 1) * size, len(dftochunk))
        yield dftochunk.iloc[lower:upper]


def calc_accuracy(X: np.ndarray, Y: np.ndarray, ks=(1, 5)):
    """
    Get the accuracy with respect to the most likely label

    :param X:
    :param Y:
    :param ks:
    :return:
    """
    assert X.shape[0] == Y.shape[0]

    # find top classes
    max_idx_class = np.argsort(X, axis=1)  # [B, n_classes] -> [B, n_classes]
    max_idx_class = np.flip(max_idx_class, axis=1)  # descending

    accs = {}
    for k in ks:
        preds = np.zeros(X.shape, dtype=np.bool)
        cols_maxima = max_idx_class[:, :k]
        rows_maxima = np.array(range(cols_maxima.shape[0]))[:, np.newaxis]
        rows_maxima = np.repeat(rows_maxima, k, 1)
        preds[rows_maxima, max_idx_class[:, :k]] = True
        accs[k] = np.sum(Y[preds]) / X.shape[0]

    return accs


def load_embeddings(df: pd.DataFrame, embs_dir=CLIP_VISUAL_EMBS_DIR, vector_normalize=True, batch_size=1024) -> np.ndarray:
    arrays = []

    for batch_df in tqdm(
            chunker(df.loc[:, [COL_NAME_ITEM_ID]], batch_size),
            total=int(np.ceil(len(df) / batch_size)),
    ):
        embeddings_array = load_batch_embeddings(embs_dir, batch_df)
        if vector_normalize:
            embeddings_array = normalize(embeddings_array)

        arrays.append(embeddings_array)

    full_array = np.concatenate(arrays)
    return full_array


def load_batch_embeddings(feature_emb_dir: str, batch: pd.DataFrame):
    return np.array(list(np.load(f"{feature_emb_dir}/{item_id}.npy") for item_id in batch[COL_NAME_ITEM_ID].values))


def normalize(x: np.ndarray, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def image_formatter(img_file):
    with open(img_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
        return f'<img width="150" src="data:image/png;base64,{encoded_string}">'


def public_dataset_to_html(df: pd.DataFrame, extra_cols: Optional[List[str]]=None):
    if extra_cols is None:
        extra_cols = []

    return display(
        HTML(
            df[[COL_NAME_ITEM_ID, COL_NAME_IMAGE_FILE] + [COL_NAME_NAME, COL_NAME_DESCRIPTION, COL_NAME_CAT_NAME] + extra_cols].to_html(
                formatters={
                    COL_NAME_IMAGE_FILE: image_formatter,
                },
                escape=False,
                index=False,
            )
        )
    )
