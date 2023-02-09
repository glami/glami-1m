import os
from glob import glob

import pandas as pd
from tqdm import tqdm
from load_dataset import (
    EXTRACT_DIR,
    download_dataset,
    DATASET_URL,
    get_dataframe,
    COL_NAME_ITEM_ID,
    COL_NAME_NAME,
    COL_NAME_DESCRIPTION,
    CLIP_VISUAL_EMBS_DIR, COL_NAME_IMAGE_FILE, DATASET_DIR,
)
from utils import chunker
import numpy as np
import torch
# import open_clip
import clip
from PIL import Image


def generate_image_embeddings(df: pd.DataFrame, all_into_memory_only: bool = False) -> np.ndarray:
    if not os.path.exists(CLIP_VISUAL_EMBS_DIR):
        os.mkdir(CLIP_VISUAL_EMBS_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # checkpoint = 'xlm-roberta-base-ViT-B-32'
    # model, _, preprocess = open_clip.create_model_and_transforms(checkpoint, pretrained='laion5b_s13b_b90k')

    def clip_encode(image_paths: str, ):
        images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            image = preprocess(image).unsqueeze(0).to(device)
            images.append(image)

        with torch.no_grad():
            image_features = model.encode_image(torch.concat(images, dim=0))
            return image_features.cpu().detach().numpy()

    if not all_into_memory_only:
        existing_embs = [os.path.basename(x) for x in glob(f"{CLIP_VISUAL_EMBS_DIR}/*")]
        file_filter = df[COL_NAME_ITEM_ID].apply(lambda x: f"{x}.npy")
        file_filter = ~file_filter.isin(existing_embs)
        df = df.loc[file_filter]

    all_embs = None
    df[COL_NAME_NAME] = df[COL_NAME_NAME].astype(str)
    df[COL_NAME_DESCRIPTION] = df[COL_NAME_DESCRIPTION].astype(str)
    BATCH_SIZE = 750
    for batch in tqdm(
            chunker(df.loc[:, [COL_NAME_ITEM_ID, COL_NAME_NAME, COL_NAME_DESCRIPTION, COL_NAME_IMAGE_FILE]],
                    BATCH_SIZE),
            total=int(np.ceil(len(df) / BATCH_SIZE)),
    ):
        image_files = batch[COL_NAME_IMAGE_FILE].values
        embs = clip_encode(image_files)
        if all_embs is None:
            all_embs = np.array(embs)

        else:
            all_embs = np.concatenate([all_embs, embs])

        if not all_into_memory_only:
            files = [f"{CLIP_VISUAL_EMBS_DIR}/{x[0]}.npy" for x in batch.values]
            for i in range(len(image_files)):
                np.save(files[i], embs[i])

    return all_embs


if __name__ == "__main__":
    download_dataset()
    train_df = get_dataframe("train")
    test_df = get_dataframe("test")
    for df, df_name in zip([train_df, test_df], ["train", "test"]):
        generate_image_embeddings(df, all_into_memory_only=False)
