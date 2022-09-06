from typing import Optional

import clip
import scipy

from load_dataset import (
    download_dataset,
    EXTRACT_DIR,
    DATASET_URL,
    get_dataframe,
    COL_NAME_CATEGORY,
    COL_NAME_LABEL_SOURCE, COL_NAME_CAT_NAME, CLIP_VISUAL_EMBS_DIR, COL_NAME_ITEM_ID, COL_NAME_NAME,
    COL_NAME_DESCRIPTION, COL_NAME_IMAGE_FILE, CLIP_TEXTUAL_EMBS_DIR,
)
from utils import calc_accuracy, chunker
import torch
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    download_dataset(EXTRACT_DIR, DATASET_URL)
    dataset_dir = EXTRACT_DIR + "/glami-2022-dataset"
    df = get_dataframe(dataset_dir, "test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print("Generating target embeddings from prompts...")
    category_name_to_prompt = dict()
    category_name_to_idx = dict()
    category_idx_to_name = dict()
    cat_embeddings = []
    for cat_i, cat_name in enumerate(df[COL_NAME_CAT_NAME].unique()):

        prompt = ("A photo of a " + cat_name.strip()
                  .replace('women-s', "women's").replace('womens', "women's")
                  .replace('men-s', "men's").replace('mens', "men's")
                  .replace('-', ' ').replace(' and ', ' or ') + ", a type of fashion product")
        # prompt = "A photo of a fashion product from " + cat_name.strip().replace('-', ' ').replace('womens', "women's").rstrip( 's') + " category"
        category_name_to_prompt[cat_name] = prompt
        category_idx_to_name[cat_i] = cat_name
        category_name_to_idx[cat_name] = cat_i

    prompt_tokens = clip.tokenize(list(category_name_to_prompt.values())).to(device)
    text_features = model.encode_text(prompt_tokens)
    text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
    cat_embeddings = text_features.cpu().detach().numpy()
    print(f'{category_name_to_prompt}')

    BATCH_SIZE = 512
    print("Evaluating...")
    features_dirs = [CLIP_VISUAL_EMBS_DIR, CLIP_TEXTUAL_EMBS_DIR]
    # features_dirs = [CLIP_VISUAL_EMBS_DIR]
    # features_dirs = [CLIP_TEXTUAL_EMBS_DIR]
    predictions = []
    targets = []
    for batch in tqdm(
            chunker(df.loc[:, [COL_NAME_ITEM_ID, COL_NAME_NAME, COL_NAME_DESCRIPTION, COL_NAME_IMAGE_FILE, COL_NAME_CAT_NAME]],
                    BATCH_SIZE),
            total=int(np.ceil(len(df) / BATCH_SIZE)),
    ):
        probabilities: Optional[np.ndarray] = None
        for feature_emb_dir in features_dirs:
            embs_arr = np.array(list(np.load(f"{feature_emb_dir}/{item_id}.npy") for item_id in batch[COL_NAME_ITEM_ID].values))
            # np.fromiter((np.load(file) for file in emb_files), dtype=float, count=len(emb_files))
            embs_arr = embs_arr / np.linalg.norm(embs_arr, axis=-1, keepdims=True)
            embs = np.stack(embs_arr, axis=0)
            similarities = embs @ cat_embeddings.transpose()
            feature_probabilities = np.exp(similarities) / np.sum(np.exp(similarities), axis=-1, keepdims=True)
            if probabilities is None:
                probabilities = feature_probabilities

            else:
                # How to ensemble the models.
                probabilities = probabilities + feature_probabilities  # bagging method
                # probabilities = np.sqrt(probabilities * feature_probabilities)

        probabilities = np.sum(probabilities, axis=-1, keepdims=True)  # not really needed
        predictions.extend(probabilities)

        target_sr = batch[COL_NAME_CAT_NAME].map(category_name_to_idx)
        targets.extend([np.eye(1, len(category_name_to_idx), i, dtype=int)[0].tolist() for i in target_sr])

    accs = calc_accuracy(np.array(predictions), np.array(targets))
    print(f"Accuracies:")
    print(accs)

# Visual features with CLIP type prompt: {1: 0.29138640811643757, 5: 0.7185566778395159}
# Textual features with CLIP type prompt: {1: 0.2608522859349575, 5: 0.5662085048628139}
# Visual+textual features with CLIP type prompt: {1: 0.30645850673745306, 5: 0.6880600484406751}
# Visual*textual features with CLIP type prompt: {1: 0.30702089847704317, 5: 0.6903546067382029}
