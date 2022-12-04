import os
from typing import Optional

import clip
import pandas as pd

from load_dataset import (
    download_dataset,
    EXTRACT_DIR,
    DATASET_URL,
    get_dataframe,
    COL_NAME_CATEGORY,
    COL_NAME_LABEL_SOURCE, COL_NAME_CAT_NAME, CLIP_VISUAL_EMBS_DIR, COL_NAME_ITEM_ID, COL_NAME_NAME,
    COL_NAME_DESCRIPTION, COL_NAME_IMAGE_FILE, CLIP_TEXTUAL_EMBS_DIR, CLIP_EN_TEXTUAL_EMBS_DIR, DATASET_DIR,
)
from utils import calc_accuracy, chunker, load_batch_embeddings, normalize
import torch
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    download_dataset()
    df = get_dataframe("test")
    print('Test dataset size:', len(df))

    device = os.environ.get('DEVICE', "cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)

    print("Generating target embeddings from prompts...")
    category_name_to_prompt = dict()
    category_name_to_idx = dict()
    category_idx_to_name = dict()
    cat_embeddings = []
    for cat_i, cat_name in enumerate(df[COL_NAME_CAT_NAME].unique()):
        human_readable_category_name = (cat_name.strip()
                                        .replace('women-s', "women's").replace('womens', "women's")
                                        .replace('men-s', "men's").replace('mens', "men's").replace('-', ' ')
                                        .replace(' and ', ' or '))
        prompt = ("A photo of a " + human_readable_category_name + ", a type of fashion product")
        # prompt = ("A photo of a fashion product from " + human_readable_category_name + " category")
        category_name_to_prompt[cat_name] = prompt
        category_idx_to_name[cat_i] = cat_name
        category_name_to_idx[cat_name] = cat_i

    prompt_tokens = clip.tokenize(list(category_name_to_prompt.values())).to(device)
    text_features = model.encode_text(prompt_tokens)
    text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
    cat_embeddings = text_features.cpu().detach().numpy()
    print(f'{category_name_to_prompt}')

    def get_probabilities(embs_arr: np.ndarray):
        embs_arr = normalize(embs_arr)
        similarities = embs_arr @ cat_embeddings.transpose()
        exp_similarities = np.exp(similarities)
        feature_probabilities = exp_similarities / np.sum(exp_similarities, axis=-1, keepdims=True)
        return feature_probabilities


    BATCH_SIZE = 256
    print("Evaluating...")
    # features_dirs = [CLIP_EN_TEXTUAL_EMBS_DIR]
    # features_dirs = [CLIP_VISUAL_EMBS_DIR]
    # features_dirs = [CLIP_TEXTUAL_EMBS_DIR]
    features_dirs = [CLIP_VISUAL_EMBS_DIR, CLIP_TEXTUAL_EMBS_DIR]

    print('Feature_dirs:')
    print(features_dirs)
    print()

    predictions = []
    targets = []
    # is_bagging_instead_of_feature_avg = True
    is_bagging_instead_of_feature_avg = False
    normalize_embeddings_before_sum = False
    for batch in tqdm(
            chunker(df.loc[:, [COL_NAME_ITEM_ID, COL_NAME_NAME, COL_NAME_DESCRIPTION, COL_NAME_IMAGE_FILE, COL_NAME_CAT_NAME]],
                    BATCH_SIZE),
            total=int(np.ceil(len(df) / BATCH_SIZE)),
    ):
        if is_bagging_instead_of_feature_avg:
            # Average probabilities
            probabilities: Optional[np.ndarray] = None
            for feature_emb_dir in features_dirs:
                embs_arr = load_batch_embeddings(feature_emb_dir, batch)
                # np.fromiter((np.load(file) for file in emb_files), dtype=float, count=len(emb_files))
                feature_probabilities = get_probabilities(embs_arr)
                if probabilities is None:
                    probabilities = feature_probabilities

                else:
                    # How to ensemble the models.
                    probabilities = probabilities + feature_probabilities  # bagging method
                    # probabilities = np.sqrt(probabilities * feature_probabilities)

            # probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)  # not really needed

        else:
            # Average embeddings
            embs_arr: Optional[np.ndarray] = None
            for feature_emb_dir in features_dirs:
                feature_embs_arr = load_batch_embeddings(feature_emb_dir, batch)
                if normalize_embeddings_before_sum:
                    feature_embs_arr = normalize(feature_embs_arr)

                if embs_arr is None:
                    embs_arr = feature_embs_arr

                else:
                    embs_arr += feature_embs_arr

            probabilities = get_probabilities(embs_arr)

        predictions.extend(probabilities)
        target_sr = batch[COL_NAME_CAT_NAME].map(category_name_to_idx)
        targets.extend([np.eye(1, len(category_name_to_idx), i, dtype=int)[0].tolist() for i in target_sr])

    accs = calc_accuracy(np.array(predictions), np.array(targets))
    print()
    print(f"Accuracies:")
    print(accs)

# Average visual and textual embeddings (ensamble) and CLIP type prompt: {1: 0.3232991965794283, 5: 0.7450087928002482}
# Visual features and CLIP type prompt: {1: 0.28931760973759524, 5: 0.7179321402710251}
# Textual features and CLIP type prompt: {1: 0.2654218820040688, 5: 0.5850574118133858}