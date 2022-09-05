import clip

from load_dataset import (
    download_dataset,
    EXTRACT_DIR,
    DATASET_URL,
    get_dataframe,
    COL_NAME_CATEGORY,
    COL_NAME_LABEL_SOURCE, COL_NAME_CAT_NAME, CLIP_VISUAL_EMBS_DIR, COL_NAME_ITEM_ID, COL_NAME_NAME,
    COL_NAME_DESCRIPTION, COL_NAME_IMAGE_FILE,
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
        # TODO men's!
        # TODO prompt = "a photo of a " + cat_name.strip().replace('women-s', "women's").replace('womens', "women's").replace('-', ' ').replace(' and ', ' or ') + ", a type of fashion product"
        prompt = "a photo of a fashion product from " + cat_name.strip().replace('-', ' ').replace('womens', "women's").rstrip( 's') + " category" # TODO women-s
        # TODO consider multi-lingual for text eval although it should not matter in english
        prompt_tokens = clip.tokenize([prompt]).to(device)
        text_features = model.encode_text(prompt_tokens).cpu().detach().numpy()
        category_name_to_prompt[cat_name] = prompt
        category_idx_to_name[cat_i] = cat_name
        category_name_to_idx[cat_name] = cat_i
        text_features = text_features / np.linalg.norm(text_features)
        cat_embeddings.append(text_features)

    cat_embeddings = np.stack(cat_embeddings, axis=0).squeeze()
    print(f'{category_name_to_prompt}')

    BATCH_SIZE = 256
    print("Evaluating...")
    predictions = []
    targets = []
    for batch in tqdm(
            chunker(df.loc[:, [COL_NAME_ITEM_ID, COL_NAME_NAME, COL_NAME_DESCRIPTION, COL_NAME_IMAGE_FILE, COL_NAME_CAT_NAME]],
                    BATCH_SIZE),
            total=int(np.ceil(len(df) / BATCH_SIZE)),
    ):
        emb_files = [f"{CLIP_VISUAL_EMBS_DIR}/{x[0]}.npy" for x in batch.values]
        embs_arr = [np.load(file) for file in emb_files]
        embs_arr = [x.squeeze() / np.linalg.norm(x) for x in embs_arr]
        embs = np.stack(embs_arr, axis=0)
        similarities = embs @ cat_embeddings.transpose()
        predictions.extend(similarities)

        target_sr = batch[COL_NAME_CAT_NAME].map(category_name_to_idx)
        targets.extend([np.eye(1, len(category_name_to_idx), i, dtype=int)[0].tolist() for i in target_sr])

    accs = calc_accuracy(np.array(predictions), np.array(targets))
    print(f"Accuracies:")
    print(accs)
