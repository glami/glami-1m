import os
from glob import glob

import torch
import transformers
from multilingual_clip import pt_multilingual_clip
from multilingual_clip.pt_multilingual_clip import MultilingualCLIP
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast

from load_dataset import (
    EXTRACT_DIR,
    download_dataset,
    DATASET_URL,
    get_dataframe,
    COL_NAME_ITEM_ID,
    COL_NAME_NAME,
    COL_NAME_DESCRIPTION,
    COL_NAME_IMAGE_FILE, CLIP_TEXTUAL_EMBS_DIR,
)
from utils import chunker
import numpy as np
import pandas as pd


MAX_LENGTH = 32


if __name__ == "__main__":
    download_dataset(EXTRACT_DIR, DATASET_URL)
    dataset_dir = EXTRACT_DIR + "/glami-2022-dataset"
    train_df = get_dataframe(dataset_dir, "train")
    test_df = get_dataframe(dataset_dir, "test")

    if not os.path.exists(CLIP_TEXTUAL_EMBS_DIR):
        os.mkdir(CLIP_TEXTUAL_EMBS_DIR)

    def load_model_and_tokenizer():
        # Load Model & Tokenizer
        model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
        model: MultilingualCLIP = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
        model = model.cuda()
        model.eval()
        tokenizer: XLMRobertaTokenizerFast = transformers.AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    model, tokenizer = load_model_and_tokenizer()

    def encode(texts: list[str]):
        with torch.no_grad():
            txt_tok = tokenizer.batch_encode_plus(texts, padding="max_length", return_tensors='pt', max_length=MAX_LENGTH, truncation=True)
            input_ids = txt_tok.input_ids.cuda()
            attn_mask = txt_tok.attention_mask.cuda()
            embs = model.transformer(input_ids=input_ids, attention_mask=attn_mask)[0]
            embs = (embs * attn_mask.unsqueeze(2)).sum(dim=1) / attn_mask.sum(dim=1)[:, None]
            # TODO We could store in the inner representation without the linear projection and feed it into a transformer. For now throwing away the embeddings.
            return model.LinearTransformation(embs).cpu().detach().numpy()

    for df, df_name in zip([train_df, test_df], ["train", "test"]):   # type: pd.DataFrame, str
        file_filter = df[COL_NAME_ITEM_ID].apply(lambda x: f"{x}.npy")
        existing_embs = [os.path.basename(x) for x in glob(f"{CLIP_TEXTUAL_EMBS_DIR}/*")]
        file_filter = ~file_filter.isin(existing_embs)
        df = df.loc[file_filter]
        df[COL_NAME_NAME] = df[COL_NAME_NAME].astype(str)
        # TODO .fillna('') !
        df[COL_NAME_DESCRIPTION] = df[COL_NAME_DESCRIPTION].astype(str)

        BATCH_SIZE = 750  #     1%|▏         | 17/1324 [01:32<1:58:52,  5.46s/it]
        # BATCH_SIZE = 64  #
        for batch in tqdm(
            chunker(df.loc[:, [COL_NAME_ITEM_ID, COL_NAME_NAME, COL_NAME_DESCRIPTION, COL_NAME_IMAGE_FILE]], BATCH_SIZE),
            total=int(np.ceil(len(df) / BATCH_SIZE)),
        ):
            # For proper comparison a space is used for separation, but potentially other separator would be better.
            texts = (batch[COL_NAME_NAME] + ' ' + batch[COL_NAME_DESCRIPTION]).tolist()
            embs = encode(texts)
            files = [f"{CLIP_TEXTUAL_EMBS_DIR}/{x[0]}.npy" for x in batch.values]
            for i in range(len(batch)):
                np.save(files[i], embs[i])
