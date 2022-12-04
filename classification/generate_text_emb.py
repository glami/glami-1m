from transformers import T5Tokenizer, MT5EncoderModel
import torch
import os
from glob import glob
from tqdm import tqdm
from load_dataset import (
    EXTRACT_DIR,
    download_dataset,
    DATASET_URL,
    get_dataframe,
    COL_NAME_ITEM_ID,
    COL_NAME_NAME,
    COL_NAME_DESCRIPTION,
    EMBS_DIR, DATASET_DIR,
)
from utils import chunker
import numpy as np


def t5_encode_text(tokenizer, model, texts, MAX_LENGTH=32):
    encoded = tokenizer.batch_encode_plus(
        texts, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True
    )

    input_ids = encoded.input_ids.cuda()
    attn_mask = encoded.attention_mask.cuda()

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state

    return encoded_text.cpu().detach().numpy(), attn_mask.float().bool().cpu().detach().numpy()


if __name__ == "__main__":
    download_dataset()
    train_df = get_dataframe("train")
    test_df = get_dataframe("test")

    if not os.path.exists(EMBS_DIR):
        os.mkdir(EMBS_DIR)

    model = MT5EncoderModel.from_pretrained("google/mt5-small")
    model = model.cuda()
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

    for df, df_name in zip([train_df, test_df], ["train", "test"]):
        file_filter = df[COL_NAME_ITEM_ID].apply(lambda x: f"{x}.npy")
        existing_embs = [os.path.basename(x) for x in glob(f"{EMBS_DIR}/*")]
        file_filter = ~file_filter.isin(existing_embs)
        df = df.loc[file_filter]
        df[COL_NAME_NAME] = df[COL_NAME_NAME].astype(str)
        df[COL_NAME_DESCRIPTION] = df[COL_NAME_DESCRIPTION].astype(str)

        BATCH_SIZE = 750
        for batch in tqdm(
            chunker(df.loc[:, [COL_NAME_ITEM_ID, COL_NAME_NAME, COL_NAME_DESCRIPTION]], BATCH_SIZE),
            total=int(np.ceil(len(df) / BATCH_SIZE)),
        ):
            texts = [" ".join(x[1:]) for x in batch.values]
            files = [f"{EMBS_DIR}/{x[0]}.npy" for x in batch.values]
            files_masks = [f"{EMBS_DIR}/{x[0]}_mask.npy" for x in batch.values]
            embs, masks = t5_encode_text(tokenizer, model, texts)

            for i in range(len(texts)):
                np.save(files[i], embs[i])
                np.save(files_masks[i], masks[i])
