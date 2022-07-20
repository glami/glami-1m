import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from ignite.metrics import Bleu
from tabulate import tabulate
from ignite.exceptions import NotComputableError

from load_dataset import (
    download_dataset,
    EXTRACT_DIR,
    DATASET_URL,
    get_dataframe,
    COL_NAME_GEO,
    COL_NAME_IMAGE_ID,
    COL_NAME_DESCRIPTION,
)
from utils import chunker

if __name__ == "__main__":
    geo_to_m2mlang = {
        "cz": "cs",
        "sk": "sk",
        "ro": "ro",
        "gr": "el",
        "hu": "hu",
        "bg": "bg",
        "hr": "hr",
        "es": "et",
        "lt": "lt",
        "si": "sl",
        "lv": "lv",
        "tr": "tr",
        "ee": "et",
    }

    geos = geo_to_m2mlang.keys()

    download_dataset(EXTRACT_DIR, DATASET_URL)
    dataset_dir = EXTRACT_DIR + "/glami-2022-dataset/"
    train_df = get_dataframe(dataset_dir, "train")
    test_df = get_dataframe(dataset_dir, "test")

    tcols = [COL_NAME_GEO, COL_NAME_IMAGE_ID, COL_NAME_DESCRIPTION]

    for df, df_name in zip([train_df, test_df], ["train", "test"]):
        counts = np.zeros((len(geos), len(geos)), dtype=int)
        for i, left_geo in enumerate(geos):
            for j, right_geo in enumerate(geos):
                if i >= j:
                    continue

                left_df = df.loc[df["geo"] == left_geo, tcols]
                right_df = df.loc[df["geo"] == right_geo, tcols]

                inner_df = pd.merge(
                    left_df, right_df, on=COL_NAME_IMAGE_ID, how="inner", suffixes=("_left", "_right"), sort=True
                )

                if len(inner_df) == 0:
                    continue

                counts[i, j] = len(inner_df)
                counts[j, i] = len(inner_df)

        print(f"Counts ({df_name}):")
        print(tabulate(counts, showindex=geos, headers=geos, tablefmt="latex"))

    bleus = np.full((len(geos), len(geos)), np.nan, dtype=float)
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    model.cuda()
    model.eval()
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    BATCH_SIZE = 128
    with torch.no_grad():
        for i, left_geo in enumerate(geos):
            for j, right_geo in enumerate(geos):
                if i >= j:
                    continue

                left_df = df.loc[df[COL_NAME_GEO] == left_geo, tcols]
                left_df[COL_NAME_DESCRIPTION] = left_df[COL_NAME_DESCRIPTION].astype(str)
                right_df = df.loc[df[COL_NAME_GEO] == right_geo, tcols]
                right_df[COL_NAME_DESCRIPTION] = right_df[COL_NAME_DESCRIPTION].astype(str)

                inner_df = pd.merge(
                    left_df, right_df, on=COL_NAME_IMAGE_ID, how="inner", suffixes=("_left", "_right"), sort=True
                )

                left_lang = geo_to_m2mlang[left_geo]
                right_lang = geo_to_m2mlang[right_geo]

                top_bleu = Bleu()
                bottom_bleu = Bleu()
                for left_chunk, right_chunk in tqdm(
                    zip(
                        chunker(inner_df.loc[:, f"{COL_NAME_DESCRIPTION}_left"], BATCH_SIZE),
                        chunker(inner_df.loc[:, f"{COL_NAME_DESCRIPTION}_right"], BATCH_SIZE),
                    ),
                    total=int(np.ceil(len(inner_df) / BATCH_SIZE)),
                    desc=f"i: {i}/{len(geos)}; j: {j - i - 1}/{len(geos)}",
                ):
                    left_tokens = tokenizer.batch_encode_plus(
                        left_chunk.tolist(), return_tensors="pt", padding="max_length", truncation=True, max_length=32
                    ).to(torch.device("cuda"))
                    right_tokens = tokenizer.batch_encode_plus(
                        right_chunk.tolist(), return_tensors="pt", padding="max_length", truncation=True, max_length=32
                    ).to(torch.device("cuda"))

                    left_generated = model.generate(
                        **left_tokens, forced_bos_token_id=tokenizer.get_lang_id(right_lang), max_length=32
                    )
                    left_generated = tokenizer.batch_decode(left_generated, skip_special_tokens=True)

                    right_generated = model.generate(
                        **right_tokens, forced_bos_token_id=tokenizer.get_lang_id(left_lang), max_length=32
                    )
                    right_generated = tokenizer.batch_decode(right_generated, skip_special_tokens=True)

                    top_bleu.update(([x.split() for x in left_generated], [[x.split()] for x in right_chunk.tolist()]))
                    bottom_bleu.update(
                        ([x.split() for x in right_generated], [[x.split()] for x in left_chunk.tolist()])
                    )

                try:
                    bleus[i, j] = top_bleu.compute().item()
                    bleus[j, i] = bottom_bleu.compute().item()
                except NotComputableError:
                    pass

    print("Mean Bleu:")
    print(np.nansum(np.array(counts) * np.array(bleus)) / np.nansum(counts))

    print("Bleus:")
    print(tabulate(bleus, showindex=geos, headers=geos, tablefmt="latex"))
