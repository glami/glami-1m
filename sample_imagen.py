from imagen_utils import SmallImagen, load_imagen
from load_dataset import (
    GENERATED_DIR,
    download_dataset,
    EXTRACT_DIR,
    DATASET_URL,
    get_dataframe,
    COL_NAME_NAME,
    COL_NAME_DESCRIPTION,
    EMBS_DIR,
    COL_NAME_ITEM_ID,
)
import os
from imagen_pytorch import ImagenTrainer
import torch
from tqdm import tqdm
import numpy as np

from utils import chunker

if __name__ == "__main__":
    download_dataset(EXTRACT_DIR, DATASET_URL)
    dataset_dir = EXTRACT_DIR + "/glami-2022-dataset"
    df = get_dataframe(dataset_dir, "test")
    df[COL_NAME_NAME] = df[COL_NAME_NAME].astype(str)
    df[COL_NAME_DESCRIPTION] = df[COL_NAME_DESCRIPTION].astype(str)

    if not os.path.exists(GENERATED_DIR):
        os.mkdir(GENERATED_DIR)

    print("Loading model checkpoint...")
    imagen = SmallImagen()
    imagen.to(torch.device("cuda"))

    trainer = ImagenTrainer(imagen)

    # load imagen:
    ie, sd, td, _ = load_imagen("imagen_1_final.pt")
    imagen.load_state_dict(sd)
    trainer.load_state_dict(td)
    imagen.eval()

    print("Sampling...")
    BATCH_SIZE = 4
    TOTAL_SAMPLES = 1000
    for batch in tqdm(
        chunker(
            df.loc[:, [COL_NAME_ITEM_ID, COL_NAME_NAME, COL_NAME_DESCRIPTION]].sample(min(TOTAL_SAMPLES, len(df))),
            BATCH_SIZE,
        ),
        total=int(np.ceil(min(TOTAL_SAMPLES, len(df)) / BATCH_SIZE)),
    ):
        texts = [" ".join(x[1:]) for x in batch.values]
        files = [f"{EMBS_DIR}/{x[0]}.npy" for x in batch.values]
        files_masks = [f"{EMBS_DIR}/{x[0]}_mask.npy" for x in batch.values]
        embs = np.array([np.load(x) for x in files])
        masks = np.array([np.load(x) for x in files_masks])

        images = trainer.sample(
            text_masks=masks,
            text_embeds=embs,
            return_pil_images=True,
            cond_scale=8.0,
            batch_size=BATCH_SIZE,
            return_all_unet_outputs=True,
        )

        for i, unet_outputs in enumerate(images):
            for im, te, itid in zip(unet_outputs, texts, batch.values[:, 0]):
                with open(f"{GENERATED_DIR}/{itid}.txt", "w") as fout:
                    fout.write(te)
                im.save(f"{GENERATED_DIR}/{itid}_{i}.jpg")
