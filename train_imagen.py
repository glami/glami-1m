from torch.utils.data import DataLoader
from imagen_utils import ImagenDataset, SmallImagen, save_imagen
from load_dataset import download_dataset, EXTRACT_DIR, DATASET_URL, get_dataframe, MODEL_DIR
from tqdm import tqdm
import numpy as np
from imagen_pytorch import ImagenTrainer
import torch


if __name__ == "__main__":
    download_dataset(EXTRACT_DIR, DATASET_URL)
    dataset_dir = EXTRACT_DIR + "/glami-2022-dataset"
    df = get_dataframe(dataset_dir, "train")

    ds = ImagenDataset(df, transform=None)
    BATCH_SIZE = 4
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    imagen = SmallImagen()
    imagen.to(torch.device("cuda"))
    trainer = ImagenTrainer(imagen)

    # feed images into imagen, training each unet in the cascade
    epochs = 2
    SAVE_EVERY = 7500
    for i_epoch in range(epochs):
        losses = []
        for i_batch, sample_batched in tqdm(
            enumerate(dataloader), desc=f"Epoch {i_epoch}:", total=int(np.ceil(len(df) / BATCH_SIZE))
        ):
            for i in range(imagen.n_unets):
                loss = trainer(
                    sample_batched["image"].cuda(),
                    text_embeds=sample_batched["emb"].cuda(),
                    text_masks=sample_batched["mask"].cuda(),
                    unet_number=i + 1,
                    max_batch_size=8,
                )

                trainer.update(unet_number=i + 1)

            losses.append(loss)
            if (i_batch + 1) % SAVE_EVERY == 0:
                print(f"Loss: {np.mean(losses)}")
                save_imagen(np.mean(losses), i_epoch, imagen.state_dict(), trainer.state_dict(), i_batch)

        print(f"Loss: {np.mean(losses)}")
        save_imagen(np.mean(losses), i_epoch, imagen.state_dict(), trainer.state_dict())
