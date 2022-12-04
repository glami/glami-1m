# minor hack to allow mt5-small config for imagen
from transformers import MT5Config
from imagen_pytorch import t5

t5.T5_CONFIGS = {"mt5-mini": {"config": MT5Config.from_pretrained("google/mt5-small")}}

import torch
from torch.utils.data import Dataset
from imagen_pytorch import Unet, Imagen
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb, rgba2rgb
import os
from load_dataset import (
    COL_NAME_EMB_FILE,
    COL_NAME_MASK_FILE,
    COL_NAME_ITEM_ID,
    EMBS_DIR,
    COL_NAME_IMAGE_FILE,
    MODEL_DIR,
)

GEN_IM_SIZE = 128


class SmallImagen(Imagen):
    def __init__(self):
        self.n_unets = 2
        unet1 = Unet(dim=128, cond_dim=512, dim_mults=(1, 2, 4), num_resnet_blocks=3, layer_attns=(False, True, True),)
        unet2 = Unet(
            dim=128,
            cond_dim=512,
            dim_mults=(1, 2, 4),
            num_resnet_blocks=(2, 4, 8),
            layer_attns=(False, False, True),
            layer_cross_attns=(False, False, True),
        )

        super(SmallImagen, self).__init__(
            unets=(unet1, unet2),
            text_encoder_name="mt5-mini",
            image_sizes=(64, GEN_IM_SIZE),
            timesteps=500,
            cond_drop_prob=0.1,
            text_embed_dim=512,
        )


class ImagenDataset(Dataset):
    """GLAMI Dataset for Imagen."""

    def __init__(self, dtfrm, transform=None):
        """
        Args:
            dtfrm (pd.DataFrame):
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = dtfrm
        self.df[COL_NAME_EMB_FILE] = self.df[COL_NAME_ITEM_ID].map(lambda x: f"{EMBS_DIR}/{x}.npy")
        self.df[COL_NAME_MASK_FILE] = self.df[COL_NAME_ITEM_ID].map(lambda x: f"{EMBS_DIR}/{x}_mask.npy")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        all_tensors = self.df[[COL_NAME_IMAGE_FILE, COL_NAME_EMB_FILE, COL_NAME_MASK_FILE]].iloc[idx].values
        image = np.ones(shape=(GEN_IM_SIZE, GEN_IM_SIZE, 3), dtype=np.float32)
        loaded_image = io.imread(all_tensors[0])
        loaded_image = resize(
            loaded_image,
            output_shape=(GEN_IM_SIZE, int(228 * GEN_IM_SIZE / 298)),
            preserve_range=False,  # normalize to 0-1
        ).astype(np.float32)
        if len(loaded_image.shape) == 2:
            loaded_image = gray2rgb(loaded_image)
        elif loaded_image.shape[2] == 4:
            loaded_image = rgba2rgb(loaded_image)
        left = int((GEN_IM_SIZE - loaded_image.shape[1]) / 2)
        image[:, left : left + loaded_image.shape[1]] = loaded_image
        image = np.moveaxis(image, -1, 0)
        emb = np.load(all_tensors[1])
        mask = np.load(all_tensors[2])

        sample = {"image": image, "emb": emb, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


def save_imagen(lo, ie, sd, td, ib="final", base_name="imagen"):
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    print(f"Loss: {lo}")
    torch.save(
        {"epoch": ie, "model_state_dict": sd, "trainer_state_dict": td, "loss": lo,},
        f"{MODEL_DIR}/{base_name}_{ie}_{ib}.pt",
    )


def load_imagen(p):
    p = f"{MODEL_DIR}/{p}"
    checkpoint = torch.load(p)
    return (
        checkpoint["epoch"],
        checkpoint["model_state_dict"],
        checkpoint["trainer_state_dict"],
        checkpoint["loss"],
    )
