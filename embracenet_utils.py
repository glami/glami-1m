from dataclasses import dataclass
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb, rgba2rgb
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnext50_32x4d
from embracenet.embracenet_pytorch import EmbraceNet
from load_dataset import (
    COL_NAME_ITEM_ID,
    COL_NAME_IMAGE_FILE,
    COL_NAME_EMB_FILE,
    COL_NAME_LABEL_SOURCE,
    COL_NAME_CATEGORY,
    DEFAULT_IMAGE_SIZE,
    EMBS_DIR,
    MODEL_DIR,
    COL_NAME_MASK_FILE,
)


class EmbraceDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            dtfrm (pd.DataFrame):
            transform (callable, optional): Optional transform to be applied
                on a a sample.
        """
        self.df = df
        # add columns with embs
        self.df[COL_NAME_EMB_FILE] = self.df[COL_NAME_ITEM_ID].map(lambda x: f"{EMBS_DIR}/{x}.npy")
        self.df[COL_NAME_MASK_FILE] = self.df[COL_NAME_ITEM_ID].map(lambda x: f"{EMBS_DIR}/{x}_mask.npy")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        all_tensors = (
            self.df[
                [COL_NAME_EMB_FILE, COL_NAME_MASK_FILE, COL_NAME_IMAGE_FILE, COL_NAME_LABEL_SOURCE, COL_NAME_CATEGORY]
            ]
            .iloc[idx]
            .values
        )
        text_emb = np.load(all_tensors[0])
        text_mask = np.load(all_tensors[1])
        text_emb = text_emb * text_mask[:, None]
        text_emb = text_emb.flatten()
        loaded_image = io.imread(all_tensors[2])
        # resize and normalize to 0-1
        loaded_image = resize(loaded_image, output_shape=DEFAULT_IMAGE_SIZE, preserve_range=False,).astype(np.float32)
        # convert to rgb format if needed
        if len(loaded_image.shape) == 2:
            loaded_image = gray2rgb(loaded_image)
        elif loaded_image.shape[2] == 4:
            loaded_image = rgba2rgb(loaded_image)

        # convert to pytorch format
        loaded_image = np.moveaxis(loaded_image, -1, 0)
        sample = {
            "text_emb": text_emb,
            "image": loaded_image,
            "label_source": np.array(all_tensors[3]).astype(np.float32),
            "category": np.array(all_tensors[4]).astype(np.float32),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class EmbraceNetTrimodalClassifier(torch.nn.Module):
    def __init__(self, device, is_training, text_emb_dim, n_label_sources, n_classes, econfig, embracement_size=512):
        super(EmbraceNetTrimodalClassifier, self).__init__()

        # input parameters
        self.device = device
        self.is_training = is_training
        self.econfig = econfig

        # the second head will be resnext
        self.pre_2 = create_feature_extractor(resnext50_32x4d(pretrained=True), return_nodes=["avgpool"])
        self.pre_2.to(device)

        # embracenet
        self.embracenet = EmbraceNet(
            device=self.device, input_size_list=[text_emb_dim, 2048, n_label_sources], embracement_size=embracement_size
        )

        # post embracement layers
        self.post = torch.nn.Linear(in_features=embracement_size, out_features=n_classes)

    def forward(self, x1, x2, x3):
        # same batch size
        assert x1.shape[0] == x2.shape[0]
        assert x3.shape[0] == x2.shape[0]

        x2 = self.pre_2(x2)["avgpool"]
        x2 = x2[:, :, 0, 0]

        # drop first, second or third modality
        availabilities = None
        if self.econfig.model_drop_first or self.econfig.model_drop_second or self.econfig.model_drop_third:
            availabilities = torch.ones([x1.shape[0], 3], device=self.device)
            if self.econfig.model_drop_first:
                availabilities[:, 0] = 0
            if self.econfig.model_drop_second:
                availabilities[:, 1] = 0
            if self.econfig.model_drop_third:
                availabilities[:, 2] = 0

        # dropout during training
        if self.is_training and self.econfig.model_dropout:
            dropout_prob = torch.rand(1, device=self.device)[0]
            if dropout_prob >= 0.5:
                target_modalities = torch.round(2 * torch.rand([x1.shape[0]], device=self.device)).to(torch.int64)
                availabilities = torch.nn.functional.one_hot(target_modalities, num_classes=3).float()

        # embrace
        x_embrace = self.embracenet([x1, x2, x3], availabilities=availabilities)

        # employ final layers
        x = self.post(x_embrace)

        # output softmax
        return torch.nn.functional.log_softmax(x, dim=-1)


@dataclass
class EmbraceConfig:
    model_drop_first = False
    model_drop_second = False
    model_drop_third = False
    model_dropout = False


def save_embracenet(lo, ie, sd, od, lb, sb, ib="final", base_name="embracenet"):
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    print(f"Loss: {lo}")
    torch.save(
        {
            "epoch": ie,
            "model_state_dict": sd,
            "optimizer_state_dict": od,
            "loss": lo,
            "label_binarizer": lb,
            "source_binarizer": sb,
        },
        f"{MODEL_DIR}/{base_name}_{ie}_{ib}.pt",
    )


def load_embracenet(p):
    p = f"{MODEL_DIR}/{p}"
    print(p)
    # load embracenet and optimizer:
    checkpoint = torch.load(p)

    return (
        checkpoint["epoch"],
        checkpoint["model_state_dict"],
        checkpoint["optimizer_state_dict"],
        checkpoint["loss"],
        checkpoint["label_binarizer"],
        checkpoint["source_binarizer"],
    )
