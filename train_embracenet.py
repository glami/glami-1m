from load_dataset import (
    download_dataset,
    EXTRACT_DIR,
    DATASET_URL,
    get_dataframe,
    COL_NAME_CATEGORY,
    COL_NAME_LABEL_SOURCE,
)
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from embracenet_utils import EmbraceDataset, EmbraceConfig, EmbraceNetTrimodalClassifier, save_embracenet
import torch
from tqdm import tqdm

if __name__ == "__main__":
    download_dataset(EXTRACT_DIR, DATASET_URL)
    dataset_dir = EXTRACT_DIR + "/glami-2022-dataset"
    df = get_dataframe(dataset_dir, "train")

    print("One hot encoding...")
    y = pd.get_dummies(df[COL_NAME_CATEGORY], prefix="cat")
    n_cats = y.values.shape[1]
    label_binarizer = {int(x[4:]): np.eye(1, len(y.columns), i, dtype=int)[0].tolist() for i, x in enumerate(y.columns)}
    df[COL_NAME_CATEGORY] = df[COL_NAME_CATEGORY].map(label_binarizer)

    df[COL_NAME_LABEL_SOURCE] = df[COL_NAME_LABEL_SOURCE].fillna("unknown")
    y = pd.get_dummies(df[COL_NAME_LABEL_SOURCE], prefix="src")
    n_sources = y.values.shape[1]
    source_binarizer = {x[4:]: np.eye(1, len(y.columns), i, dtype=int)[0].tolist() for i, x in enumerate(y.columns)}
    df[COL_NAME_LABEL_SOURCE] = df[COL_NAME_LABEL_SOURCE].map(source_binarizer)

    ds = EmbraceDataset(df, transform=None)

    BATCH_SIZE = 64
    dl_train = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    embrace_config = EmbraceConfig()
    embrace_config.model_drop_first = False
    embrace_config.model_drop_second = False
    embrace_config.model_drop_third = False
    embracenet = EmbraceNetTrimodalClassifier(torch.device("cuda"), True, 16384, n_sources, n_cats, embrace_config, 512)
    embracenet.cuda()

    EPOCHS = 2
    SAVE_EVERY = 5000
    basename = "full_embracenet"
    optimizer = torch.optim.Adam(embracenet.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    print("Training...")
    for i_epoch in range(EPOCHS):
        losses = []
        for i_batch, sample_batched in tqdm(
            enumerate(dl_train), desc=f"Epoch {i_epoch}:", total=int(np.ceil(len(ds) / BATCH_SIZE))
        ):
            optimizer.zero_grad()
            outputs = embracenet(
                sample_batched["text_emb"].cuda(), sample_batched["image"].cuda(), sample_batched["label_source"].cuda()
            )
            loss = loss_fn(outputs, sample_batched["category"].cuda())
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())

            if (i_batch + 1) % SAVE_EVERY == 0:
                save_embracenet(
                    np.mean(losses),
                    i_epoch,
                    embracenet.state_dict(),
                    optimizer.state_dict(),
                    label_binarizer,
                    source_binarizer,
                    i_batch,
                    base_name=basename,
                )

        save_embracenet(
            np.mean(losses),
            i_epoch,
            embracenet.state_dict(),
            optimizer.state_dict(),
            label_binarizer,
            source_binarizer,
            base_name=basename,
        )
