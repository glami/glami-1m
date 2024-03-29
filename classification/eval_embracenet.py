from load_dataset import (
    download_dataset,
    EXTRACT_DIR,
    DATASET_URL,
    get_dataframe,
    COL_NAME_CATEGORY,
    COL_NAME_LABEL_SOURCE, DATASET_DIR,
)
from embracenet_utils import EmbraceConfig, load_embracenet, EmbraceNetTrimodalClassifier, EmbraceDataset
from utils import calc_accuracy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    download_dataset()
    df = get_dataframe("test")

    econfig = EmbraceConfig()

    # drop what is needed:
    econfig.model_drop_first = False
    econfig.model_drop_second = False
    econfig.model_drop_third = False

    print("Loading checkpoint...")
    ie, ms, os, ls, label_binarizer, source_binarizer = load_embracenet("embracenet_1_final.pt")

    embracenet = EmbraceNetTrimodalClassifier(
        torch.device("cuda"), False, 16384, len(source_binarizer), len(label_binarizer), econfig, 512,
    )
    embracenet.load_state_dict(ms)
    optimizer = torch.optim.Adam(embracenet.parameters())
    optimizer.load_state_dict(os)

    print(f"Loaded Embracenet after epoch {ie}.")

    embracenet.cuda()
    embracenet.eval()
    print("Binarizing targets...")
    df[COL_NAME_CATEGORY] = df[COL_NAME_CATEGORY].map(label_binarizer)
    print("Binarizing sources...")
    df[COL_NAME_LABEL_SOURCE] = df[COL_NAME_LABEL_SOURCE].map(source_binarizer)

    BATCH_SIZE = 256
    ds = EmbraceDataset(df, transform=None)
    dl_test = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    print("Evaluating...")
    preds = []
    targets = []
    with torch.no_grad():
        for i_batch, sample_batched in tqdm(
            enumerate(dl_test), desc=f"Predicting:", total=int(np.ceil(len(ds) / BATCH_SIZE))
        ):
            outputs = embracenet(
                sample_batched["text_emb"].cuda(), sample_batched["image"].cuda(), sample_batched["label_source"].cuda()
            )
            preds.extend(outputs.cpu().detach().tolist())
            targets.extend(sample_batched["category"].cpu().bool().detach().tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    accs = calc_accuracy(preds, targets)
    print(f"Accuracies:")
    print(accs)
    geos = df["geo"].unique()
    for geo in geos:
        geo_mask = df["geo"] == geo
        geo_mask = geo_mask.values
        geo_preds = preds[geo_mask]
        geo_targets = targets[geo_mask]

        accs = calc_accuracy(geo_preds, geo_targets)
        print(f"Accuracies ({geo}):")
        print(accs)
