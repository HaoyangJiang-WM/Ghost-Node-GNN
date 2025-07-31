import copy
import numpy as np
import os
import random
import torch
import torch.nn as nn
from math import floor, ceil


# import dataset_ext1 as ds         # ghost
import dataset_ext_dense as ds  # ghost and dense
# import data_noext_dense as ds

import models_ext1 as models
# import models_o as models

from torch.nn.functional import mse_loss
from torch.utils.data import random_split
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_undirected, to_torch_coo_tensor

from torchinfo import summary
from tqdm import tqdm


def ensure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_edge_weights(adjacency_type, edge_attr):
    if adjacency_type == "isolated":
        return torch.zeros(edge_attr.size(0))
    elif adjacency_type == "binary":
        return torch.ones(edge_attr.size(0))
    elif adjacency_type == "stream_length":
        return edge_attr[:, 0]
    elif adjacency_type == "elevation_difference":
        return edge_attr[:, 1]
    elif adjacency_type == "average_slope":
        return edge_attr[:, 2]
    elif adjacency_type == "learned":
        return nn.Parameter(torch.nn.init.uniform_(torch.empty(edge_attr.size(0)), 0.9, 1.1))
    elif adjacency_type == "all":
        return edge_attr[:, :]
    else:
        raise ValueError("invalid adjacency type", adjacency_type)


def construct_model(hparams, dataset):
    # ---  edge_attr_with_ghosts to initialize ---
    edge_weights = get_edge_weights(hparams["model"]["adjacency_type"], dataset.edge_attr_with_ghosts)
    # edge_weights = get_edge_weights(hparams["model"]["adjacency_type"], dataset.edge_attr)

    model_arch = hparams["model"]["architecture"]
    if model_arch == "MLP":
        return models.MLP(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                          hidden_channels=hparams["model"]["hidden_channels"],
                          num_hidden=hparams["model"]["num_layers"],
                          param_sharing=hparams["model"]["param_sharing"])
    elif model_arch == "GCN":
        return models.GCN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                          hidden_channels=hparams["model"]["hidden_channels"],
                          num_hidden=hparams["model"]["num_layers"],
                          param_sharing=hparams["model"]["param_sharing"],
                          edge_orientation=hparams["model"]["edge_orientation"],
                          edge_weights=edge_weights
                          )
    elif model_arch == "ResGCN":
        return models.ResGCN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                             hidden_channels=hparams["model"]["hidden_channels"],
                             num_hidden=hparams["model"]["num_layers"],
                             param_sharing=hparams["model"]["param_sharing"],
                             edge_orientation=hparams["model"]["edge_orientation"],
                             edge_weights=edge_weights)
    elif model_arch == "GCNII":
        return models.GCNII(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                            hidden_channels=hparams["model"]["hidden_channels"],
                            num_hidden=hparams["model"]["num_layers"],
                            param_sharing=hparams["model"]["param_sharing"],
                            edge_orientation=hparams["model"]["edge_orientation"],
                            edge_weights=edge_weights)
    elif model_arch == "ResGAT":
        return models.ResGAT(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                             hidden_channels=hparams["model"]["hidden_channels"],
                             num_hidden=hparams["model"]["num_layers"],
                             param_sharing=hparams["model"]["param_sharing"],
                             edge_orientation=hparams["model"]["edge_orientation"],
                             edge_weights=edge_weights)

    elif model_arch == "ResGraphSAGE":
        return models.ResGraphSAGE(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                             hidden_channels=hparams["model"]["hidden_channels"],
                             num_hidden=hparams["model"]["num_layers"],
                             param_sharing=hparams["model"]["param_sharing"],
                             edge_orientation=hparams["model"]["edge_orientation"],
                             edge_weights=edge_weights)

    elif model_arch == "CustomMPNN":
        return models.CustomMPNN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                             hidden_channels=hparams["model"]["hidden_channels"],
                             num_hidden=hparams["model"]["num_layers"],
                             param_sharing=hparams["model"]["param_sharing"],
                             edge_orientation=hparams["model"]["edge_orientation"],
                             edge_weights=edge_weights)


    raise ValueError("unknown model architecture", model_arch)


def load_dataset(path, hparams, split):
    if split == "train":
        years = hparams["training"]["train_years"]
    elif split == "test":
        years = [2016, 2017]
    else:
        raise ValueError("unknown split", split)

    use_extrapolation = hparams["data"].get("use_feature_extrapolation", False)
    use_concat = hparams["data"].get("use_feature_concatenation", True)
    print(f"Feature extrapolation: {'Enabled' if use_extrapolation else 'Disabled'}")
    print(f"Feature concatenation: {'Enabled' if use_concat else 'Disabled'}")

    return ds.LamaHDataset(path,
                           years=years,
                           root_gauge_id=hparams["data"]["root_gauge_id"],
                           rewire_graph=hparams["data"]["rewire_graph"],
                           window_size=hparams["data"]["window_size"],
                           stride_length=hparams["data"]["stride_length"],
                           lead_time=hparams["data"]["lead_time"],
                           normalized=hparams["data"]["normalized"],
                           use_feature_extrapolation=use_extrapolation)

    # return ds.LamaHDataset(path,
    #                        years=years,
    #                        root_gauge_id=hparams["data"]["root_gauge_id"],
    #                        rewire_graph=hparams["data"]["rewire_graph"],
    #                        window_size=hparams["data"]["window_size"],
    #                        stride_length=hparams["data"]["stride_length"],
    #                        lead_time=hparams["data"]["lead_time"],
    #                        normalized=hparams["data"]["normalized"])


def load_model_and_dataset(chkpt, dataset_path):
    model_params = chkpt["history"]["best_model_params"]
    dataset = load_dataset(dataset_path, chkpt["hparams"], split="test")
    model = construct_model(chkpt["hparams"], dataset)
    model.load_state_dict(model_params, strict=False)
    return model, dataset


def train_step(model, train_loader, criterion, optimizer, device, reset_running_loss_after=10):
    model.train()
    train_loss = 0.0
    running_loss = 0.0
    running_counter = 1
    with tqdm(train_loader, desc="Training") as pbar:
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch.x, batch.edge_index,
                         ghost_mask=batch.ghost_mask,
                         x_ghost_concat=batch.x_ghost_concat)

            loss = criterion(pred[batch.mask], batch.y)

            # pred = model(batch.x, batch.edge_index)
            #
            # loss = criterion(pred, batch.y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs / len(train_loader.dataset)
            running_loss += loss.item() / reset_running_loss_after
            running_counter += 1
            if running_counter >= reset_running_loss_after:
                pbar.set_postfix({"loss": running_loss})
                running_counter = 1
                running_loss = 0.0
    return train_loss


def val_step(model, val_loader, criterion, device, load_path="node_classes.npy"):
    model.eval()
    val_loss = 0.0
    node_classes = np.load(load_path, allow_pickle=True).item()
    no_input_nodes = node_classes['no_input']
    single_input_nodes = node_classes['single_input']
    multi_input_nodes = node_classes['multi_input']
    no_input_losses, single_input_losses, multi_input_losses = [], [], []

    with torch.no_grad():
        with tqdm(val_loader, desc="Validating") as pbar:
            for batch in pbar:
                batch = batch.to(device)

                # --- 修改: 传递所有必需的参数给模型 ---
                pred = model(batch.x, batch.edge_index,
                             ghost_mask=batch.ghost_mask,
                             x_ghost_concat=batch.x_ghost_concat)

                loss = criterion(pred[batch.mask], batch.y)
                val_loss += loss.item() * batch.num_graphs / len(val_loader.dataset)
                loss_per_sample = criterion(pred[batch.mask], batch.y, reduction='none')

                node_offset = torch.cumsum(torch.bincount(batch.batch[batch.mask]), dim=0)

                # pred = model(batch.x, batch.edge_index)
                #
                # loss = criterion(pred, batch.y)
                # val_loss += loss.item() * batch.num_graphs / len(val_loader.dataset)
                # loss_per_sample = criterion(pred, batch.y, reduction='none')
                #
                # node_offset = torch.cumsum(torch.bincount(batch.batch), dim=0)

                node_offset = torch.cat([torch.tensor([0], device=device), node_offset[:-1]])
                for graph_idx in range(batch.num_graphs):
                    offset = node_offset[graph_idx].item()
                    no_input_idx = [i + offset for i in no_input_nodes]
                    single_input_idx = [i + offset for i in single_input_nodes]
                    multi_input_idx = [i + offset for i in multi_input_nodes]

                    if len(no_input_idx) > 0:
                        no_input_losses.append(loss_per_sample[no_input_idx].mean().item())
                    if len(single_input_idx) > 0:
                        single_input_losses.append(loss_per_sample[single_input_idx].mean().item())
                    if len(multi_input_idx) > 0:
                        multi_input_losses.append(loss_per_sample[multi_input_idx].mean().item())

    no_input_error = sum(no_input_losses) / len(no_input_losses) if no_input_losses else 0.0
    single_input_error = sum(single_input_losses) / len(single_input_losses) if single_input_losses else 0.0
    multi_input_error = sum(multi_input_losses) / len(multi_input_losses) if multi_input_losses else 0.0

    return val_loss, no_input_error, single_input_error, multi_input_error




def train(model, dataset, hparams):
    print(summary(model, depth=2))
    holdout_size = hparams["training"]["holdout_size"]
    dataset_length = len(dataset)
    val_size = int(holdout_size * dataset_length)
    train_size = dataset_length - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=hparams["training"]["batch_size"], shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["training"]["batch_size"], shuffle=False, num_workers=2,
                            pin_memory=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion1 = mse_loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams["training"]["learning_rate"],
                                 weight_decay=hparams["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)
    model = model.to(device)
    print("Training on", device)
    history = {"train_loss": [], "val_loss": [], "nse_score": [], "other_error": [], "best_model_params": None}
    min_val_loss = float("inf")
    for epoch in range(hparams["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{hparams['training']['num_epochs']}")
        train_loss = train_step(model, train_loader, criterion1, optimizer, device)
        val_loss, no_input_error, single_input_error, multi_input_error = val_step(
            model, val_loader, criterion1, device, load_path="node_classes.npy"
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["nse_score"].append(no_input_error)
        history["other_error"].append(single_input_error)
        print(
            f"[Epoch {epoch + 1}/{hparams['training']['num_epochs']}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"no: {no_input_error:.4f} |  "
            f"single: {single_input_error:.4f}| "
            f"multi: {multi_input_error:.4f} ")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            history["best_model_params"] = copy.deepcopy(model.state_dict())
        scheduler.step(val_loss)
    return history


def save_checkpoint(history, hparams, filename, directory="./runs"):
    directory = directory.rstrip("/")
    os.makedirs(directory, exist_ok=True)
    out_path = f"{directory}/{filename}"
    torch.save({
        "history": history,
        "hparams": hparams
    }, out_path)
    print("Saved checkpoint", out_path)

