import copy
import numpy as np
import os
import random
import torch
import torch.nn as nn
from math import floor, ceil

from dataset import LamaHDataset
# from dataset_dense import LamaHDataset
# from dataset_initial_bidi import LamaHDataset
# from dataset_depth import LamaHDataset
# from dataset_ghost_3 import LamaHDataset
# from dataset_ghost_dense import LamaHDataset
# from dataset_ghost_end import LamaHDataset
# from models_o import MLP, GCN, ResGCN, GCNII, ResGAT, ResGraphSAGE, CustomMPNN, ResRWGCN, ResGraphWavelet
from models_o import MLP, GCN, ResGCN, GCNII, ResGAT, ResGraphSAGE, CustomMPNN
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
    edge_weights = get_edge_weights(hparams["model"]["adjacency_type"], dataset.edge_attr)
    # edge_weights = get_edge_weights(hparams["model"]["adjacency_type"], dataset.edge_attr1)
    print(' d', edge_weights.shape)
    model_arch = hparams["model"]["architecture"]
    if model_arch == "MLP":
        return MLP(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                   hidden_channels=hparams["model"]["hidden_channels"],
                   num_hidden=hparams["model"]["num_layers"],
                   param_sharing=hparams["model"]["param_sharing"])
    elif model_arch == "GCN":
        return GCN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                   hidden_channels=hparams["model"]["hidden_channels"],
                   num_hidden=hparams["model"]["num_layers"],
                   param_sharing=hparams["model"]["param_sharing"],
                   edge_orientation=hparams["model"]["edge_orientation"],
                   edge_weights=edge_weights
                   )
    elif model_arch == "ResGCN":
        return ResGCN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                      hidden_channels=hparams["model"]["hidden_channels"],
                      num_hidden=hparams["model"]["num_layers"],
                      param_sharing=hparams["model"]["param_sharing"],
                      edge_orientation=hparams["model"]["edge_orientation"],
                      edge_weights=edge_weights)
    elif model_arch == "GCNII":
        return GCNII(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                     hidden_channels=hparams["model"]["hidden_channels"],
                     num_hidden=hparams["model"]["num_layers"],
                     param_sharing=hparams["model"]["param_sharing"],
                     edge_orientation=hparams["model"]["edge_orientation"],
                     edge_weights=edge_weights)
    elif model_arch == "ResGAT":
        return ResGAT(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                      hidden_channels=hparams["model"]["hidden_channels"],
                      num_hidden=hparams["model"]["num_layers"],
                      param_sharing=hparams["model"]["param_sharing"],
                      edge_orientation=hparams["model"]["edge_orientation"],
                      edge_weights=edge_weights)

    elif model_arch == "ResGraphSAGE":
        return ResGraphSAGE(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                            hidden_channels=hparams["model"]["hidden_channels"],
                            num_hidden=hparams["model"]["num_layers"],
                            param_sharing=hparams["model"]["param_sharing"],
                            edge_orientation=hparams["model"]["edge_orientation"],
                            edge_weights=edge_weights)

    elif model_arch == "CustomMPNN":
        return CustomMPNN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                          hidden_channels=hparams["model"]["hidden_channels"],
                          num_hidden=hparams["model"]["num_layers"],
                          param_sharing=hparams["model"]["param_sharing"],
                          edge_orientation=hparams["model"]["edge_orientation"],
                          edge_weights=edge_weights)

    elif model_arch == "ResRWGCN":
        return ResRWGCN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                          hidden_channels=hparams["model"]["hidden_channels"],
                          num_hidden=hparams["model"]["num_layers"],
                          param_sharing=hparams["model"]["param_sharing"],
                          edge_orientation=hparams["model"]["edge_orientation"],
                          edge_weights=edge_weights)

    elif model_arch == "ResGraphWavelet":
        return ResGraphWavelet(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
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
    return LamaHDataset(path,
                        years=years,
                        root_gauge_id=hparams["data"]["root_gauge_id"],
                        rewire_graph=hparams["data"]["rewire_graph"],
                        window_size=hparams["data"]["window_size"],
                        stride_length=hparams["data"]["stride_length"],
                        lead_time=hparams["data"]["lead_time"],
                        normalized=hparams["data"]["normalized"])


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

            # # 打印 batch 中的所有变量及其形状
            # print("Batch Variables and Shapes:")
            # for attr in dir(batch):
            #     # 排除特殊方法和不可访问的属性
            #     if not attr.startswith('_'):
            #         value = getattr(batch, attr)
            #         # 如果属性是 PyTorch 张量，打印它的形状
            #         if isinstance(value, torch.Tensor):
            #             print(f"{attr}: {value.shape}")
            #         else:
            #             print(f"{attr}: (not a tensor)")

            #             # 打印 batch 中的所有变量
            #             print("Batch Variables:")
            #             for key, value in batch.__dict__.items():
            #                 if torch.is_tensor(value):
            #                     print(f"{key}: {value.shape}")
            #                 else:
            #                     print(f"{key}: {value}")

            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            # loss = criterion(pred[batch.mask], batch)
            # 使用 mask 过滤鬼点，只计算真实节点的损失
            # print('batch.mask',batch.mask.shape)
            # print("pred[batch.mask].shape:", pred[batch.mask].shape)
            # print("batch.y.shape:", batch.y.shape)
            loss = criterion(pred, batch.y)
            # loss = criterion(pred[batch.mask], batch.y)
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


import torch
from tqdm import tqdm


def val_step(model, val_loader, criterion, device, load_path="node_classes.npy"):
    """
    验证步骤：支持批量图处理，并根据偏移量修正节点分类索引。
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        load_path: 已保存的节点分类结果路径

    Returns:
        val_loss_avg: 验证集的平均损失
        no_input_error: 没有输入节点的平均误差
        single_input_error: 1个输入节点的平均误差
        multi_input_error: 多个输入节点的平均误差
    """
    model.eval()  # 设置为评估模式
    val_loss = 0.0  # 总损失

    # === 加载已保存的节点分类结果 ===
    node_classes = np.load(load_path, allow_pickle=True).item()
    no_input_nodes = node_classes['no_input']
    single_input_nodes = node_classes['single_input']
    multi_input_nodes = node_classes['multi_input']

    # 各类节点损失列表
    no_input_losses = []
    single_input_losses = []
    multi_input_losses = []

    # 遍历验证集的批量图数据
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating") as pbar:
            for batch in pbar:

                batch = batch.to(device)


                pred = model(batch.x, batch.edge_index)


                loss = criterion(pred, batch.y)

                val_loss += loss.item() * batch.num_graphs / len(val_loader.dataset)


                loss_per_sample = criterion(pred, batch.y, reduction='none')
                # loss_per_sample = criterion(pred[batch.mask], batch.y, reduction='none')
                # print(loss_per_sample.shape)



                node_offset = torch.cumsum(torch.bincount((batch.batch)), dim=0)
                # node_offset = torch.cumsum(torch.bincount((batch.batch[batch.mask])), dim=0)
                node_offset = torch.cat([torch.tensor([0], device=device), node_offset[:-1]])


                for graph_idx in range(batch.num_graphs):

                    offset = node_offset[graph_idx].item()
                    no_input_idx = [i + offset for i in no_input_nodes]
                    single_input_idx = [i + offset for i in single_input_nodes]
                    multi_input_idx = [i + offset for i in multi_input_nodes]


                    if len(no_input_idx) > 0:
                        no_input_loss = loss_per_sample[no_input_idx].mean().item()
                        no_input_losses.append(no_input_loss)


                    if len(single_input_idx) > 0:
                        single_input_loss = loss_per_sample[single_input_idx].mean().item()
                        single_input_losses.append(single_input_loss)


                    if len(multi_input_idx) > 0:
                        multi_input_loss = loss_per_sample[multi_input_idx].mean().item()
                        multi_input_losses.append(multi_input_loss)


    # val_loss_avg = val_loss / len(val_loader)


    no_input_error = sum(no_input_losses) / len(no_input_losses) if no_input_losses else 0.0
    single_input_error = sum(single_input_losses) / len(single_input_losses) if single_input_losses else 0.0
    multi_input_error = sum(multi_input_losses) / len(multi_input_losses) if multi_input_losses else 0.0

    return val_loss, no_input_error, single_input_error, multi_input_error


def interestingness_score(batch, dataset, device):
    mean = dataset.mean[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    std = dataset.std[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    # unnormalized_discharge = mean + std * batch.x[:, :, 0]
    unnormalized_discharge = mean + std * batch.x[batch.mask][:, :, 0]
    assert unnormalized_discharge.min() >= 0.0
    comparable_discharge = unnormalized_discharge / mean

    mean_central_diff = torch.gradient(comparable_discharge, dim=-1)[0].mean()
    trapezoid_integral = torch.trapezoid(comparable_discharge, dim=-1)

    score = 1e3 * (mean_central_diff ** 2) * trapezoid_integral
    assert not trapezoid_integral.isinf().any()
    assert not trapezoid_integral.isnan().any()
    return score.unsqueeze(-1)


def interestingness_score_normalization_const(loader, device):
    total_score = 0.0
    for batch in tqdm(loader, desc="Summing all scores"):
        total_score += interestingness_score(batch, loader.dataset, device).item()
    return total_score


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
    # criterion = mse_loss  #  interestingness_score
    criterion = lambda pred, batch: (interestingness_score(batch, dataset, device) * mse_loss(pred, batch.y,
                                                                                              reduction="none")).mean()  # mse_loss(pred, batch.y)
    criterion1 = mse_loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams["training"]["learning_rate"],
                                 weight_decay=hparams["training"]["weight_decay"])

    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)

    model = model.to(device)
    print("Training on", device)

    history = {"train_loss": [], "val_loss": [], "nse_score": [], "other_error": [], "best_model_params": None}
    min_val_loss = float("inf")

    for epoch in range(hparams["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{hparams['training']['num_epochs']}")
        train_loss = train_step(model, train_loader, criterion1, optimizer, device)
        #         val_loss = val_step(model, val_loader, criterion1, device)
        # train_loss = train_step(model, train_loader, criterion, optimizer, device, accumulation_steps=4,
        #                         grad_clip=hparams["training"]["grad_clip"])
        # val_loss, no_input_error, single_input_error, multi_input_error = val_step(model, val_loader, criterion1, device)
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


def load_checkpoint(chkpt_path):
    return torch.load(chkpt_path, map_location=torch.device("cpu"))



def evaluate_nse(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mean = dataset.mean[:, [0]].to(device)
    std_squared = dataset.std[:, [0]].square().to(device)

    with torch.no_grad():
        weighted_model_error = torch.zeros(dataset[0].num_nodes, 1).to(device)
        weighted_mean_error = torch.zeros(dataset[0].num_nodes, 1).to(device)
        for data in tqdm(dataset, desc="Testing"):
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr)
            model_mse = mse_loss(pred, data.y, reduction="none")
            mean_mse = mse_loss(mean, data.y, reduction="none")

            if dataset.normalized:
                model_mse *= std_squared
                mean_mse *= std_squared


            weighted_model_error += model_mse
            weighted_mean_error += mean_mse

    weighted_nse = 1 - weighted_model_error / weighted_mean_error
    return weighted_nse.cpu()


def calculate_predictions_and_deviations_on_gauge(model, dataset, gauge_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = []
    deviations = []
    with torch.no_grad():
        for data in tqdm(dataset, desc="Testing"):
            data = data.to(device)
            pred = model(data.x, data.edge_index)[gauge_index]
            target = data.y[gauge_index]
            predictions.append(pred.item())
            deviations.append(abs(pred - target).item())
    return predictions, deviations


def dirichlet_energy(x, edge_index, edge_weight, normalization=None):
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization=normalization)
    lap = to_torch_coo_tensor(edge_index=edge_index, edge_attr=edge_weight)
    return 0.5 * torch.trace(torch.mm(x.T, torch.sparse.mm(lap, x)))


def evaluate_dirichlet_energy(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    dirichlet_stats = []
    with torch.no_grad():
        edge_weights = model.edge_weights.detach().nan_to_num().to(device)
        for data in tqdm(dataset, desc="Testing"):
            data = data.to(device)
            _, evo = model(data.x, data.edge_index, evo_tracking=True)
            dir_energies = torch.tensor([dirichlet_energy(h, data.edge_index, edge_weights) for h in evo])
            dirichlet_stats.append(dir_energies)
    dirichlet_stats = torch.stack(dirichlet_stats)
    return dirichlet_stats
