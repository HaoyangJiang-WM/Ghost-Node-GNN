

import os
import pandas as pd
import tarfile
import torch
import urllib.request
import warnings

from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class LamaHDataset(Dataset):

    DATA_URL = "https://zenodo.org/record/5153305/files/1_LamaH-CE_daily_hourly.tar.gz"
    Q_COL = "qobs"
    MET_COLS = [
        "prec",  # precipitation
        "volsw_123",  # topsoil moisture
        "2m_temp",  # air temperature
        "surf_press",  # surface pressure
    ]

    def __init__(self, root_dir, years=range(2000, 2018), root_gauge_id=399, rewire_graph=True,
                 window_size=24, stride_length=1, lead_time=6, normalized=False,
                 use_feature_extrapolation=False):
        if not set(years).issubset(range(2000, 2018)):
            raise ValueError("Only years between 2000 and 2017 are supported")

        self.years = years
        self.root_gauge_id = root_gauge_id
        self.rewire_graph = rewire_graph
        self.window_size = window_size
        self.stride_length = stride_length
        self.lead_time = lead_time
        self.normalized = normalized
        self.use_feature_extrapolation = use_feature_extrapolation

        super().__init__(root_dir)

        adj_df = pd.read_csv(self.processed_paths[0])
        self.gauges = list(sorted(set(adj_df["ID"]).union(adj_df["NEXTDOWNID"])))
        self.rev_index = {gauge_id: i for i, gauge_id in enumerate(self.gauges)}

        edge_cols = adj_df[["ID", "NEXTDOWNID"]].applymap(lambda x: self.rev_index.get(x, -1))
        valid_edges_mask = (edge_cols["ID"] != -1) & (edge_cols["NEXTDOWNID"] != -1)
        edge_cols = edge_cols[valid_edges_mask]

        edge_index = torch.tensor(edge_cols.values.transpose(), dtype=torch.long)
        weight_cols = adj_df.loc[edge_cols.index][["dist_hdn", "elev_diff", "strm_slope"]]
        edge_attr = torch.tensor(weight_cols.values, dtype=torch.float)

        self.original_edge_index = edge_index.clone()
        self.original_edge_attr = edge_attr.clone()

        self.edge_index_with_ghosts, self.edge_attr_with_ghosts, self.ghost_node_map = self.create_ghost_graph_structure()

        stats_df = pd.read_csv(self.processed_paths[1], index_col="ID")
        self.mean = torch.tensor(stats_df[[f"{col}_mean" for col in [self.Q_COL] + self.MET_COLS]].values,
                                 dtype=torch.float)
        self.std = torch.tensor(stats_df[[f"{col}_std" for col in [self.Q_COL] + self.MET_COLS]].values,
                                dtype=torch.float)

        self.year_sizes = [(24 * (365 + int(year % 4 == 0)) - (window_size + lead_time)) // stride_length + 1 for year
                           in years]

        self.year_tensors = [[] for _ in years]
        print("Loading dataset into memory...")
        for gauge_id in tqdm(self.gauges):
            q_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[2]}/hourly/ID_{gauge_id}.csv", sep=";",
                               usecols=["YYYY", self.Q_COL])
            met_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[1]}/hourly/ID_{gauge_id}.csv", sep=";",
                                 usecols=["YYYY"] + self.MET_COLS)
            if normalized:
                q_df[self.Q_COL] = (q_df[self.Q_COL] - stats_df.loc[gauge_id, f"{self.Q_COL}_mean"]) / (
                            stats_df.loc[gauge_id, f"{self.Q_COL}_std"] + 1e-9)
                for col in self.MET_COLS:
                    met_df[col] = (met_df[col] - stats_df.loc[gauge_id, f"{col}_mean"]) / (
                                stats_df.loc[gauge_id, f"{col}_std"] + 1e-9)

            for i, year in enumerate(years):
                q_tensor = torch.tensor(q_df[q_df["YYYY"] == year][self.Q_COL].values, dtype=torch.float).unsqueeze(-1)
                met_tensor = torch.tensor(met_df[met_df["YYYY"] == year][self.MET_COLS].values, dtype=torch.float)
                self.year_tensors[i].append(torch.cat([q_tensor, met_tensor], dim=1))

        self.year_tensors[:] = map(torch.stack, self.year_tensors)


    @property
    def raw_file_names(self):
        return ["B_basins_intermediate_all/1_attributes", "B_basins_intermediate_all/2_timeseries",
                "D_gauges/2_timeseries"]

    @property
    def processed_file_names(self):
        return [f"adjacency_{self.root_gauge_id}_{self.rewire_graph}.csv",
                f"statistics_{self.root_gauge_id}_{self.rewire_graph}.csv"]

    def download(self):
        print("Downloading LamaH-CE from Zenodo to", self.raw_dir)
        total_size = int(urllib.request.urlopen(self.DATA_URL).info().get("Content-Length"))
        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading") as pbar:
            filename, _ = urllib.request.urlretrieve(self.DATA_URL, filename="./archive.tar",
                                                     reporthook=lambda _, n, __: pbar.update(n))
        archive = tarfile.open(filename)
        for member in tqdm(archive.getmembers(), desc="Extracting"):
            if member.name.startswith(tuple(self.raw_file_names)):
                archive.extract(member, self.raw_dir)
        os.remove(filename)

    def process(self):
        adj_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[0]}/Stream_dist.csv", sep=";")
        if "strm_slope" in adj_df.columns:
            adj_df.drop(columns="strm_slope", inplace=True)
        stats_df = pd.DataFrame(
            columns=sum([[f"{col}_mean", f"{col}_std"] for col in [self.Q_COL] + self.MET_COLS], []),
            index=pd.Index([], name="ID"))
        adj_df['ID'] = pd.to_numeric(adj_df['ID'], errors='coerce').dropna().astype(int)
        adj_df['NEXTDOWNID'] = pd.to_numeric(adj_df['NEXTDOWNID'], errors='coerce').dropna().astype(int)
        connected_gauges = set(adj_df["ID"]).union(adj_df["NEXTDOWNID"])
        print(f"Discovering feasible gauges...")
        feasible_gauges = set(self._collect_upstream(self.root_gauge_id, adj_df.copy(), stats_df))
        print()
        assert feasible_gauges.issubset(connected_gauges)
        print(f"Discovered {len(feasible_gauges)} feasible gauges...")
        for gauge_id in tqdm(connected_gauges - feasible_gauges, desc="Bad gauge removal"):
            adj_df = self._remove_gauge_edges(gauge_id, adj_df)
        print("Saving final adjacency list...")
        adj_df["strm_slope"] = adj_df["elev_diff"] / adj_df["dist_hdn"].clip(lower=1e-9)
        adj_df.sort_values(by="ID", inplace=True)
        adj_df.to_csv(self.processed_paths[0], index=False)
        print("Saving feature summary statistics...")
        stats_df = stats_df[stats_df.index.isin(feasible_gauges)]
        stats_df.sort_values(by="ID", inplace=True)
        stats_df.to_csv(self.processed_paths[1], index=True)

    def _collect_upstream(self, gauge_id, adj_df, stats_df):
        print(f"Processing gauge #{gauge_id}", end="\r", flush=True)
        collected_ids = set()
        is_valid, gauge_stats = self._has_valid_data(gauge_id)
        if is_valid:
            collected_ids.add(gauge_id)
            stats_df.loc[gauge_id] = gauge_stats
        if is_valid or self.rewire_graph:
            predecessor_ids = set(adj_df[adj_df["NEXTDOWNID"] == gauge_id]["ID"])
            collected_ids.update(*[self._collect_upstream(pred_id, adj_df, stats_df) for pred_id in predecessor_ids])
        return collected_ids

    def _has_valid_data(self, gauge_id):
        try:
            q_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[2]}/hourly/ID_{gauge_id}.csv", sep=";",
                               usecols=["YYYY", self.Q_COL])
            met_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[1]}/hourly/ID_{gauge_id}.csv", sep=";",
                                 usecols=["YYYY"] + self.MET_COLS)
            if (q_df[self.Q_COL] <= 0).any() or (q_df[
                                                     self.Q_COL] > 1e30).any() or q_df.isnull().values.any() or met_df.isnull().values.any(): return False, None
            df_slice = q_df[(q_df["YYYY"] >= 2000) & (q_df["YYYY"] <= 2017)]
            met_slice = met_df[(met_df["YYYY"] >= 2000) & (met_df["YYYY"] <= 2017)]
            if len(df_slice) == (18 * 365 + 5) * 24 and len(met_slice) == (18 * 365 + 5) * 24:
                q_df_train = q_df[q_df["YYYY"] <= 2015]
                met_df_train = met_df[met_df["YYYY"] <= 2015]
                return True, [q_df_train[self.Q_COL].mean(), q_df_train[self.Q_COL].std()] + sum(
                    [[met_df_train[col].mean(), met_df_train[col].std()] for col in self.MET_COLS], [])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return False, None
        return False, None

    def _remove_gauge_edges(self, gauge_id, adj_df):
        incoming_edges = adj_df.loc[adj_df["NEXTDOWNID"] == gauge_id]
        outgoing_edges = adj_df.loc[adj_df["ID"] == gauge_id]
        adj_df.drop(labels=incoming_edges.index, inplace=True)
        adj_df.drop(labels=outgoing_edges.index, inplace=True)
        if self.rewire_graph:
            bypass = incoming_edges.merge(outgoing_edges, how="cross", suffixes=["", "_"])
            if not bypass.empty:
                bypass["NEXTDOWNID"] = bypass["NEXTDOWNID_"]
                bypass["dist_hdn"] += bypass["dist_hdn_"]
                bypass["elev_diff"] += bypass["elev_diff_"]
                adj_df = pd.concat([adj_df, bypass[["ID", "NEXTDOWNID", "dist_hdn", "elev_diff"]]], ignore_index=True)
        return adj_df.reset_index(drop=True)

    def len(self):
        return sum(self.year_sizes)

    def create_ghost_graph_structure(self):
        edge_index = self.original_edge_index
        edge_attr = self.original_edge_attr
        source_nodes = edge_index[0].tolist()
        target_nodes = edge_index[1].tolist()
        initial_nodes = list(set(source_nodes) - set(target_nodes))
        num_original_nodes = len(self.gauges)
        ghost_node_map = {}
        new_edges, new_attrs = [], []
        for i, init_node in enumerate(initial_nodes):
            ghost_node_id = num_original_nodes + i
            ghost_node_map[ghost_node_id] = init_node
            new_edges.append([ghost_node_id, init_node])
            try:
                downstream_edge_idx = source_nodes.index(init_node)
                new_attrs.append(edge_attr[downstream_edge_idx].tolist())
            except (ValueError, IndexError):
                num_features = edge_attr.shape[1] if edge_attr.numel() > 0 else 3
                new_attrs.append([0.0] * num_features)
        if new_edges:
            ghost_edges = torch.tensor(new_edges, dtype=torch.long).t()
            edge_index_with_ghosts = torch.cat([edge_index, ghost_edges], dim=1)
            ghost_attrs = torch.tensor(new_attrs, dtype=torch.float)
            edge_attr_with_ghosts = torch.cat([edge_attr, ghost_attrs], dim=0)
        else:
            edge_index_with_ghosts, edge_attr_with_ghosts = edge_index, edge_attr
        return edge_index_with_ghosts, edge_attr_with_ghosts, ghost_node_map

    def get(self, idx):
        year_tensor, offset = self._decode_index(idx)
        x_original = year_tensor[:, offset:(offset + self.window_size)]
        y = year_tensor[:, offset + self.window_size + (self.lead_time - 1), 0]
        num_original_nodes = x_original.shape[0]

        placeholder_ghost_features = []
        if self.ghost_node_map:
            for ghost_idx in sorted(self.ghost_node_map.keys()):
                initial_node_idx = self.ghost_node_map[ghost_idx]
                placeholder_ghost_features.append(x_original[initial_node_idx])
            x_final = torch.cat([x_original, torch.stack(placeholder_ghost_features)], dim=0)
        else:
            x_final = x_original

        concat_ghost_features_list = []
        if self.ghost_node_map:
            for ghost_idx in sorted(self.ghost_node_map.keys()):
                initial_node_idx = self.ghost_node_map[ghost_idx]
                mask = (self.original_edge_index[0] == initial_node_idx)
                downstream_node_idx = self.original_edge_index[1, mask][0].item() if mask.any() else initial_node_idx
                x_initial, x_downstream = x_original[initial_node_idx].clone(), x_original[downstream_node_idx].clone()
                if self.use_feature_extrapolation:
                    if x_initial.shape[0] >= 2 and x_downstream.shape[0] >= 2:
                        last_step_i, second_last_step_i = x_initial[-1, :], x_initial[-2, :]
                        step_diff_i = last_step_i - second_last_step_i
                        future_steps_i = torch.stack(
                            [last_step_i + (i + 1) * step_diff_i for i in range(self.lead_time)])
                        x_initial = torch.cat([x_initial[self.lead_time:], future_steps_i], dim=0)
                        last_step_d, second_last_step_d = x_downstream[-1, :], x_downstream[-2, :]
                        step_diff_d = last_step_d - second_last_step_d
                        future_steps_d = torch.stack(
                            [last_step_d + (i + 1) * step_diff_d for i in range(self.lead_time)])
                        x_downstream = torch.cat([x_downstream[self.lead_time:], future_steps_d], dim=0)
                x_concat_flat = torch.cat([x_initial.flatten(), x_downstream.flatten()])
                concat_ghost_features_list.append(x_concat_flat)
            x_ghost_concat = torch.stack(concat_ghost_features_list)
        else:
            x_ghost_concat = torch.empty((0, x_final.shape[1] * x_final.shape[2] * 2), dtype=x_final.dtype)

        num_total_nodes = x_final.shape[0]
        real_node_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
        real_node_mask[:num_original_nodes] = True
        ghost_node_mask = ~real_node_mask

        return Data(x=x_final, y=y.unsqueeze(-1),
                    edge_index=self.edge_index_with_ghosts, edge_attr=self.edge_attr_with_ghosts,
                    mask=real_node_mask, ghost_mask=ghost_node_mask, x_ghost_concat=x_ghost_concat)

    def _decode_index(self, idx):
        for i, size in enumerate(self.year_sizes):
            idx -= size
            if idx < 0:
                return self.year_tensors[i], self.stride_length * (idx + size)
        raise AssertionError("Corrupt internal state. This should never happen!")

    def normalize(self, x):
        return (x - self.mean[:, None, :]) / (self.std[:, None, :] + 1e-9)

    def denormalize(self, x):
        return self.std[:, None, :] * x + self.mean[:, None, :]

    def longest_path(self):

        adj = {i: [] for i in range(len(self.gauges))}
        if self.original_edge_index.numel() > 0:
            for i in range(self.original_edge_index.size(1)):
                u, v = self.original_edge_index[:, i].tolist()
                adj[u].append(v)

        memo = {}

        def get_longest_path_from(u):
            if u in memo: return memo[u]
            if u not in adj or not adj[u]: return 0
            max_len = max((1 + get_longest_path_from(v) for v in adj[u]), default=0)
            memo[u] = max_len
            return max_len

        if not self.gauges: return 0
        return max((get_longest_path_from(i) for i in range(len(self.gauges))), default=0)


    def build_dense_connections(self):

        print("Building dense connections with cumulative attributes...")
        if self.original_edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float)

        adj = {i: [] for i in range(len(self.gauges))}
        for i in range(self.original_edge_index.shape[1]):
            u, v = self.original_edge_index[:, i].tolist()
            adj[u].append((v, self.original_edge_attr[i].tolist()))

        memo = {}

        def get_all_downstream(u):
            if u in memo: return memo[u]
            downstream = {}
            if u not in adj:
                memo[u] = {}
                return {}
            for v, attr_uv in adj[u]:
                if v not in downstream: downstream[v] = attr_uv
                recursive_downstream = get_all_downstream(v)
                for w, attr_vw in recursive_downstream.items():
                    new_dist = attr_uv[0] + attr_vw[0]
                    new_elev = attr_uv[1] + attr_vw[1]
                    new_slope = new_elev / new_dist if new_dist > 1e-9 else 0.0
                    new_attr = [new_dist, new_elev, new_slope]
                    if w not in downstream or new_dist < downstream[w][0]:
                        downstream[w] = new_attr
            memo[u] = downstream
            return downstream

        all_dense_edges, all_dense_attrs = [], []
        for start_node in tqdm(range(len(self.gauges)), desc="Building Dense Edges"):
            downstream_nodes = get_all_downstream(start_node)
            for end_node, attr in downstream_nodes.items():
                all_dense_edges.append([start_node, end_node])
                all_dense_attrs.append(attr)

        if not all_dense_edges:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float)

        return torch.tensor(all_dense_edges, dtype=torch.long).t(), torch.tensor(all_dense_attrs, dtype=torch.float)

    def to_dense(self):

        print("Converting dataset to use dense connections...")
        dense_edge_index, dense_edge_attr = self.build_dense_connections()

        self.original_edge_index = dense_edge_index
        self.original_edge_attr = dense_edge_attr
        print(f"New dense physical graph has {self.original_edge_index.shape[1]} edges.")

        print("Re-creating ghost graph structure for the new dense graph...")
        self.edge_index_with_ghosts, self.edge_attr_with_ghosts, self.ghost_node_map = self.create_ghost_graph_structure()
        print(f"Final graph (with ghosts) has {self.edge_index_with_ghosts.shape[1]} edges.")

        return self


# =================================================================
#  TEST CODE BLOCK
# =================================================================
if __name__ == '__main__':
    DATASET_PATH = "./LamaH-CE"
    print(f"--- Starting Test ---")
    print(f"Dataset will be stored in: {DATASET_PATH}")
    print("NOTE: First run will download and process several GB of data, which will take a long time.")

    try:
        print("\n--- Test Case 1: Standard graph structure ---")
        dataset = LamaHDataset(DATASET_PATH, years=[2000])

        if len(dataset) > 0:
            print(f"Dataset loaded successfully. Samples: {len(dataset)}")
            print(f"Original physical edges: {dataset.original_edge_index.shape[1]}")
            print(f"Total edges with ghosts: {dataset.edge_index_with_ghosts.shape[1]}")
            print(f"Longest path in original graph: {dataset.longest_path()}")

            sample_original = dataset[0]
            print("\nSample from original graph:")
            print(sample_original)

            print("\n--- Test Case 2: Converting to dense graph ---")
            dataset.to_dense()
            print(f"Dense physical edges: {dataset.original_edge_index.shape[1]}")
            print(f"Total edges with ghosts (after dense): {dataset.edge_index_with_ghosts.shape[1]}")

            print("\n--- Test Case 3: Getting a sample from dense graph ---")
            sample_dense = dataset[0]
            print("First sample from DENSE dataset:")
            print(sample_dense)
            print(f"Shape of x_ghost_concat from dense graph: {sample_dense.x_ghost_concat.shape}")

        else:
            print("Dataset loaded but contains 0 samples.")

    except Exception as e:
        print(f"\nAN ERROR OCCURRED DURING TESTING: {e}")
        import traceback

        traceback.print_exc()