# import sys, os, io, logging
# os.makedirs("logs", exist_ok=True)
#
# class Tee(io.TextIOBase):
#     def __init__(self, path, mode="w"):
#         super().__init__()
#         self.file = open(path, mode, encoding="utf-8", buffering=1)
#         self.stdout = sys.__stdout__
#     def write(self, s):
#         self.stdout.write(s)
#         self.file.write(s)
#     def flush(self):
#         self.stdout.flush()
#         self.file.flush()
#
# sys.stdout = sys.stderr = Tee("logs/train_full_stdout.log")
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s: %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)]
# )


# import functions_diri as functions
import torch
# import functions
# import functions_split_2 as functions
import functions_ext1 as functions
# import functions_all as functions  # all has ghost
# import functionsplit1 as functions

hparams = {
    "data": {
        "root_gauge_id": 399, # 558, #399
        "rewire_graph": True,
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 6,
        "normalized": True,
    },
    "model": {
        "architecture": None,  # set below
        "num_layers": None,  # set below
        "hidden_channels": 128,
        "param_sharing": False,
        "edge_orientation": None,  # set below
        "adjacency_type": None,  # set below
    },
    "training": {
        "num_epochs": 200,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "random_seed": 748,
        "train_years": None,  # set below
        "holdout_size": 1/5,
    }
}

DATASET_PATH = "./LamaH-CE"
CHECKPOINT_PATH = "./checkpoint"
if __name__ == '__main__':
    for fold_id, (train_years, test_years) in enumerate([(list(range(2000, 2016, 2)), [2016, 2017])]):
        for num_layers in range(8, 9, 1):
            for architecture in ["ResGAT"]:  # "ResTAG", "ResCheb", "ResGEN", "ResGIN"
            # for architecture in ["ResTAG", "ResCheb", "ResGEN", "ResGIN"]: #
                for edge_orientation in ["bidirectional", "downstream"]: #"downstream","upstream","bidirectional"
            # for edge_orientation in ["downstream"]:
                  # , ResGraphWavelet ResGCN "ResGAT GCNII" ResGraphSAGE CustomMPNN, ResRWGCN "ResGAT", "GCNII" "CustomMPNN"
                    for adjacency_type in ["average_slope"]:
                    # for adjacency_type in ["binary", "stream_length", "elevation_difference",
                    #                        "average_slope" ]:
                    # for adjacency_type in ["isolated", "binary", "stream_length", "elevation_difference", "average_slope"  if architecture == "ResGAT" else "learned"]:
                        hparams["training"]["train_years"] = train_years
                        dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")
                        dataset.to_dense()
                        # dataset = torch.load('cached_dataset.pt')
                        hparams["model"]["architecture"] = architecture
                        hparams["model"]["edge_orientation"] = edge_orientation
                        hparams["model"]["adjacency_type"] = adjacency_type
                        # hparams["model"]["num_layers"] = dataset.longest_path()
                        hparams["model"]["num_layers"] = num_layers
                        functions.ensure_reproducibility(hparams["training"]["random_seed"])

                        print(hparams["model"]["num_layers"], "layers used")
                        model = functions.construct_model(hparams, dataset)
                        history = functions.train(model, dataset, hparams)

                        chkpt_name = f"{architecture}_{edge_orientation}_{adjacency_type}_{fold_id}.run"
                        functions.save_checkpoint(history, hparams, chkpt_name, directory=CHECKPOINT_PATH)

# import functions_diri as functions
# import functions
#
# hparams = {
#     "data": {
#         "root_gauge_id": 399,
#         "rewire_graph": True,
#         "window_size": 24,
#         "stride_length": 1,
#         "lead_time": 6,
#         "normalized": True,
#     },
#     "model": {
#         "architecture": None,  # set below
#         "num_layers": None,  # set below
#         "hidden_channels": 128,
#         "param_sharing": False,
#         "edge_orientation": None,  # set below
#         "adjacency_type": None,  # set below
#     },
#     "training": {
#         "num_epochs": 200,
#         "batch_size": 64,
#         "learning_rate": 1e-4,
#         "weight_decay": 1e-5,
#         "random_seed": 42,
#         "train_years": None,  # set below
#         "holdout_size": 1/5,
#     }
# }
#
# DATASET_PATH = "./LamaH-CE"
# CHECKPOINT_PATH = "./checkpoint"
#
# for fold_id, (train_years, test_years) in enumerate([(list(range(2000, 2016, 2)), [2016, 2017])]):
#     for architecture in ["ResGCN", "GCNII", "ResGAT"]:
#         for num_layers in range(1, 31, 2):
#             for edge_orientation in ["downstream", "upstream", "bidirectional"]:
#                 for adjacency_type in ["average_slope"]:
#
#                     hparams["training"]["train_years"] = train_years
#                     dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")
#
#                     hparams["model"]["architecture"] = architecture
#                     hparams["model"]["edge_orientation"] = edge_orientation
#                     hparams["model"]["adjacency_type"] = adjacency_type
#                     hparams["model"]["num_layers"] = num_layers
#                     functions.ensure_reproducibility(hparams["training"]["random_seed"])
#
#                     print(f"{num_layers} layers used")
#                     model = functions.construct_model(hparams, dataset)
#                     history = functions.train(model, dataset, hparams)
#
#                     chkpt_name = f"{architecture}_{edge_orientation}_{adjacency_type}_{num_layers}_layers_fold_{fold_id}.run"
#                     functions.save_checkpoint(history, hparams, chkpt_name, directory=CHECKPOINT_PATH)
