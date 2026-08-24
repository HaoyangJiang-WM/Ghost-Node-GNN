from dataset_ext1 import LamaHDataset

# Minimal river-data example. The dataset class downloads LamaH-CE automatically.
dataset = LamaHDataset(
    root_dir="./data",
    years=range(2000, 2018),
    root_gauge_id=399,
    window_size=24,
    lead_time=6,
    normalized=True,
)

sample = dataset[0]

print("x:", sample.x.shape)
print("y:", sample.y.shape)
print("edge_index:", sample.edge_index.shape)
print("real nodes:", int(sample.mask.sum()))
print("ghost nodes:", int(sample.ghost_mask.sum()))
