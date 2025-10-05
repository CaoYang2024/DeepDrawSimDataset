# how to use
from torch.utils.data import DataLoader

ds = HSHDataset("/mnt/data/hsh", h5_subdir="data", metadata_file="metadata.csv")
print(ds)

sim_id, data, attrs = ds[0]
print("ID:", sim_id)
print("Attributes:", attrs["Parameters"])
print("Binder nodes:", data["binder"]["nodes"].shape)
print("Blank stages:", list(data["blank"].keys()))

## Iterating with PyTorch DataLoader
loader = DataLoader(ds, batch_size=1, shuffle=False)
for sid, data, attrs in loader:
    print("Batch:", sid)
    break
---