"""
PyTorch integration for HSH dataset (.h5 structured deep drawing simulations).

Each .h5 file contains four tool parts (binder, blank, die, punch)
with node coordinates and element connectivity, plus Attributes
describing geometry parameters (radii, delta, cr, height, etc.).
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import logging
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HSHDataset(Dataset):
    """
    PyTorch-compatible dataset for HSH deep drawing simulation files.

    Structure per .h5 file:
        binder/
            node_coordinates            (N, 3)
            element_shell_node_indexes  (M, 4)
        blank/Tiefgezogenes Bauteil_*/  (multiple stages)
            node_coordinates
            element_shell_node_ids
            element_shell_thickness
        die/
            node_coordinates
            element_shell_node_indexes
        punch/
            node_coordinates
            element_shell_node_indexes
        attrs:
            Parameters = {radii1, radii2, cr, delta, height}
            source_tag = "tool_radii2_20_radii1_5_cr_1.1_delta_0_height_25"

    Returns:
        (int, dict, dict)
        → (simulation_id, data_dict, attributes)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        h5_subdir: str = "data",
        metadata_file: str = "metadata.csv",
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self._h5_dir = self.data_dir / h5_subdir
        self._metadata_path = self.data_dir / metadata_file
        self.transform = transform

        if not self._h5_dir.exists():
            raise FileNotFoundError(f"H5 directory not found: {self._h5_dir}")
        if not self._metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self._metadata_path}")

        # 读取元数据
        self._metadata = pd.read_csv(self._metadata_path)
        self._metadata = self._filter_existing_files()

    def _filter_existing_files(self) -> pd.DataFrame:
        """仅保留存在对应 .h5 文件的样本"""
        mask = self._metadata["id"].apply(
            lambda sid: (self._h5_dir / f"{int(sid)}.h5").exists()
        )
        filtered = self._metadata[mask]
        n_total, n_valid = len(self._metadata), len(filtered)
        if n_valid < n_total:
            logger.warning(f"Found {n_valid}/{n_total} valid .h5 files.")
        return filtered

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, idx: int) -> Tuple[int, Dict[str, dict], dict]:
        """读取一个样本，返回 (id, data_dict, attributes)"""
        row = self._metadata.iloc[idx]
        sim_id = int(row["id"])
        h5_path = self._h5_dir / f"{sim_id}.h5"

        with h5py.File(h5_path, "r") as f:
            # 1️⃣ Attributes
            attrs = {
                k: v.decode() if isinstance(v, bytes) else v
                for k, v in f.attrs.items()
            }

            # 2️⃣ 读取四个主要组
            data = {}
            for part in ["binder", "die", "punch"]:
                group = f[part]
                data[part] = {
                    "nodes": np.array(group["node_coordinates"]),
                    "elements": np.array(group["element_shell_node_indexes"]),
                }

            # 3️⃣ 读取 blank 下的所有阶段
            blank_group = f["blank"]
            blank_stages = {}
            for stage_name in blank_group.keys():
                g = blank_group[stage_name]
                blank_stages[stage_name] = {
                    "nodes": np.array(g["node_coordinates"]),
                    "elements": np.array(g["element_shell_node_ids"]),
                    "thickness": np.array(g["element_shell_thickness"]),
                }
            data["blank"] = blank_stages

        if self.transform:
            data = self.transform(data)

        return sim_id, data, attrs

    def __str__(self) -> str:
        lines = [
            "HSH Dataset (PyTorch)",
            f"  Directory: {self.data_dir}",
            f"  Samples: {len(self)}",
            f"  Example file: {self._h5_dir}/0.h5",
        ]
        return "\n".join(lines)
