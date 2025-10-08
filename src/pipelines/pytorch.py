# deep_draw_sim_dataset.py
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass
class MappingConfig:
    """
    Configuration for dataset mapping between CSV and H5 files.
    """
    mapping_csv: Union[str, Path]
    data_dir: Union[str, Path]
    id_column: str = "id"              # e.g., "id" / "new_id"
    path_column: Optional[str] = None  # e.g., "path" / "filepath" / "h5_path"
    zero_pad: int = 3                  # zero-padding digits, e.g., 001.h5
    stage_prefix: str = "Tiefgezogenes Bauteil_"


class DeepDrawSimDataset:
    """
    A helper class for working with the Deep Drawing Simulation (DDACS/HSH) dataset.

    Features:
    - Select specific H5 files using a mapping CSV (e.g., mapping_01.csv).
    - Read H5 file groups (blank/die/punch/binder).
    - Extract faces (E, 4), root attributes, and other metadata.
    - Provide visualization utilities:
        * visualize_blank(): visualize a blank part at a specific stage.
        * visualize_tool(): visualize binder/die/punch (combined).
    """

    def __init__(self, cfg: MappingConfig):
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.rows: List[Dict[str, str]] = self._load_csv(cfg.mapping_csv)

        # Auto-detect the path column if not explicitly provided
        if self.cfg.path_column is None:
            self.cfg.path_column = self._auto_detect_path_column(self.rows)

        # Sanity check: ensure at least one row can be resolved
        if not self.rows:
            raise ValueError("CSV is empty — no rows loaded.")
        _ = self._resolve_h5_path(self.rows[0])  # fail early if broken

    # --------------- CSV / path handling -----------------

    def _load_csv(self, csv_path: Union[str, Path]) -> List[Dict[str, str]]:
        """Load a CSV into a list of dictionaries."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        out: List[Dict[str, str]] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append({k.strip(): v.strip() for k, v in row.items()})
        return out

    def _auto_detect_path_column(self, rows: List[Dict[str, str]]) -> Optional[str]:
        """Try to detect which column contains the file path."""
        candidates = ["path", "filepath", "h5_path", "file", "filename", "rel_path"]
        if not rows:
            return None
        cols = rows[0].keys()
        for c in candidates:
            if c in cols:
                return c
        return None  # fallback: derive from id_column + zero_pad

    def _resolve_h5_path(self, row: Dict[str, str]) -> Path:
        """
        Resolve the absolute path to an H5 file.
        Priority:
          1) Use path_column if provided (absolute or relative to data_dir).
          2) Otherwise, generate using id_column + zero_pad → data_dir/001.h5
        """
        # 1) Direct path from CSV
        if self.cfg.path_column and row.get(self.cfg.path_column):
            p = Path(row[self.cfg.path_column])
            if not p.is_absolute():
                p = self.data_dir / p
            return p

        # 2) Derive from ID
        if self.cfg.id_column not in row:
            raise KeyError(
                f"Missing id column '{self.cfg.id_column}' in CSV, and no path_column provided."
            )
        try:
            num = int(row[self.cfg.id_column])
        except Exception:
            s = row[self.cfg.id_column]
            if s.isdigit():
                num = int(s)
            else:
                raise ValueError(f"Cannot parse integer from id column: {s}")

        fname = f"{num:0{self.cfg.zero_pad}d}.h5"
        return self.data_dir / fname

    # --------------- File-level API -----------------

    def get_root_attrs(self, h5_file: Union[str, Path]) -> Dict[str, Any]:
        """Return all root-level attributes from the H5 file as a dictionary."""
        h5_file = Path(h5_file)
        attrs: Dict[str, Any] = {}
        with h5py.File(h5_file, "r") as f:
            for k, v in f.attrs.items():
                if isinstance(v, bytes):
                    try:
                        attrs[k] = v.decode("utf-8")
                    except Exception:
                        attrs[k] = v
                else:
                    attrs[k] = v
        return attrs

    def list_blank_stages(self, h5_file: Union[str, Path]) -> List[str]:
        """List all available blank stages (e.g., 'Tiefgezogenes Bauteil_30000'), sorted by numeric value."""
        h5_file = Path(h5_file)
        stages: List[str] = []
        with h5py.File(h5_file, "r") as f:
            if "blank" not in f:
                return stages
            for k in f["blank"].keys():
                if isinstance(f["blank"][k], h5py.Group):
                    stages.append(k)

        def key_fn(name: str) -> Tuple[int, str]:
            try:
                return (int(name.split("_")[-1]), name)
            except Exception:
                return (0, name)

        stages.sort(key=key_fn)
        return stages

    def get_faces(
        self,
        h5_file: Union[str, Path],
        part: str,
        stage: Optional[Union[str, int]] = None,
    ) -> np.ndarray:
        """
        Return a numpy array of quadrilateral face indices (E, 4), 0-based.

        - blank: requires stage; int → will automatically append stage_prefix.
        - binder/die/punch: tries 'element_shell_node_indexes' first, otherwise builds id→index mapping.
        """
        mesh = self.get_mesh(h5_file, part=part, stage=stage)
        return mesh["faces"]

    def get_mesh(
        self,
        h5_file: Union[str, Path],
        part: str,
        stage: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Return a complete mesh dictionary:
          {
            "pos": (N,3) float64,
            "faces": (E,4) int64,
            "thickness": (E,) float64 or None,
            "meta": {"file", "part", "stage"},
            "attrs": root-level attributes
          }
        """
        h5_file = Path(h5_file)
        part = part.lower()
        if part not in {"blank", "binder", "die", "punch"}:
            raise ValueError("part must be one of: 'blank', 'binder', 'die', 'punch'")

        # Determine stage
        stage_name: Optional[str] = None
        if part == "blank":
            if stage is None:
                raise ValueError("part='blank' requires a stage (int or str).")
            stage_name = f"{self.cfg.stage_prefix}{stage}" if isinstance(stage, int) else str(stage)

        with h5py.File(h5_file, "r") as f:
            # Select group
            grp = None
            is_blank = (part == "blank")
            if is_blank:
                grp_path = f"blank/{stage_name}"
                if grp_path not in f:
                    raise KeyError(f"Missing group '{grp_path}'")
                grp = f[grp_path]
            else:
                if part not in f:
                    raise KeyError(f"Missing group '{part}'")
                grp = f[part]

            pos = np.asarray(grp["node_coordinates"], dtype=np.float64)
            node_ids = np.asarray(grp["node_ids"], dtype=np.int64)

            thickness = None
            faces_idx = None

            # 1) Prefer precomputed 0-based indices
            if not is_blank and "element_shell_node_indexes" in grp:
                faces_idx = np.asarray(grp["element_shell_node_indexes"], dtype=np.int64)
            else:
                # 2) Build from node IDs (blank always uses this)
                if "element_shell_node_ids" not in grp:
                    raise KeyError("Missing 'element_shell_node_indexes' or 'element_shell_node_ids'.")
                esn_ids = np.asarray(grp["element_shell_node_ids"], dtype=np.int64)  # (E, 4)
                id2idx = {int(nid): i for i, nid in enumerate(node_ids.tolist())}
                E = esn_ids.shape[0]
                faces_idx = np.empty_like(esn_ids, dtype=np.int64)
                for e in range(E):
                    for j in range(4):
                        nid = int(esn_ids[e, j])
                        if nid not in id2idx:
                            raise KeyError(f"node id {nid} not found in node_ids.")
                        faces_idx[e, j] = id2idx[nid]

            # Thickness (only for blank)
            if is_blank and "element_shell_thickness" in grp:
                thickness = np.asarray(grp["element_shell_thickness"], dtype=np.float64)

            # Root attributes
            root_attrs = {}
            for k, v in f.attrs.items():
                if isinstance(v, bytes):
                    try:
                        root_attrs[k] = v.decode("utf-8")
                    except Exception:
                        root_attrs[k] = v
                else:
                    root_attrs[k] = v

            return {
                "pos": pos,                         # (N, 3) float64
                "faces": faces_idx,                 # (E, 4) int64
                "thickness": thickness,             # (E,) or None
                "meta": {
                    "file": str(h5_file),
                    "part": part,
                    "stage": stage_name,
                },
                "attrs": root_attrs,
            }

    # --------------- Selection API (based on CSV) -----------------

    def select_files(self, **filters: Any) -> List[Path]:
        """
        Filter H5 files using CSV metadata.
        Examples:
          ds.select_files(id=1)                     → [<.../001.h5>]
          ds.select_files(new_id="7")               → [...]
          ds.select_files(orig_sim_id="tool_...")   → [...]
          ds.select_files(radii1="5.0", cr="1.1")   → supports multi-key filtering
        Returns a deduplicated list of existing H5 paths.
        """
        matched: List[Path] = []
        for row in self.rows:
            ok = True
            for k, v in filters.items():
                if str(row.get(k, "")).strip() != str(v).strip():
                    ok = False
                    break
            if ok:
                p = self._resolve_h5_path(row)
                if p.exists():
                    matched.append(p)
        # Deduplicate
        uniq: List[Path] = []
        seen = set()
        for p in matched:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq

# ---------- Utility: Equal 3D aspect ratio ----------

    def _set_equal_aspect_3d(ax, xyz: np.ndarray):
        """Ensure equal aspect ratio for 3D axes."""
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
        Xb = 0.5 * max_range
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - Xb, mid_x + Xb)
        ax.set_ylim(mid_y - Xb, mid_y + Xb)
        ax.set_zlim(mid_z - Xb, mid_z + Xb)

    # --------------- Visualization -----------------

    def _build_quads(self, pos: np.ndarray, faces: np.ndarray, max_faces: int) -> List[List[np.ndarray]]:
        """Build a list of quadrilateral faces limited by max_faces."""
        E = min(len(faces), max_faces)
        quads: List[List[np.ndarray]] = []
        for e in range(E):
            i0, i1, i2, i3 = faces[e]
            quads.append([pos[i0], pos[i1], pos[i2], pos[i3]])
        return quads

    def visualize_blank(
        self,
        h5_file,
        stage,
        max_faces: int = 10000,
        title: str | None = None,
        use_thickness: bool = True,
    ):
        """
        Visualize the blank mesh at a specific stage.

        - Color represents thickness (if available).
        - Limited to max_faces for performance.
        """
        from pathlib import Path
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh = self.get_mesh(h5_file, part="blank", stage=stage)
        pos, faces, th = mesh["pos"], mesh["faces"], mesh["thickness"]

        E = min(len(faces), max_faces)
        faces = faces[:E]
        th_clip = None if th is None else th[:E]

        quads = [[pos[i0], pos[i1], pos[i2], pos[i3]] for (i0, i1, i2, i3) in faces]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title or f"blank: {Path(h5_file).name} | {mesh['meta']['stage']}")

        poly = Poly3DCollection(quads, linewidths=0.2, edgecolors="k", alpha=1.0)

        if use_thickness and th_clip is not None:
            poly.set_array(th_clip.astype(float))
            ax.add_collection3d(poly)
            fig.colorbar(poly, ax=ax, fraction=0.02, pad=0.04)
        else:
            ax.add_collection3d(poly)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # self._set_equal_aspect_3d(ax, pos)
        plt.tight_layout()
        plt.show()

    def visualize_tool(
        self,
        h5_file,
        part: str,
        max_faces: int = 10000,
        title: str | None = None,
    ):
        """
        Visualize a tooling part: binder / die / punch.
        """
        from pathlib import Path
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        part = part.lower()
        if part not in {"binder", "die", "punch"}:
            raise ValueError("part must be one of: binder / die / punch")

        mesh = self.get_mesh(h5_file, part=part)
        pos, faces = mesh["pos"], mesh["faces"]

        E = min(len(faces), max_faces)
        faces = faces[:E]
        quads = [[pos[i0], pos[i1], pos[i2], pos[i3]] for (i0, i1, i2, i3) in faces]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title or f"{part}: {Path(h5_file).name}")

        poly = Poly3DCollection(quads, linewidths=0.2, edgecolors="k", alpha=1.0)
        ax.add_collection3d(poly)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # self._set_equal_aspect_3d(ax, pos)
        plt.tight_layout()
        plt.show()
