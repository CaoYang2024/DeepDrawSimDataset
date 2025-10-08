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
    mapping_csv: Union[str, Path]
    data_dir: Union[str, Path]
    id_column: str = "id"              # 比如 "id" / "new_id"
    path_column: Optional[str] = None  # 比如 "path" / "filepath" / "h5_path"
    zero_pad: int = 3                  # 001.h5 的 3 位 zero pad
    stage_prefix: str = "Tiefgezogenes Bauteil_"


class DeepDrawSimDataset:
    """
    支持按 mapping_01.csv 选择特定文件，读取 H5（blank/die/punch/binder），
    取出 faces(E,4)、根属性，并提供可视化（blank 单独，binder/die/punch 合并一个）。
    """

    def __init__(self, cfg: MappingConfig):
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.rows: List[Dict[str, str]] = self._load_csv(cfg.mapping_csv)

        # 自动识别 path 列（如果没显式给）
        if self.cfg.path_column is None:
            self.cfg.path_column = self._auto_detect_path_column(self.rows)

        # 预检查：至少要能解析出一个文件
        if not self.rows:
            raise ValueError("CSV 为空：未读取到任何行。")
        _ = self._resolve_h5_path(self.rows[0])  # 早失败早修

    # --------------- CSV / 路径相关 -----------------

    def _load_csv(self, csv_path: Union[str, Path]) -> List[Dict[str, str]]:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到 CSV：{csv_path}")
        out: List[Dict[str, str]] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append({k.strip(): v.strip() for k, v in row.items()})
        return out

    def _auto_detect_path_column(self, rows: List[Dict[str, str]]) -> Optional[str]:
        candidates = ["path", "filepath", "h5_path", "file", "filename", "rel_path"]
        if not rows:
            return None
        cols = rows[0].keys()
        for c in candidates:
            if c in cols:
                return c
        return None  # 可能需要用 id_column + zero_pad 推断

    def _resolve_h5_path(self, row: Dict[str, str]) -> Path:
        """
        解析出 H5 文件的绝对路径：
          1) 如果 CSV 有 path_column（相对/绝对路径），优先使用
          2) 否则用 id_column + zero_pad 生成形如 data_dir/001.h5
        """
        # 1) 直接路径
        if self.cfg.path_column and row.get(self.cfg.path_column):
            p = Path(row[self.cfg.path_column])
            if not p.is_absolute():
                p = self.data_dir / p
            return p

        # 2) 用 id 拼文件名
        if self.cfg.id_column not in row:
            raise KeyError(
                f"CSV 中缺少 id 列 '{self.cfg.id_column}'，且未提供 path_column。"
            )
        try:
            num = int(row[self.cfg.id_column])
        except Exception:
            # 允许 id 形如 "001" 的字符串
            s = row[self.cfg.id_column]
            if s.isdigit():
                num = int(s)
            else:
                raise ValueError(f"id 列无法解析整数：{s}")

        fname = f"{num:0{self.cfg.zero_pad}d}.h5"
        return self.data_dir / fname

    # --------------- 文件级 API -----------------

    def get_root_attrs(self, h5_file: Union[str, Path]) -> Dict[str, Any]:
        """返回 H5 根属性字典（自动解码 bytes）。"""
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
        """列出 blank 下所有阶段组名（如 'Tiefgezogenes Bauteil_30000'），按数值排序。"""
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
        返回四边形面索引 (E,4) 的 numpy 数组（0-based）。
        - blank：需要 stage；若传入 int，会自动拼接 stage_prefix；若 None 抛错
        - binder/die/punch：优先读取 element_shell_node_indexes；否则回退到 ids→index 映射
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
        返回完整网格字典：
          {
            "pos": (N,3) float64,
            "faces": (E,4) int64,
            "thickness": (E,) float64 或 None,
            "meta": {"file","part","stage"},
            "attrs": 根属性
          }
        """
        h5_file = Path(h5_file)
        part = part.lower()
        if part not in {"blank", "binder", "die", "punch"}:
            raise ValueError("part 必须是 'blank' | 'binder' | 'die' | 'punch'")

        # 解析 stage
        stage_name: Optional[str] = None
        if part == "blank":
            if stage is None:
                raise ValueError("part='blank' 时必须提供 stage（int 或 str）。")
            if isinstance(stage, int):
                stage_name = f"{self.cfg.stage_prefix}{stage}"
            else:
                stage_name = str(stage)

        with h5py.File(h5_file, "r") as f:
            grp = None
            is_blank = (part == "blank")
            if is_blank:
                grp_path = f"blank/{stage_name}"
                if grp_path not in f:
                    raise KeyError(f"缺少组 '{grp_path}'")
                grp = f[grp_path]
            else:
                if part not in f:
                    raise KeyError(f"缺少组 '{part}'")
                grp = f[part]

            pos = np.asarray(grp["node_coordinates"], dtype=np.float64)
            node_ids = np.asarray(grp["node_ids"], dtype=np.int64)

            thickness = None
            faces_idx = None

            # 1) 优先读取现成的 0-based indexes
            if not is_blank and "element_shell_node_indexes" in grp:
                faces_idx = np.asarray(grp["element_shell_node_indexes"], dtype=np.int64)
            else:
                # 2) 用 ids 做映射（blank 必走这里）
                if "element_shell_node_ids" not in grp:
                    raise KeyError("缺少 'element_shell_node_indexes' 或 'element_shell_node_ids'。")
                esn_ids = np.asarray(grp["element_shell_node_ids"], dtype=np.int64)  # (E,4)
                id2idx = {int(nid): i for i, nid in enumerate(node_ids.tolist())}
                E = esn_ids.shape[0]
                faces_idx = np.empty_like(esn_ids, dtype=np.int64)
                for e in range(E):
                    for j in range(4):
                        nid = int(esn_ids[e, j])
                        if nid not in id2idx:
                            raise KeyError(f"node id {nid} 不在 node_ids 中。")
                        faces_idx[e, j] = id2idx[nid]

            # 厚度（仅 blank 有）
            if is_blank and "element_shell_thickness" in grp:
                thickness = np.asarray(grp["element_shell_thickness"], dtype=np.float64)

            # 根属性
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
                "pos": pos,                         # (N,3) float64
                "faces": faces_idx,                 # (E,4) int64
                "thickness": thickness,             # (E,) 或 None
                "meta": {
                    "file": str(h5_file),
                    "part": part,
                    "stage": stage_name,
                },
                "attrs": root_attrs,
            }

    # --------------- 数据选择（基于 CSV） -----------------

    def select_files(self, **filters: Any) -> List[Path]:
        """
        从 CSV 筛选文件。示例：
          ds.select_files(id=1)                     -> [<.../001.h5>]
          ds.select_files(new_id="7")               -> [...]
          ds.select_files(orig_sim_id="tool_...")   -> [...]
          ds.select_files(radii1="5.0", cr="1.1")   -> 支持多条件同时匹配（字符串精确匹配）
        返回 H5 Path 列表（去重且存在）。
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
        # 去重
        uniq: List[Path] = []
        seen = set()
        for p in matched:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq

# ---------- 辅助：3D 轴等比例 ----------
    def _set_equal_aspect_3d(ax, xyz: np.ndarray):
        """
        让 3D 轴按数据范围等比例显示。
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
        Xb = 0.5 * max_range
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - Xb, mid_x + Xb)
        ax.set_ylim(mid_y - Xb, mid_y + Xb)
        ax.set_zlim(mid_z - Xb, mid_z + Xb)
  
    # --------------- 可视化 -----------------

    def _build_quads(self, pos: np.ndarray, faces: np.ndarray, max_faces: int) -> List[List[np.ndarray]]:
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
        from pathlib import Path
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh = self.get_mesh(h5_file, part="blank", stage=stage)
        pos, faces, th = mesh["pos"], mesh["faces"], mesh["thickness"]

        # 同步裁剪
        E = min(len(faces), max_faces)
        faces = faces[:E]
        th_clip = None if th is None else th[:E]

        # 构造四边形面
        quads = [[pos[i0], pos[i1], pos[i2], pos[i3]] for (i0, i1, i2, i3) in faces]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title or f"blank: {Path(h5_file).name} | {mesh['meta']['stage']}")

        poly = Poly3DCollection(quads, linewidths=0.2, edgecolors="k", alpha=1.0)

        if use_thickness and th_clip is not None:
            # ✅ 关键：对 poly（而不是 numpy）设置标量数组
            poly.set_array(th_clip.astype(float))
            ax.add_collection3d(poly)
            # 可选：稳定颜色范围并显示颜色条
            # poly.set_clim(vmin=float(th_clip.min()), vmax=float(th_clip.max()))
            fig.colorbar(poly, ax=ax, fraction=0.02, pad=0.04)
        else:
            ax.add_collection3d(poly)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        #self._set_equal_aspect_3d(ax, pos)
        plt.tight_layout()
        plt.show()


    def visualize_tool(
        self,
        h5_file,
        part: str,
        max_faces: int = 10000,
        title: str | None = None,
    ):
        from pathlib import Path
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        part = part.lower()
        if part not in {"binder", "die", "punch"}:
            raise ValueError("part 必须是 binder/die/punch")

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


