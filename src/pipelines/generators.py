from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple, Union, Optional, List
import os
import re
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """
    Configuration for locating the dataset and selecting metadata columns.

    Args:
        data_dir: Root directory of the dataset.
        h5_subdir: Subdirectory (under data_dir) containing the H5 files.
        metadata_file: CSV filename (under data_dir) with metadata rows.
        id_column: Column name that identifies a sample; if None, auto-detect.
        feature_columns: Metadata columns to return besides the ID; if None,
            use all non-ID columns.
    """
    data_dir: Union[str, Path] = "/mnt/data/hsh"  # Type annotation is required
    h5_subdir: str = "data_copy"
    metadata_file: str = "mapping_01.csv"
    id_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None


# ==============================
# Utilities
# ==============================
_COMMON_ID_CANDIDATES = [
    "ID", "id", "new_id", "sim_id", "sample_id", "index", "filename", "file"
]


def _autodetect_id_column(df: pd.DataFrame, preferred: Optional[str]) -> str:
    """
    Choose an ID column from the metadata DataFrame.

    Priority:
      1) 'preferred' if present
      2) Common candidate names
      3) First integer-like column
      4) Fallback to the first column with a warning
    """
    if preferred and preferred in df.columns:
        return preferred

    for cand in _COMMON_ID_CANDIDATES:
        if cand in df.columns:
            return cand

    for c in df.columns:
        s = df[c]
        if pd.api.types.is_integer_dtype(s):
            return c
        try:
            # All non-null values numeric-like?
            if s.dropna().astype(str).str.fullmatch(r"\d+").all():
                return c
        except Exception:
            pass

    logger.warning("Could not confidently detect ID column; using the first column.")
    return df.columns[0]


def _load_metadata(cfg: DatasetConfig) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Load the CSV metadata and return (filtered_df, id_col, feature_cols).
    """
    data_dir = Path(cfg.data_dir)
    meta_path = data_dir / cfg.metadata_file
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)
    id_col = _autodetect_id_column(df, cfg.id_column)

    if cfg.feature_columns is None:
        feature_cols = [c for c in df.columns if c != id_col]
    else:
        feature_cols = cfg.feature_columns
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"feature_columns missing in CSV: {missing}")

    # Keep only ID + selected features (preserve order)
    df = df[[id_col] + feature_cols]
    return df, id_col, feature_cols


def _as_sim_id(val) -> int:
    """
    Normalize many ID formats to an integer:

    Accepts: 113, "113", "001.h5", "h5/001.h5", "foo_001.h5"
    Returns: 113
    """
    if val is None:
        raise ValueError("Empty ID value")

    # Fast path for numeric-like values
    try:
        return int(val)
    except (TypeError, ValueError):
        pass

    s = str(val).strip()
    s = os.path.basename(s)          # drop directories
    s = os.path.splitext(s)[0]       # drop extension
    m = re.search(r"(\d+)$", s)      # trailing digits
    if not m:
        raise ValueError(f"Cannot parse numeric ID from: {val!r}")
    return int(m.group(1))


def _row_to_h5_path(row: pd.Series, id_col: str, h5_dir: Path) -> Path:
    """
    Resolve the H5 path for a metadata row.

    If the ID cell already looks like a filename (*.h5), use it under h5_dir.
    Otherwise, build '{ID}.h5' from the numeric ID.
    """
    val = row[id_col]
    if isinstance(val, str) and val.strip().lower().endswith(".h5"):
        fname = os.path.basename(val.strip())
        return h5_dir / fname

    # Otherwise treat as numeric-like and format a simple '{sid}.h5'
    sid = _as_sim_id(val)
    # If your files are zero-padded (e.g., 001.h5), adjust here to f"{sid:03d}.h5"
    return h5_dir / f"{sid}.h5"


# ==============================
# Public API (generators)
# ==============================
def iter_ddacs(
    data_dir: Union[str, Path],
    h5_subdir: str = "data_copy",
    metadata_file: str = "mapping_01.csv",
    id_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
    """
    Stream (sim_id, metadata_values, h5_path) from the CSV + H5 directory.

    Notes:
      - Supports CSV ID cells like 113 / "113" / "001.h5" / "h5/001.h5"
      - h5_path resolves to data_dir/h5_subdir/<filename>.h5

    Raises:
      FileNotFoundError: if the CSV or H5 directory is missing.
    """
    cfg = DatasetConfig(
        data_dir=data_dir,
        h5_subdir=h5_subdir,
        metadata_file=metadata_file,
        id_column=id_column,
        feature_columns=feature_columns,
    )

    data_dir = Path(cfg.data_dir)
    h5_dir = data_dir / cfg.h5_subdir
    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")

    df, id_col, feat_cols = _load_metadata(cfg)

    for _, row in df.iterrows():
        h5_path = _row_to_h5_path(row, id_col, h5_dir)
        if not h5_path.exists():
            # Choose to skip missing files (more robust for batch runs)
            logger.warning(f"H5 file missing, skipping: {h5_path}")
            continue
        sim_id = _as_sim_id(row[id_col])
        meta_vals = row[feat_cols].to_numpy(copy=False)
        yield sim_id, meta_vals, h5_path


def iter_h5_files(
    data_dir: Union[str, Path],
    h5_subdir: str = "h5",
) -> Generator[Path, None, None]:
    """
    Yield each '*.h5' file path under data_dir/h5_subdir.
    """
    h5_dir = Path(data_dir) / h5_subdir
    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")
    for p in h5_dir.glob("*.h5"):
        yield p


def get_simulation_by_id(
    sim_id: Union[int, str],
    data_dir: Union[str, Path],
    h5_subdir: str = "data_copy",
    metadata_file: str = "mapping_01.csv",
    id_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> Optional[Tuple[int, np.ndarray, Path]]:
    """
    Fetch a single sample by its ID or filename-like string.

    Returns:
      (normalized_sim_id, metadata_values, h5_path) or None if not found.
    """
    cfg = DatasetConfig(
        data_dir=data_dir,
        h5_subdir=h5_subdir,
        metadata_file=metadata_file,
        id_column=id_column,
        feature_columns=feature_columns,
    )

    data_dir = Path(cfg.data_dir)
    h5_dir = data_dir / cfg.h5_subdir
    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")

    df, id_col, feat_cols = _load_metadata(cfg)

    target = _as_sim_id(sim_id)
    found_row = None
    for _, row in df.iterrows():
        if _as_sim_id(row[id_col]) == target:
            found_row = row
            break

    if found_row is None:
        return None

    h5_path = _row_to_h5_path(found_row, id_col, h5_dir)
    if not h5_path.exists():
        return None

    meta_vals = found_row[feat_cols].to_numpy(copy=False)
    return target, meta_vals, h5_path


def sample_simulations(
    n: int,
    data_dir: Union[str, Path],
    h5_subdir: str = "data_copy",
    metadata_file: str = "mapping_01.csv",
    id_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
    """
    Randomly sample up to n simulations that have existing H5 files.
    """
    if seed is not None:
        import random
        random.seed(seed)

    cfg = DatasetConfig(
        data_dir=data_dir,
        h5_subdir=h5_subdir,
        metadata_file=metadata_file,
        id_column=id_column,
        feature_columns=feature_columns,
    )

    data_dir = Path(cfg.data_dir)
    h5_dir = data_dir / cfg.h5_subdir
    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")

    df, id_col, feat_cols = _load_metadata(cfg)

    # Collect candidates that actually exist on disk
    items: List[Tuple[int, np.ndarray, Path]] = []
    for _, row in df.iterrows():
        h5_path = _row_to_h5_path(row, id_col, h5_dir)
        if h5_path.exists():
            sid = _as_sim_id(row[id_col])
            meta_vals = row[feat_cols].to_numpy(copy=False)
            items.append((sid, meta_vals, h5_path))

    if not items:
        logger.warning("No simulations with existing H5 files found.")
        return

    import random
    for item in random.sample(items, k=min(n, len(items))):
        yield item


def count_available_simulations(
    data_dir: Union[str, Path] = "/mnt/data/hsh",
    h5_subdir: str = "data_copy",
    metadata_file: str = "mapping_01.csv",
    id_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> int:
    """
    Count how many metadata rows have a corresponding existing H5 file.
    """
    cfg = DatasetConfig(
        data_dir=data_dir,
        h5_subdir=h5_subdir,
        metadata_file=metadata_file,
        id_column=id_column,
        feature_columns=feature_columns,
    )

    data_dir = Path(cfg.data_dir)
    h5_dir = data_dir / cfg.h5_subdir
    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")

    df, id_col, _ = _load_metadata(cfg)
    return sum((_row_to_h5_path(row, id_col, h5_dir)).exists() for _, row in df.iterrows())


# ==============================
# Quick test
# ==============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Adjust these defaults to your dataset location:
    # Example: /mnt/data/hsh/mapping_01.csv and /mnt/data/hsh/data_copy/{ID}.h5
    DATA_DIR = "/mnt/data/hsh"
    H5_SUBDIR = "data_copy"
    META_FILE = "mapping_01.csv"
    ID_COL = None            # Auto-detect (supports ID/new_id/filename/etc.)
    FEAT_COLS = None         # Use all non-ID columns

    try:
        total = count_available_simulations(
            data_dir=DATA_DIR,
            h5_subdir=H5_SUBDIR,
            metadata_file=META_FILE,
            id_column=ID_COL,
            feature_columns=FEAT_COLS,
        )
        print(f"Total available simulations: {total}")

        # Show the first 3 items
        k = 0
        for sim_id, meta, h5_path in iter_ddacs(
            DATA_DIR, H5_SUBDIR, META_FILE, ID_COL, FEAT_COLS
        ):
            print(f"[HEAD] id={sim_id} h5={h5_path.name} meta_shape={meta.shape}")
            k += 1
            if k >= 3:
                break

        # Sample 2 items
        for sim_id, meta, h5_path in sample_simulations(
            2, DATA_DIR, H5_SUBDIR, META_FILE, ID_COL, FEAT_COLS, seed=42
        ):
            print(f"[SAMPLE] id={sim_id} h5={h5_path.name}")

        # Example: precise lookup
        # item = get_simulation_by_id(1, DATA_DIR, H5_SUBDIR, META_FILE, ID_COL, FEAT_COLS)
        # print("GET:", item)

    except Exception as e:
        logger.exception(e)
