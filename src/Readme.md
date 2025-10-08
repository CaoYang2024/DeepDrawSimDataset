# DeepDrawSimDataset
A lightweight dataset and utility toolkit for *Deep Drawing and Cutting Simulations*.
It provides convenient APIs to map CSV → H5, load meshes (blank / binder / die / punch), manage multi-stage blanks, extract metadata, and visualize results.
---
## Features
### CSV-based file mapping
Select samples using mapping.csv by any column such as id, new_id, orig_sim_id, or parameters like radii1, cr, height, etc.

### Unified mesh interface
- Extract consistent mesh data:

- pos: node coordinates (N, 3)

- faces: quadrilateral faces (E, 4) (0-based)

- thickness: element thickness (E,) (only for blanks)

- attrs: root-level H5 attributes (parameter dictionary)

### Stage management (blank)
Automatically list and sort all stages under the blank/ group (e.g., Tiefgezogenes Bauteil_30000).

Visualization utilities

visualize_blank(): color-coded blank mesh by thickness

visualize_tool(): visualize binder, die, or punch mesh

### Robust handling
Auto-detects CSV path column; if not found, generates filenames from id + zero_pad (e.g., data/001.h5).
---
## Processing Pipeline
The DeepDrawSimDataset class wraps the typical preprocessing and management steps into a clean, ready-to-use API:

1) CSV → File resolution

- Automatically detects path columns (path, file, h5_path, etc.)

- If not found, generates filenames using id + zero_pad (e.g., 001.h5)

2) Multi-stage blank management

- Enumerates and sorts all stage groups under blank/

3) ID → index mapping

- binder/die/punch: use element_shell_node_indexes directly

- blank: reconstruct 0-based indices from node_ids and element_shell_node_ids

4) Metadata aggregation

- Collects root-level attributes and meta info (file, part, stage) into a structured dictionary

5) Quick visualization

- Built-in methods for blank and tool rendering

- Supports color-mapped thickness visualization and max_faces limit for performance

## 🧪 Examples

Complete runnable examples, including visualization and dataset usage, are available in the `examples/` directory.
Start with `examples/dataset_demo.ipynb`.