# 🧠 DeepDrawing Dataset  
*A Data-Driven Dataset for Automated Deep Drawing Tool Design*

This repository provides a dataset for **data-driven tool surface generation** in deep drawing,  
based on the publication:

> **M. Hohmann, A. Yiming, L. Penter, S. Ihlenfeldt, O. Niggemann**  
> *A Data-Driven Approach for Automating the Design Process of Deep Drawing Tools*  
> *Journal of Physics: Conference Series*, Vol. 3104, 012061 (2025).  
> DOI: [10.1088/1742-6596/3104/1/012061](https://doi.org/10.1088/1742-6596/3104/1/012061)
![Example under different pressure](data/processed/animation.gif)
> *Example: Comparison of the deformation behavior of the blank under different blankholder forces in the deep drawing simulation.*

---

## 📖 Overview

This dataset enables research on **automated tool design** using **generative neural networks**.  
It provides both **deep drawn part geometries** (input) and **active tool surfaces** (output)  
for **dies** and **punches**, along with the associated process parameters.

| Component                | Description                                                                                                      | Function                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Blank**                | The *workpiece* or *sheet metal* before forming. It is usually a flat circular or rectangular plate.             | Serves as the input material that will be plastically deformed into the desired shape. |
| **Binder (Blankholder)** | A *clamping tool* that presses the blank’s edges against the die using a controlled **Blankholder Force (FBH)**. | Prevents wrinkling and regulates the material flow into the die cavity.                |
| **Die**                  | The *female* part of the forming tool set, containing the cavity that defines the outer contour of the product.  | Provides the external surface geometry and supports the blank during forming.          |
| **Punch**                | The *male* part of the forming tool set that moves downward into the die cavity.                                 | Forms the internal contour of the part by pushing the blank into the die.              |

---

## ⚙️ Data Generation Pipeline

Data was generated through an **automated FE-simulation workflow** combining  
[Gmsh](https://gmsh.info/) and **LS-Dyna**, with custom Python scripts for preprocessing.

**Workflow Summary**
1. Parameterize tool geometry (R1, R2, h, c, α).
2. Generate and mesh geometries using Gmsh.
3. Run LS-Dyna forming simulations with varying `FBH`.
4. Extract part meshes and tool surfaces.
5. Convert results into `.xls`, `.pt`, or `.h5` for ML training.



---

## 📊 Parameter Overview

| Parameter | Symbol | Range | Unit | Description |
|------------|---------|--------|------|-------------|
| Corner radius | R1 | 5 – 8 | mm | Outer corner of die/punch |
| Fillet radius | R2 | 20 – 55 | mm | Transition curvature |
| Drawing depth | h | 25 – 50 | mm | Height of part cavity |
| Clearance | c | 1.1 – 1.4 | mm | Gap between die & punch |
| Bevel angle | α | 0 – 10 | ° | Conical wall inclination |
| Blankholder force | FBH | 15 – 40 | kN | Process pressure |


---

## 🧩 Data Format

Each dataset sample consists of:

```python
{
  HDF5 File Overview
============================================================
Attributes:
  Parameters = {radii2:20.0, radii1:5.0, delta:0.0, cr:1.1, height:25.0}
  source_tag = tool_radii2_20_radii1_5_cr_1.1_delta_0_height_25

binder/
  node_coordinates            (N, 3)
  element_shell_node_indexes  (M, 4)
  element_shell_ids           (M,)

blank/
  Tiefgezogenes Bauteil_*     
    node_coordinates          (N, 3)
    element_shell_node_ids    (M, 4)
    element_shell_thickness   (M,)

die/
  node_coordinates            (N, 3)
  element_shell_node_indexes  (M, 4)
  element_shell_ids           (M,)

punch/
  node_coordinates            (N, 3)
  element_shell_node_indexes  (M, 4)
  element_shell_ids           (M,)
}