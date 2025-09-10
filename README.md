# XAI Medical Segmentation (Swin UNETR + CLIPSeg)

Multi-modal medical image segmentation framework, with dataset-aware prompt engineering, efficient data pipelines (2D slice or 3D volume), persistent caching, mixed-precision finetuning, and task-vector model editing for adaptation analysis. The main interactive workflow lives in `local.ipynb`.

## TL;DR
Train or adapt Swin-UNETR / CLIPSeg on CHAOS (CT/MR) or MMWHS datasets, generate metrics (Dice / Hausdorff), visualize predictions, blend model deltas via task vectors, and aggregate experiment JSON logs to CSV for analysis.

---
## Core Notebook: `local.ipynb`
The notebook orchestrates:
1. Environment & imports
2. Dataset selection (CHAOS / MMWHS, domain & 2D vs 3D mode)
3. Model construction (`MedicalSegmenter` with `encoder_type="swin_unetr" | "clipseg"`)
4. (Optional) Loading pretrained or finetuned checkpoint
5. Finetuning loop (warmup + cosine LR, mixed precision)
6. Validation metrics logging to JSON in `outputs/`
7. Visualization of sample slices / volumes and overlays
8. (Optional) Task vector creation and arithmetic between baseline / finetuned checkpoints

You can re-run cells selectively to compare adaptation strategies (e.g., different alpha scaling for task vectors) and then consolidate results via `outputs/make_csv.py`.

---
## Datasets

### CHAOS
Supports CT and MR. MRI optionally filtered to liver-only. Loader scans both train & test sets, retaining only samples with existing masks; splits into train/val/test with user-set ratios (default 70/15/15).

Labels (MR full): Background, Liver, Right kidney, Left kidney, Spleen.  
Labels (CT): Background, Liver.

### MMWHS (Multi-Modality Whole Heart Segmentation)
Maps multi-class anatomical structures to contiguous class indices (see `mmwhs.py`).

### 2D vs 3D
Set `slice_2d=True` to convert volumetric DICOM stacks into individual slices, or `False` for full 3D volumes with Swin-UNETR.

---
## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Ensure a CUDA-enabled PyTorch build is installed (adjust `torch` install if needed).

---
## Visualization
Use `visualize_sample_predictions.py` or notebook cells to overlay predictions:
```bash
python visualize_sample_predictions.py --checkpoint checkpoints/CHAOS_CT_2d_finetuned.pth --dataset CHAOS --domain CT
```
Figures stored in `outputs/figures/` (e.g., `adaptation_4panel.png`).

---
## Caching & Performance Tips
* Adjust `num_workers` based on I/O.


---
## References
MM-WHS:
1. X. Zhuang. Multivariate mixture model for myocardial segmentation combining multi-source images. TPAMI 41(12):2933–2946, 2019.  
2. X. Zhuang & J. Shen. Multi-scale patch and multi-modality atlases for whole heart segmentation of MRI. MedIA 31:77–87, 2016.  
3. F. Wu & X. Zhuang. Minimizing Estimated Risks on Unlabeled Data... TPAMI 45(5):6021–6036, 2023.  
4. S. Gao et al. BayeSeg: Bayesian Modeling for Medical Image Segmentation. MedIA 89:102889, 2023.  

Swin-UNETR & MONAI:
* Hatamizadeh et al., Swin UNETR: Transformers for 3D medical image segmentation (arXiv:2201.01266)

CLIP / CLIPSeg:
* Radford et al. Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)
* Lüddecke & Ecker. Image Segmentation Using Text and Image Prompts (CVPR 2022 – CLIPSeg)

Task Vectors:
* Ilharco et al. Editing Models with Task Arithmetic (ICLR 2023)

---
## License & Acknowledgments
This project builds upon MONAI, CLIPSeg and published medical imaging datasets. Ensure you have permission & proper dataset licensing before use. All third-party assets retain their original licenses.

---
## Contact
For questions or improvements, open an issue or submit a PR.