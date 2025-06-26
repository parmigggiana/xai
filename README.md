# MONAI Volumetric Segmentation Project

This project provides a skeleton for volumetric segmentation tasks using the MONAI framework. It includes:

- An interface function to evaluate a segmentation model on 3D medical images.
- Default support for pretrained Swin-UNETR.
- Standard evaluation metrics (Dice and Hausdorff Distance).
- Dataset loaders for CHAOS, MM-WHS, and APIS medical imaging datasets.
- Visualization utilities for 3D medical data.

## Features

✅ **Fixed dimension mismatch errors** - Properly handles 2D/3D model compatibility  
✅ **MONAI-based segmentation** - Uses state-of-the-art Swin-UNETR architecture 
✅ **Dataset-specific classification heads** - Automatically saves/loads heads per dataset  
✅ **Standard medical imaging metrics** - Dice coefficient and Hausdorff distance  
✅ **Multiple dataset support** - CHAOS, MM-WHS, APIS with proper data loaders  
✅ **Visualization tools** - View volumetric slices and segmentation overlays  

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the example:

   ```python
   python example_usage.py
   ```

3. Use the interface function:

   ```python
   from monai_project.interface import evaluate_segmentation_performance
   
   metrics = evaluate_segmentation_performance(
       dataset_name="MyDataset",
       dataloader=my_dataloader,
       save_path="./checkpoints"
   )
   print(metrics)
   ```

## Project Structure

- `monai_project/` - Core MONAI interface and utilities
- `src/datasets/` - Dataset loaders for medical imaging datasets
- `src/head.py` - Segmentation head building and management
- `src/modelseg.py` - Segmentation model wrapper with dimension handling
- `notebooks/` - Jupyter notebooks for dataset visualization
- `example_usage.py` - Complete working example

## Datasets

The project supports the following medical imaging datasets:

- **CHAOS** - CT and MRI liver segmentation
- **MM-WHS** - Multi-modal whole heart segmentation  
- **APIS** - Additional medical imaging dataset

Place datasets in the `data/` directory following the expected structure.

## References

MM-WHS:
[1] Xiahai Zhuang: Multivariate mixture model for myocardial segmentation combining multi-source images. IEEE Transactions on Pattern Analysis and Machine Intelligence 41(12): 2933-2946, 2019. [link](https://ieeexplore.ieee.org/document/8458220/) [code](https://github.com/xzluo97/MvMM-Demo)  
[2] X Zhuang & J Shen: Multi-scale patch and multi-modality atlases for whole heart segmentation of MRI, Medical Image Analysis 31: 77-87, 2016 ([link](http://dx.doi.org/10.1016/j.media.2016.02.006))  
[3] F Wu & X Zhuang. Minimizing Estimated Risks on Unlabeled Data: A New Formulation for Semi-Supervised Medical Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence (T PAMI) 45(5): 6021 - 6036, 2023 [link](https://ieeexplore.ieee.org/document/9921323) [code](https://github.com/FupingWu90/MERU)  
[4] S Gao, H Zhou, Y Gao, X Zhuang. BayeSeg: Bayesian Modeling for Medical Image Segmentation with Interpretable Generalizability. Medical Image Analysis 89, 102889, 2023 [code&tutorial](https://github.com/obiyoag/BayeSeg), [link](https://www.sciencedirect.com/journal/medical-image-analysis/special-issue/10MFST0CK73) (Elsevier-MedIA 1st Prize & Best Paper Award of MICCAl society 2023)
