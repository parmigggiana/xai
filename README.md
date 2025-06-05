# MONAI Volumetric Segmentation Project

This project provides a skeleton for volumetric segmentation tasks using the MONAI framework. It includes:

- An interface function to evaluate a segmentation model on 3D medical images.
- Default support for pretrained Swin-UNETR.
- Standard evaluation metrics (Dice and Hausdorff Distance).

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Import and call the evaluation function:
   ```python
   from monai_project.interface import evaluate_segmentation_performance
   
   metrics = evaluate_segmentation_performance(
       dataset_name="MyDataset",
       dataloader=my_dataloader
   )
   print(metrics)
   ```
