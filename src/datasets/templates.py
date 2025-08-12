# Medical segmentation templates for CLIPSeg

chaos_ct_template = [
    lambda c: f"CT scan showing {c}.",
    lambda c: f"CT image of {c}.",
    lambda c: f"computed tomography of {c}.",
    lambda c: f"abdominal CT scan with {c}.",
    lambda c: f"CT segmentation of {c}.",
    lambda c: f"radiological CT image showing {c}.",
    lambda c: f"medical CT scan of {c}.",
    lambda c: f"anatomical structure {c} in CT.",
    lambda c: f"organ {c} in CT imaging.",
    lambda c: f"CT-based segmentation of {c}.",
]

chaos_mri_template = [
    lambda c: f"MRI scan showing {c}.",
    lambda c: f"MRI image of {c}.",
    lambda c: f"magnetic resonance imaging of {c}.",
    lambda c: f"abdominal MRI scan with {c}.",
    lambda c: f"MRI segmentation of {c}.",
    lambda c: f"radiological MRI image showing {c}.",
    lambda c: f"medical MRI scan of {c}.",
    lambda c: f"anatomical structure {c} in MRI.",
    lambda c: f"organ {c} in MRI imaging.",
    lambda c: f"MRI-based segmentation of {c}.",
    lambda c: f"T1-weighted image of {c}.",
    lambda c: f"T2-weighted image of {c}.",
]

mmwhs_ct_template = [
    lambda c: f"cardiac CT scan showing {c}.",
    lambda c: f"CT image of {c}.",
    lambda c: f"computed tomography of {c}.",
    lambda c: f"cardiac CT with {c}.",
    lambda c: f"CT segmentation of {c}.",
    lambda c: f"heart CT scan showing {c}.",
    lambda c: f"cardiovascular CT imaging of {c}.",
    lambda c: f"cardiac structure {c} in CT.",
    lambda c: f"heart anatomy {c} in CT imaging.",
    lambda c: f"CT-based cardiac segmentation of {c}.",
]

mmwhs_mri_template = [
    lambda c: f"cardiac MRI scan showing {c}.",
    lambda c: f"MRI image of {c}.",
    lambda c: f"magnetic resonance imaging of {c}.",
    lambda c: f"cardiac MRI with {c}.",
    lambda c: f"MRI segmentation of {c}.",
    lambda c: f"heart MRI scan showing {c}.",
    lambda c: f"cardiovascular MRI imaging of {c}.",
    lambda c: f"cardiac structure {c} in MRI.",
    lambda c: f"heart anatomy {c} in MRI imaging.",
    lambda c: f"MRI-based cardiac segmentation of {c}.",
    lambda c: f"cine MRI of {c}.",
    lambda c: f"cardiac T1-weighted image of {c}.",
]


dataset_to_template = {
    ("CHAOS", "CT"): chaos_ct_template,
    ("CHAOS", "MR"): chaos_mri_template,
    ("MMWHS", "CT"): mmwhs_ct_template,
    ("MMWHS", "MR"): mmwhs_mri_template,
}
