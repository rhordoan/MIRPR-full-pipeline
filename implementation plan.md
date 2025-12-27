This implementation plan details the technical execution for predicting EGFR, KRAS, and Ki-67 using your existing Vista3D model and NVIDIA B300 infrastructure.

The strategy relies on a **"Transfer and Inference"** workflow: your LIDC-trained Vista3D model acts as a high-precision mask generator for the molecularly labeled datasets (NSCLC-Radiogenomics and TCGA-LUAD), feeding into a **Late Fusion** prediction architecture.

### Phase 1: Infrastructure & Data Harmonization (B300 Setup)

**Objective:** Leverage the 288GB VRAM of the B300 to handle high-resolution volumetric data without tiling artifacts.

* **Hardware Configuration:**
* **VRAM Utilization:** Configure your data loaders to use large 3D patch sizes (e.g., ). The B300's memory allows this without "Gradient Checkpointing," preventing the 30-40% speed penalty usually incurred on smaller GPUs.


* 
**Precision:** Enable **FP4** or FP8 precision in the Transformer Engine if using Swin Transformer backbones to maximize the ~14 PFLOPS throughput.



* **Data Standardization:**
* 
**Format:** Convert all NSCLC-Radiogenomics (EGFR/KRAS source) and TCGA-LUAD (Ki-67 source) DICOMs to NIfTI (`.nii.gz`).


* **Resampling:** You *must* resample all volumes to isotropic resolution () using B-Spline interpolation. This is critical for 3D CNN kernels and radiomics texture matrices to remain rotationally invariant.


* 
**Intensity Normalization:** Clip HU values to a lung window (e.g., -1200 to +600) and normalize to  for the Deep Learning branch.





### Phase 2: The "Mask-to-Marker" Transfer

**Objective:** Use the LIDC-trained Vista3D model to generate ground truth regions for the genomic datasets.

1. **Inference Execution:** Run Vista3D inference on the 211 NSCLC-Radiogenomics scans and the ~500 TCGA-LUAD scans.
* 
*Configuration:* Set `drop_point_prob=1.0` in your inference config to force fully automated segmentation without user prompts.




2. **Quality Control (QC):**
* Since these datasets may use different scanner protocols (GE vs. Siemens), domain shift is a risk. Visually inspect a 10% random sample.
* 
*Fall-back:* If segmentation fails, fine-tune Vista3D for 5-10 epochs on a small subset of the target data (10-20 scans) annotated manually, leveraging the B300's speed.





### Phase 3: The Radiomics Branch (Feature Engineering)

**Objective:** Extract interpretable, texture-based biomarkers using **PySERA** or **PyRadiomics**. This branch is essential for capturing the "heterogeneity" that correlates with Ki-67 and EGFR.

* 
**Library:** Use **PySERA** (or PyRadiomics) for IBSI-compliant extraction.


* **Configuration (`params.yaml`):**
* **Bin Width:** Set `binWidth: 25`. This is mandatory for CT to maintain physical Hounsfield Unit meaning (do not use binCount).


* **Filters:** Enable **Wavelet** decompositions. The High-Frequency bands (specifically **LLH** and **HHL**) are the strongest predictors of EGFR mutation status.


* 
**LoG:** Enable Laplacian of Gaussian with sigma values  to capture multi-scale spiculation.



* **Output:** A feature vector of ~1,500 metrics per nodule.
* 
**Preprocessing:** Apply **ComBat Harmonization** to the extracted features to remove batch effects between the Stanford (NSCLC) and TCGA datasets.



### Phase 4: The Deep Learning Branch (Mask-Channel Architecture)

**Objective:** Train a 3D classification network that explicitly sees the tumor shape.

* **Input Strategy (Mask-Channel):**
* Do not just crop the image. Construct a **4D input tensor** of shape .
* **Channel 0:** The CT Volume (Normalized).
* **Channel 1:** The Binary Segmentation Mask (from Phase 2).
* **Implementation:** Use MONAI's `ConcatItemsd(keys=["image", "label"])` transform. This "hard-codes" attention to the ROI while preserving peritumoral context.




* **Architecture:**
* Use a **Swin Transformer 3D** or **3D-DenseNet** backbone. The B300 allows you to use the **DCSwinB** (Dual-Branch) architecture, combining CNN local features with Transformer global attention.


* **Multi-Task Heads:** Since you are predicting multiple markers, use a shared backbone with separate prediction heads:
* *Head A (Binary):* EGFR (Mutant/WT)
* *Head B (Binary):* KRAS (Mutant/WT)
* 
*Head C (Regression):* Ki-67 (Targeting *MKI67* mRNA expression levels from TCGA).







### Phase 5: Late Fusion (The SOTA Integrator)

**Objective:** Combine the two branches for maximum accuracy (AUC > 0.90).

1. **Train Branch A (Radiomics):** Train an **XGBoost** classifier on the PySERA features. Select top 20 features using LASSO to avoid overfitting. Output a probability .


2. **Train Branch B (Deep Learning):** Train the Swin Transformer on the Mask-Channel inputs. Output a probability .
3. **Fusion:** Implement a meta-learner (Logistic Regression) or a simple weighted average:


* 
*Note:* The "Late Fusion" strategy consistently outperforms end-to-end models for biomarker prediction because radiomics stabilizes the prediction on smaller datasets like NSCLC-Radiogenomics.





### Summary of Tech Stack

* **Hardware:** NVIDIA B300 (Nebius AI Cloud or On-Prem).
* **Framework:** PyTorch + MONAI.
* **Transforms:** `ConcatItemsd` (Mask Fusion), `RandAffine` (Augmentation).
* **Radiomics:** PySERA / PyRadiomics (with Wavelets).
* **Classifier:** XGBoost (Radiomics) + SwinTransformer3D (Deep Learning).

