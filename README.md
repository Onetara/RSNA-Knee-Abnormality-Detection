# RSNA Knee Abnormality Detection
**Author:** Dr. S. L. Toriola, MD, MSc.

**Goal:** Learn medical imaging AI by building deep-learning models in Python (PyTorch, MONAI, scikit-learn) on Kaggle.

## Start Here (plain-language guide)

- **MRI study** = a set of knee scans from one patient visit, made of many image slices.
- **Slice** = one flat image from the scan. A study stacks many slices into a 3D volume.
- **DICOM** = the standard medical image file format.
- **Radiology report** = the written findings from the radiologist. Training studies have one; the test set does not.
- **AUC** = how well a model ranks positive cases above negative ones. 0.5 is a coin flip, 1.0 is perfect.
- **Macro AUC** = the average AUC across all 12 findings. This is the competition score.
- **Transfer learning** = starting from a model already trained on other images, then training it on this task.

## Project

**What it asks**
- Can a model predict 12 knee findings from an MRI study?
- Can radiology report text help during training, even though the test set has no reports?
- Does a 2D model or a 3D model do better on the same task?

**The 12 findings per knee study:** ACL, MCL, medial meniscus, lateral meniscus, medial OA, lateral OA, patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture. (OA = osteoarthritis, joint wear.)

**What it provides**
- Two Kaggle notebooks, one per approach (see Foundation).
- Trained weights for both models and the report-text vectorizer.
- A first end-to-end attempt, entered as a competition and as a learning project.

**Results**

| Model | Score |
|---|---|
| ResNet-18 (2D) | Best validation AUC 0.520 (epoch 9 of 10); best Kaggle score 0.509 |
| MONAI (3D) | Kaggle scores around 0.51 to 0.535 |
| Best overall Kaggle score | 0.540 |
| Top team in the competition | 0.942 |

- All of my scores sit near chance (0.5), so the repo is best read as a first end-to-end attempt.
- Likely causes to investigate: slice sampling (32 evenly spaced slices), per-slice normalization, and labels extracted from report text.

**What I learned**
- Building a full pipeline: DICOM loading, 2D vs 3D inputs, transfer learning, imbalanced-label losses, validation, early stopping, checkpoint resuming.
- Comparing two architectures on the same task and reading the results honestly.
- Near-chance AUC points to the data pipeline, not model choice.

**Next steps**
- Sanity checks: overfit a tiny subset, view the model inputs, verify label quality.
- Revisit slice selection and normalization.
- Use the right series per finding instead of pooling all series.

## Foundation

**Structure**
- `notebooks/`: the two Kaggle notebooks.
- `models/`: trained weights and the text vectorizer.

**Data**
- RSNA knee MRI competition data on Kaggle (DICOM studies, labels, and training reports). It is not included in this repo.
- Notebooks are written for the Kaggle environment (`/kaggle/input`, `/kaggle/working`). Outputs are stripped, and CSVs and checkpoints are git-ignored except the files in `models/`.

**Variables and settings (what each one means)**

| Item | What it means |
|---|---|
| Macro AUC | Average AUC over the 12 findings. The competition score |
| Slices per study | 32 evenly spaced slices are sampled from each scan |
| Per-slice normalization | Each slice is rescaled using its own darkest and brightest values |
| Attention slice pooling | The 2D model learns which slices matter most and weights them |
| Expert prediction heads | One small output network per finding, all fed the same image features |
| Focal class-balanced loss | Training loss that gives rare findings more weight |
| Report labels | Findings pulled from report text with medspaCy, negspacy, and RadLex (negation-aware) |
| TF-IDF vector | Numeric summary of report words, used by the 3D model |
| Text dropout (85%) | The 3D model sees report text only 15% of the time, so it learns to work from images alone |

**Files and what each does**

| File | What it does |
|---|---|
| `notebooks/resnet18_mri_model.ipynb` | 2D ResNet-18 with attention slice pooling, expert heads, and focal class-balanced loss; report labels via medspaCy, negspacy and RadLex |
| `notebooks/mona_mri_model.ipynb` | 3D MONAI ResNet-18 (MedicalNet pretrained) with TF-IDF report fusion; images only at test time |
| `models/resnet18_model_best.pt` | ResNet-18 best weights |
| `models/mona_tfidf_vectorizer.joblib` | TF-IDF vectorizer for the Mona model |
| `models/mona_model_3d_best.pt.part00`, `.part01` | Mona 3D model weights, split to fit GitHub's 100 MB limit |

**How to run**
- Open a notebook on Kaggle with the competition data attached and run the cells in order.
- To use the split Mona weights, rejoin them from the `models/` folder:

```
cat mona_model_3d_best.pt.part* > model_3d_best.pt
```
