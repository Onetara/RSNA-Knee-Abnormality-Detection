# RSNA Knee Abnormality Detection

Kaggle competition code: multimodal model (knee MRI + radiology report text) predicting 12 knee abnormalities.

Entered as a competition and as a learning project in building AI models. Scores stayed near chance, so the repo is best read as a first end-to-end attempt.

## Goal
- Predict 12 findings per knee MRI study: ACL, MCL, medial/lateral meniscus, medial/lateral/PF OA, effusion, synovitis, Baker's cyst, contusion, fracture.
- Metric: macro AUC (0.5 = chance). Top team: 0.942.
- Training studies have radiology reports; the test set has none, so text can only help during training.

## Notebooks
| File | Approach |
|---|---|
| `notebooks/resnet18_mri_model.ipynb` | 2D ResNet-18 + attention slice pooling, expert prediction heads, focal class-balanced loss; report labels via medspaCy/negspacy + RadLex |
| `notebooks/mona_mri_model.ipynb` | 3D MONAI ResNet-18 (MedicalNet pretrained) + TF-IDF report fusion (85% text dropout; images only at test time) |

## Results
| | Score |
|---|---|
| ResNet-18 (2D) | best validation AUC 0.520 (epoch 9/10); best Kaggle score 0.509 |
| MONAI (3D) | scored around 0.51 to 0.535 on Kaggle |
| Best overall Kaggle score | 0.540 |

All scores are near chance (0.5). Likely causes to investigate: slice sampling (32 evenly spaced slices), per-slice normalization, and labels extracted from report text.

## What I learned
- Building a full pipeline: DICOM loading, 2D vs 3D inputs, transfer learning, imbalanced-label losses, validation, early stopping, checkpoint resuming.
- Comparing two architectures on the same task and reading the results honestly.
- Near-chance AUC points to the data pipeline, not model choice.

## Next steps
- Sanity checks: overfit a tiny subset, view the model inputs, verify label quality.
- Revisit slice selection and normalization (currently 32 evenly spaced slices, per-slice scaling).
- Use the right series per finding instead of pooling all series.

## Notes
- Written for the Kaggle environment (`/kaggle/input`, `/kaggle/working`); competition data is not included.
- Outputs are stripped. CSVs and checkpoints are git-ignored, except the files in `models/`.

## Model weights (`models/`)
- `resnet18_model_best.pt`: ResNet-18 best weights.
- `mona_tfidf_vectorizer.joblib`: TF-IDF vectorizer for the Mona model.
- `mona_model_3d_best.pt.part00`, `.part01`: Mona 3D model weights, split to fit GitHub's 100 MB limit. Rejoin from the `models/` folder:

```
cat mona_model_3d_best.pt.part* > model_3d_best.pt
```

Author: S. L. Toriola, MD, MSc.
