# RSNA Knee Abnormality Detection

Kaggle competition code: multimodal model (knee MRI + radiology report text) predicting 12 knee abnormalities.

## Notebooks
| File | Approach |
|---|---|
| `notebooks/resnet18_mri_model.ipynb` | 2D ResNet-18 + attention slice pooling, expert prediction heads, focal class-balanced loss; report labels via medspaCy/negspacy + RadLex |
| `notebooks/mona_mri_model.ipynb` | 3D MONAI ResNet-18 (MedicalNet pretrained) + TF-IDF report fusion |

## Notes
- Written for the Kaggle environment (`/kaggle/input`, `/kaggle/working`); competition data is not included.
- Outputs are stripped. CSVs and checkpoints are git-ignored, except `models/resnet18_model_best.pt` (ResNet-18 best weights, 45.7 MB).

Author: S. L. Toriola, MD, MSc.

## Model weights (`models/`)
- `resnet18_model_best.pt`: ResNet-18 best weights.
- `mona_tfidf_vectorizer.joblib`: TF-IDF vectorizer for the Mona model.
- `mona_model_3d_best.pt.part00`, `.part01`: Mona 3D model weights, split to fit GitHub's 100 MB limit. Rejoin from the `models/` folder:

```
cat mona_model_3d_best.pt.part* > model_3d_best.pt
```
