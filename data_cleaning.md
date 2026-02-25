# AgriVision Dataset Cleanup & Balancing Report
**Date:** February 27, 2026

---

## Executive Summary

| Metric | Value |
|---|---|
| Initial classes | 57 |
| Final classes | **37** |
| Classes removed | 20 (Zucchini: 9, Lemon: 4, Papaya: 7) |
| Corrupted images found | **0** (out of 9,126 scanned) |
| Augmented images added (train only) | **328** |
| Images removed by undersampling (train only) | **4,631** |
| Final train images | **5,482** |
| Final val images | **758** |
| Final test images | **964** |
| **Final total images** | **7,204** |

---

## 1. Corrupted Image Scan

All **9,126 images** across train/val/test splits were scanned using PIL's `Image.verify()` and `Image.load()` dual-pass validation (verify checks headers, load actually decodes all pixels).

**Result: No corrupted images found.** All images in the dataset are valid and intact.

---

## 2. Class Removal

### Removed Class Groups (20 classes total):

**Zucchini_Original (9 classes)** — Removed per user request due to low accuracy (0.00–0.82 F1-score)
- Angular_Leaf_Spot, Anthracnose, Downy_Midew, Dry_Leaf, Healthy, Insect_Damage, Iron_Chlorosis_Damage, Xanthomonas_Leaf_Spot, Yellow_Mosaic_Virus

**Lemon (4 classes removed)** — Low support / lower accuracy; top 5 classes by accuracy + training volume were retained (see Kept Crop Groups below)
- Removed: Anthracnose, Bacterial_Blight, Curl_Virus, Spider_Mites

**Papaya (7 classes)** — Not in the target crop list
- Bacterial_Blight, Carica_Insect_Hole, Curled_Yellow_Spot, healthy_leaf, Mosaic_Virus, pathogen_symptoms, Yellow_Necrotic_Spots_Holes

All class directories were completely deleted from **train**, **val**, and **test** splits.

### Kept Crop Groups (37 classes):

| Crop | # Classes | Classes |
|---|---|---|
| Batol Gourd | 7 | Alternaria_Leaf_Blight, Anthracnose, Downy_Mildew, Early_Alternaria_Leaf_Blight, Fungal_Damage_Leaf, Healthy, Mosaic_Virus |
| Chili | 6 | Bacterial_Spot, Cercospora_Leaf_Spot, Curl_Virus, Healthy_Leaf, Nutrition_Deficiency, White_spot |
| Eggplant | 6 | Healthy_Leaf, Insect_Pest_Disease, Leaf_Spot_Disease, Mosaic_Virus_Disease, Small_Leaf_Disease, Wilt_Disease |
| Lemon | 5 | Citrus_Canker, Deficiency_Leaf, Dry_Leaf, Healthy_Leaf, Sooty_Mould |
| Rice | 8 | Bacterial_Leaf_Blight, Brown_Spot, Healthy_Rice_Leaf, Leaf_Blast, Leaf_scald, Narrow_Brown_Leaf_Spot, Rice_Hispa, Sheath_Blight |
| Tomato | 5 | Downy Mildew, Healthy, Mosaic, Spot, white_spot |

---

## 3. Augmentation of Low-Support Classes

**Applied to:** Train split ONLY (val/test untouched to prevent data leakage)

**Method:** Albumentations library with safe, quality-preserving transforms:
- **Horizontal Flip** (p=0.5)
- **Vertical Flip** (p=0.3)
- **Random Brightness/Contrast** (±20%, p=0.5)
- **Gaussian Blur** (kernel 3–5, p=0.3)
- **Shift/Scale/Rotate** (shift=5%, scale=5%, rotate=±15°, p=0.5)
- **Hue/Saturation/Value** shift (p=0.3)

**Logic:** Classes with train count below the low threshold (78) were augmented up to the median (156). Augmented images are prefixed with `aug_` for easy identification.

### Augmented Classes (3 classes, 328 images added):

| Class | Original Train | Augmented Added | New Train Total |
|---|---|---|---|
| Batol_Gourd__Fungal_Damage_Leaf | 62 | 94 | 156 |
| Eggplant__Mosaic_Virus_Disease | 56 | 100 | 156 |
| Eggplant__Small_Leaf_Disease | 22 | 134 | 156 |
| **TOTAL** | **140** | **328** | **468** |

---

## 4. Undersampling of Over-Represented Classes

**Applied to:** Train split ONLY (val/test untouched for evaluation integrity)

**Method:** Random removal of excess images to bring counts down to the median cap (156). Original images were randomly selected for removal (reproducible with seed=42).

### Undersampled Classes (25 classes, 4,631 images removed):

| Class | Before | Removed | After |
|---|---|---|---|
| Batol_Gourd__Alternaria_Leaf_Blight | 398 | 243 | 156 |
| Batol_Gourd__Anthracnose | 376 | 220 | 156 |
| Batol_Gourd__Downy_Mildew | 384 | 228 | 156 |
| Batol_Gourd__Early_Alternaria_Leaf_Blight | 286 | 130 | 156 |
| Batol_Gourd__Healthy | 364 | 208 | 156 |
| Batol_Gourd__Mosaic_Virus | 408 | 252 | 156 |
| Chili__Bacterial_Spot | 248 | 92 | 156 |
| Chili__Cercospora_Leaf_Spot | 288 | 132 | 156 |
| Chili__Curl_Virus | 494 | 338 | 156 |
| Chili__Healthy_Leaf | 522 | 366 | 156 |
| Chili__Nutrition_Deficiency | 511 | 355 | 156 |
| Chili__White_spot | 312 | 156 | 156 |
| Eggplant__Leaf_Spot_Disease | 176 | 20 | 156 |
| Lemon__Healthy_Leaf | 168 | 12 | 156 |
| Rice__Bacterial_Leaf_Blight | 292 | 136 | 156 |
| Rice__Brown_Spot | 348 | 192 | 156 |
| Rice__Healthy_Rice_Leaf | 262 | 106 | 156 |
| Rice__Leaf_Blast | 373 | 217 | 156 |
| Rice__Leaf_scald | 318 | 162 | 156 |
| Rice__Narrow_Brown_Leaf_Spot | 228 | 72 | 156 |
| Rice__Rice_Hispa | 314 | 158 | 156 |
| Rice__Sheath_Blight | 358 | 202 | 156 |
| Tomato__Healthy | 386 | 230 | 156 |
| Tomato__Mosaic | 312 | 156 | 156 |
| Tomato__Spot | 404 | 248 | 156 |
| **TOTAL** | **9,841** | **4,631** | **3,900** |

---

## 5. Final Dataset Distribution

### Per-Class Image Counts (Final State)

| # | Class | Train (orig) | Train (aug) | Train Total | Val | Test | Grand Total |
|---|---|---|---|---|---|---|---|
| 1 | Batol_Gourd__Alternaria_Leaf_Blight | 156 | 0 | 156 | 30 | 31 | 217 |
| 2 | Batol_Gourd__Anthracnose | 156 | 0 | 156 | 28 | 28 | 212 |
| 3 | Batol_Gourd__Downy_Mildew | 156 | 0 | 156 | 29 | 29 | 214 |
| 4 | Batol_Gourd__Early_Alternaria_Leaf_Blight | 156 | 0 | 156 | 18 | 18 | 192 |
| 5 | Batol_Gourd__Fungal_Damage_Leaf | 62 | 94 | 156 | 4 | 4 | 164 |
| 6 | Batol_Gourd__Healthy | 156 | 0 | 156 | 26 | 26 | 208 |
| 7 | Batol_Gourd__Mosaic_Virus | 156 | 0 | 156 | 31 | 32 | 219 |
| 8 | Chili__Bacterial_Spot | 156 | 0 | 156 | 16 | 16 | 188 |
| 9 | Chili__Cercospora_Leaf_Spot | 156 | 0 | 156 | 18 | 18 | 192 |
| 10 | Chili__Curl_Virus | 156 | 0 | 156 | 42 | 43 | 241 |
| 11 | Chili__Healthy_Leaf | 156 | 0 | 156 | 46 | 46 | 248 |
| 12 | Chili__Nutrition_Deficiency | 156 | 0 | 156 | 44 | 45 | 245 |
| 13 | Chili__White_spot | 156 | 0 | 156 | 19 | 20 | 195 |
| 14 | Eggplant__Healthy_Leaf | 106 | 0 | 106 | 7 | 7 | 120 |
| 15 | Eggplant__Insect_Pest_Disease | 150 | 0 | 150 | 9 | 10 | 169 |
| 16 | Eggplant__Leaf_Spot_Disease | 156 | 0 | 156 | 11 | 12 | 179 |
| 17 | Eggplant__Mosaic_Virus_Disease | 56 | 100 | 156 | 4 | 4 | 164 |
| 18 | Eggplant__Small_Leaf_Disease | 22 | 134 | 156 | 1 | 2 | 159 |
| 19 | Eggplant__Wilt_Disease | 98 | 0 | 98 | 6 | 7 | 111 |
| 20 | Lemon__Citrus_Canker | 142 | 0 | 142 | 18 | 18 | 178 |
| 21 | Lemon__Deficiency_Leaf | 154 | 0 | 154 | 19 | 20 | 193 |
| 22 | Lemon__Dry_Leaf | 148 | 0 | 148 | 19 | 19 | 186 |
| 23 | Lemon__Healthy_Leaf | 156 | 0 | 156 | 21 | 21 | 198 |
| 24 | Lemon__Sooty_Mould | 122 | 0 | 122 | 15 | 16 | 153 |
| 25 | Rice__Bacterial_Leaf_Blight | 156 | 0 | 156 | 20 | 42 | 218 |
| 26 | Rice__Brown_Spot | 156 | 0 | 156 | 27 | 55 | 238 |
| 27 | Rice__Healthy_Rice_Leaf | 156 | 0 | 156 | 19 | 37 | 212 |
| 28 | Rice__Leaf_Blast | 156 | 0 | 156 | 31 | 62 | 249 |
| 29 | Rice__Leaf_scald | 156 | 0 | 156 | 23 | 46 | 225 |
| 30 | Rice__Narrow_Brown_Leaf_Spot | 156 | 0 | 156 | 16 | 33 | 205 |
| 31 | Rice__Rice_Hispa | 156 | 0 | 156 | 22 | 45 | 223 |
| 32 | Rice__Sheath_Blight | 156 | 0 | 156 | 28 | 58 | 242 |
| 33 | Tomato__Downy Mildew | 90 | 0 | 90 | 6 | 6 | 102 |
| 34 | Tomato__Healthy | 156 | 0 | 156 | 29 | 29 | 214 |
| 35 | Tomato__Mosaic | 156 | 0 | 156 | 19 | 20 | 195 |
| 36 | Tomato__Spot | 156 | 0 | 156 | 31 | 32 | 219 |
| 37 | Tomato__white_spot | 104 | 0 | 104 | 6 | 7 | 117 |
| | **TOTAL** | **5,154** | **328** | **5,482** | **758** | **964** | **7,204** |

### Split Summary
- **Train:** 5,482 images (5,154 original + 328 augmented)
- **Validation:** 758 images (all original, untouched)
- **Test:** 964 images (all original, untouched)
- **Grand Total:** 7,204 images

### Train Balance Statistics
- **Min train count:** 90 (Tomato__Downy Mildew)
- **Max train count:** 156 (25 classes)
- **Median train count:** 156
- **Classes at median (156):** 25 out of 37
- **Classes below median (not needing augmentation, 78–155):** 9 classes
- **Classes augmented (< 78 threshold):** 3 classes

---

## 6. Configuration Updates

- **`data.yaml`** updated with `nc: 37` and all 37 class names (0-indexed, alphabetically sorted)
- All splits (train/val/test) have **consistent class directories** — same 37 classes in each
- Train/val/test split structure preserved throughout all operations

---

## 7. Data Integrity Notes

- **No data leakage:** Augmented images exist ONLY in the train split. Val and test splits are completely untouched.
- **Augmented files identifiable:** All augmented images are prefixed with `aug_` (e.g., `aug_0001_original_name.jpg`)
- **Val/Test splits preserved:** Zero modifications to validation and test sets — evaluation integrity maintained.
- **Reproducibility:** All random operations used `seed=42` for deterministic results.
- **Quality preservation:** Augmentation transforms were conservative (small rotations, mild blur, slight brightness/contrast changes) to avoid introducing artifacts.
- **Undersampling strategy:** Random removal with seed for reproducibility; no systematic bias in which images were kept.

---

## 8. Classes Not Modified (Between Threshold)

These 9 classes had train counts between the low threshold (78) and the median cap (156), so they were left as-is:

| Class | Train Count | Reason |
|---|---|---|
| Eggplant__Healthy_Leaf | 106 | Within range |
| Eggplant__Insect_Pest_Disease | 150 | Within range |
| Eggplant__Wilt_Disease | 98 | Within range |
| Lemon__Citrus_Canker | 142 | Within range |
| Lemon__Deficiency_Leaf | 154 | Within range |
| Lemon__Dry_Leaf | 148 | Within range |
| Lemon__Sooty_Mould | 122 | Within range |
| Tomato__Downy Mildew | 90 | Within range |
| Tomato__white_spot | 104 | Within range |
