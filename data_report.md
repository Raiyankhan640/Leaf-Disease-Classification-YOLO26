# AgriVision — Multi-Crop Plant Disease Dataset

Unified dataset prepared for **YOLO image classification** training.  
All images are organised into `train / val / test` splits under a flat class-folder structure that Ultralytics YOLO reads natively.

---

## Dataset at a Glance

| Property | Value |
|---|---|
| Total crops | **8** |
| Total classes | **57** |
| Total images | **10,727** |
| Train split | ~80 % |
| Val split | ~10 % |
| Test split | ~10 % |
| Split strategy | Stratified random (seed = 42) — Rice uses original pre-split |

---

## Folder Structure

```
AgriVision_Split/
├── train/
│   ├── Batol_Gourd__Alternaria_Leaf_Blight/
│   ├── Chili__Curl_Virus/
│   ├── Lemon__Healthy_Leaf/
│   ├── Rice__Leaf_Blast/
    └── ... (57 class folders)
├── val/
│   └── ... (same 57 class folders)
└── test/
    └── ... (same 51 class folders)
```

Class folder naming convention: **`{Crop}__{Disease}`** (double underscore separates crop from disease).

---

## Crop-wise Summary

### 1. Batol Gourd — 7 classes · 1,659 images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Batol_Gourd__Alternaria_Leaf_Blight | 243 | 30 | 31 | **304** |
| Batol_Gourd__Anthracnose | 220 | 28 | 28 | **276** |
| Batol_Gourd__Downy_Mildew | 228 | 29 | 29 | **286** |
| Batol_Gourd__Early_Alternaria_Leaf_Blight | 143 | 18 | 18 | **179** |
| Batol_Gourd__Fungal_Damage_Leaf | 31 | 4 | 4 | **39** |
| Batol_Gourd__Healthy | 208 | 26 | 26 | **260** |
| Batol_Gourd__Mosaic_Virus | 252 | 31 | 32 | **315** |
| **Subtotal** | **1,325** | **166** | **168** | **1,659** |

> Split method: 80/10/10 stratified random split.

---

### 2. Chili — 6 classes · 1,856 images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Chili__Bacterial_Spot | 124 | 16 | 16 | **156** |
| Chili__Cercospora_Leaf_Spot | 144 | 18 | 18 | **180** |
| Chili__Curl_Virus | 338 | 42 | 43 | **423** |
| Chili__Healthy_Leaf | 366 | 46 | 46 | **458** |
| Chili__Nutrition_Deficiency | 355 | 44 | 45 | **444** |
| Chili__White_spot | 156 | 19 | 20 | **195** |
| **Subtotal** | **1,483** | **185** | **188** | **1,856** |

> Split method: 80/10/10 stratified random split.

---

### 3. Lemon — 9 classes · 1,354 images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Lemon__Anthracnose | 80 | 10 | 10 | **100** |
| Lemon__Bacterial_Blight | 84 | 10 | 11 | **105** |
| Lemon__Citrus_Canker | 142 | 18 | 18 | **178** |
| Lemon__Curl_Virus | 92 | 11 | 12 | **115** |
| Lemon__Deficiency_Leaf | 154 | 19 | 20 | **193** |
| Lemon__Dry_Leaf | 148 | 19 | 19 | **186** |
| Lemon__Healthy_Leaf | 168 | 21 | 21 | **210** |
| Lemon__Sooty_Mould | 122 | 15 | 16 | **153** |
| Lemon__Spider_Mites | 91 | 11 | 12 | **114** |
| **Subtotal** | **1,081** | **134** | **139** | **1,354** |

> Split method: 80/10/10 stratified random split. Source folder was `Lemon/Original Dataset/`.

---

### 4. Papaya — 7 classes · 1,684 images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| papaya_main_dataset__Bacterial_Blight | 146 | 18 | 19 | **183** |
| papaya_main_dataset__Carica_Insect_Hole | 254 | 32 | 32 | **318** |
| papaya_main_dataset__Curled_Yellow_Spot | 430 | 54 | 54 | **538** |
| papaya_main_dataset__healthy_leaf | 151 | 19 | 19 | **189** |
| papaya_main_dataset__Mosaic_Virus | 95 | 12 | 12 | **119** |
| papaya_main_dataset__pathogen_symptoms | 228 | 29 | 29 | **286** |
| papaya_main_dataset__Yellow_Necrotic_Spots_Holes | 40 | 5 | 6 | **51** |
| **Subtotal** | **1,344** | **169** | **171** | **1,684** |

> Split method: 80/10/10 stratified random split.

---

### 5. Rice — 8 classes · 1,886 images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Rice__Bacterial_Leaf_Blight | 146 | 20 | 42 | **208** |
| Rice__Brown_Spot | 192 | 27 | 55 | **274** |
| Rice__Healthy_Rice_Leaf | 131 | 19 | 37 | **187** |
| Rice__Leaf_Blast | 217 | 31 | 62 | **310** |
| Rice__Leaf_scald | 162 | 23 | 46 | **231** |
| Rice__Narrow_Brown_Leaf_Spot | 114 | 16 | 33 | **163** |
| Rice__Rice_Hispa | 158 | 22 | 45 | **225** |
| Rice__Sheath_Blight | 202 | 28 | 58 | **288** |
| **Subtotal** | **1,322** | **186** | **378** | **1,886** |

> Split method: **Pre-split** (original dataset already came with `Training data / Validation data / Testing data` folders).  
> Source: *Rice Leaf Bacterial and Fungal Disease Dataset* — Bangladeshi rice. [Mendeley Data](https://data.mendeley.com/datasets/hx6f852hw4/1)

---

### 6. Tomato — 5 classes · 916 images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Tomato__Downy Mildew | 45 | 6 | 6 | **57** |
| Tomato__Healthy | 230 | 29 | 29 | **288** |
| Tomato__Mosaic | 156 | 19 | 20 | **195** |
| Tomato__Spot | 248 | 31 | 32 | **311** |
| Tomato__white_spot | 52 | 6 | 7 | **65** |
| **Subtotal** | **731** | **91** | **94** | **916** |

> Split method: 80/10/10 stratified random split.

---

### 7. Zucchini — 9 classes · 988 images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Zucchini_Original__Angular_Leaf_Spot | 96 | 12 | 12 | **120** |
| Zucchini_Original__Anthracnose | 103 | 13 | 13 | **129** |
| Zucchini_Original__Downy_Midew | 122 | 15 | 16 | **153** |
| Zucchini_Original__Dry_Leaf | 53 | 7 | 7 | **67** |
| Zucchini_Original__Healthy | 70 | 9 | 9 | **88** |
| Zucchini_Original__Insect_Damage | 62 | 8 | 8 | **78** |
| Zucchini_Original__Iron_Chlorosis_Damage | 52 | 6 | 7 | **65** |
| Zucchini_Original__Xanthomonas_Leaf_Spot | 68 | 9 | 9 | **86** |
| Zucchini_Original__Yellow_Mosaic_Virus | 161 | 20 | 21 | **202** |
| **Subtotal** | **787** | **99** | **102** | **988** |

> Split method: 80/10/10 stratified random split.

---

### 8. Eggplant — 6 classes · 384 images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Eggplant__Healthy_Leaf | 53 | 7 | 7 | **67** |
| Eggplant__Insect_Pest_Disease | 75 | 9 | 10 | **94** |
| Eggplant__Leaf_Spot_Disease | 88 | 11 | 12 | **111** |
| Eggplant__Mosaic_Virus_Disease | 28 | 4 | 4 | **36** |
| Eggplant__Small_Leaf_Disease | 11 | 1 | 2 | **14** |
| Eggplant__Wilt_Disease | 49 | 6 | 7 | **62** |
| **Subtotal** | **304** | **38** | **42** | **384** |

> Split method: 80/10/10 stratified random split. Class `Eggplant White Mold Disease` was skipped (only 8 images — below the 10-image threshold). The "Eggplant " prefix was stripped from disease names to keep labels clean (e.g. `Eggplant__Healthy_Leaf` instead of `Eggplant__Eggplant_Healthy_Leaf`).

---

## Grand Total by Split

| Split | Images |
|---|---|
| Train | **8,377** |
| Val | **1,068** |
| Test | **1,282** |
| **Total** | **10,727** |

---

## YOLO Training Usage

```bash
# Ultralytics YOLO classification — example command
yolo classify train \
  model=yolov8m-cls.pt \
  data="d:/499A Dataset/AgriVision/AgriVision_Split" \
  epochs=50 \
  imgsz=224 \
  batch=32 \
  name=agrivision_57cls
```

YOLO auto-discovers all 57 class folders under `train/` — no YAML file required for classification tasks.

---

## Scripts

| Script | Purpose |
|---|---|
| `dataset_split.py` | Split Batol Gourd / Papaya / Tomato / Zucchini (80/10/10) |
| `merge_rice.py` | Merge pre-split Rice dataset into `AgriVision_Split` |
| `merge_chili_lemon.py` | Split and merge Chili and Lemon (80/10/10) |
| `merge_eggplant.py` | Split and merge Eggplant (80/10/10) |
