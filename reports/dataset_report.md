# AgriVision — Plant Disease Recognition Dataset

**Weekly Progress Report | Prepared for Faculty Review**

**Date:** March 3, 2026
**Project Phase:** Dataset Construction & Preparation
**Status:** ✅ Complete — Ready for Model Training

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Construction Pipeline](#2-dataset-construction-pipeline)
   - [Step 1 — Source Data Collection](#step-1--source-data-collection)
   - [Step 2 — Initial Merge & Normalization](#step-2--initial-merge--normalization)
   - [Step 3 — Dataset Restructuring](#step-3--dataset-restructuring)
   - [Step 4 — Offline Augmentation (Phase 1)](#step-4--offline-augmentation-phase-1)
   - [Step 5 — New Crop Integration (Phase 2)](#step-5--new-crop-integration-phase-2)
   - [Step 6 — Research-Driven Class Trimming](#step-6--research-driven-class-trimming)
3. [Final Dataset Statistics](#3-final-dataset-statistics)
   - [Grand Summary](#grand-summary)
   - [Per-Crop Breakdown](#per-crop-breakdown)
   - [Class-Level Detail](#class-level-detail)
4. [Augmentation Details](#4-augmentation-details)
5. [New Crops Added — Detailed Accounting](#5-new-crops-added--detailed-accounting)
6. [Dataset Directory Structure](#6-dataset-directory-structure)
7. [Class Mapping Reference](#7-class-mapping-reference)
8. [Training Configuration Recommendations](#8-training-configuration-recommendations)
9. [Known Limitations & Notes](#9-known-limitations--notes)

---

## 1. Project Overview

**AgriVision** is a multi-crop, multi-class plant disease image classification dataset assembled for an agricultural AI recognition system targeting Bangladeshi farming conditions. The dataset aggregates images from multiple publicly available sources, applies systematic class normalization, stratified splitting, and targeted offline augmentation to produce a training-ready benchmark.

| Attribute | Value |
|---|---|
| Total Crops | 9 |
| Total Disease / Health Classes | 45 |
| Total Images | **17,506** |
| Training Images | 13,596 (77.7%) |
| Validation Images | 1,922 (11.0%) |
| Test Images | 1,988 (11.4%) |
| Split Strategy | Stratified 70 / 15 / 15 |
| Random Seed | 42 (fully reproducible) |
| Image Formats Accepted | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |

---

## 2. Dataset Construction Pipeline

The full dataset was constructed in **six sequential, scripted steps**. Each step is implemented as an independent Python script for transparency and reproducibility.

```
Source Datasets
      │
      ▼
Step 1 ─── Data Collection (AgriVision Base + Extra AgriVision Images + Potato)
      │
      ▼
Step 2 ─── merge_dataset.py ──── Merge + Normalize + Re-split (AgriVision_Merged)
      │
      ▼
Step 3 ─── restructure_dataset.py ── Target-class filtering + Potato add-in (AgriVision_Final v1)
      │
      ▼
Step 4 ─── augment_train.py ──── Offline augmentation for low-support train classes
      │
      ▼
Step 5 ─── merge_new_crops.py ── New crop integration (Cabbage, Cauliflower, Guava) + Augment
      │
      ▼
Step 6 ─── Research-driven class trimming ── Drop low-prevalence / ambiguous classes
      │
      ▼
AgriVision_Final (v3) ─── Final Dataset  [17,506 images | 45 classes | 9 crops]
```

---

### Step 1 — Source Data Collection

Three distinct raw data sources were identified and collected:

| Source | Description | Crops Contributed |
|---|---|---|
| **AgriVision Base Dataset** | Pre-split dataset structured into `train/val/test` folders | Rice, Tomato, Chili, Eggplant, Gourd |
| **Extra AgriVision Images** | Additional unlabelled/loosely labelled image pool in crop-subfolders | Supplementary images for Eggplant, Gourd, Tomato; raw data for Cauliflower, Cabbage, Guava |
| **Potato Dataset** | Separate external dataset with three potato disease classes | Potato (Early Blight, Late Blight, Healthy Leaf) |

---

### Step 2 — Initial Merge & Normalization

**Script:** `merge_dataset.py`

The base AgriVision dataset and the Extra AgriVision Images pool were merged into a unified intermediate output (`AgriVision_Merged`).

**Operations performed:**

- All images from both sources were pooled per target class using an explicit `CLASS_MAP` lookup table.
- Class folder names were normalized to the `Crop__DiseaseName` convention (e.g., `Eggplant__Leaf_Spot_Disease`).
- All images were renamed sequentially (e.g., `1.jpg`, `2.jpg`, …) to eliminate filename conflicts.
- The full per-class image pool was re-split using a **stratified 70 / 15 / 15** ratio (train / val / test).
- Random seed 42 was applied for deterministic reproducibility.

**Key class mappings applied at this step:**

| Source Folder | Target Class |
|---|---|
| `Eggplant/Aphids` | `Eggplant__Insect_Pest_Disease` |
| `Eggplant/Flea Beetles` | `Eggplant__Insect_Pest_Disease` |
| `Eggplant/Cercospora Leaf Spot` | `Eggplant__Leaf_Spot_Disease` |
| `Eggplant/Fresh Eggplant` + `Fresh Eggplant Leaf` | `Eggplant__Healthy_Leaf` |
| `Eggplant/Leaf Wilt` | `Eggplant__Wilt_Disease` |
| `Eggplant/Tobacco Mosaic Virus` | `Eggplant__Mosaic_Virus_Disease` |
| `Gourd_Leaf/Healthy` | `Gourd__Healthy_Leaf` |
| `Gourd_Leaf/Downy_Mildew` | `Gourd__Downy_Mildew` |
| `Gourd_Leaf/Mosaic_Virus` | `Gourd__Mosaic_Virus` |
| `Gourd_Leaf/Alternaria_Leaf_Blight` | `Gourd__Alternaria_Leaf_Blight` |
| Tomato train subfolders | Respective Tomato classes |

---

### Step 3 — Dataset Restructuring

**Script:** `restructure_dataset.py`

The merged dataset was filtered down to the **target class structure** defined for the final benchmark. Classes that did not correspond to target diseases (e.g., `Eggplant__Phytophthora_Blight`, `Eggplant__Powdery_Mildew`, `Gourd__Angular_Leaf_Spot`, `Lemon` crop) were **dropped entirely**.

**Operations performed:**

- Target class list was defined covering 6 original crops: Rice (8), Tomato (6), Chili (5), Eggplant (5), Gourd (4), and Potato (3).
- Potato images were ingested from the separate Potato dataset and assigned to the three Potato classes.
- Source-class-to-target-class pooling was applied where multiple source folders map to a single target class (e.g., `Rice__Healthy_Rice_Leaf` → `Rice__Healthy_Leaf`).
- The stratified 70/15/15 split was reapplied to the final image pool of each retained class.
- Output was written to `AgriVision_Final`.

**Result at end of Step 3:** 31 classes, 6 crops.

---

### Step 4 — Offline Augmentation (Phase 1)

**Script:** `augment_train.py`

After restructuring, the train split was inspected for low-support classes. The following per-crop minimum train targets were defined:

| Crop | Min Train Images per Class |
|---|---|
| Rice | 350 |
| Tomato | 250 |
| Chili | 250 |
| Eggplant | 200 |
| Potato | 250 |
| Gourd | 200 |

Classes that fell below these targets had their train sets augmented with **offline synthetic images**. The **val and test splits were never augmented** to preserve evaluation integrity.

**Augmentation techniques applied (2–4 transforms per image, randomly selected):**

| Technique | Parameter Range | Simulated Condition |
|---|---|---|
| Gaussian Blur | radius 0.5–2.0 | Defocus / hand shake |
| Brightness Jitter | 0.6–1.4× | Outdoor lighting variation |
| Contrast Jitter | 0.6–1.5× | Camera auto-exposure |
| Color / Saturation Jitter | 0.7–1.4× | White balance differences |
| Horizontal Flip | — | Orientation invariance |
| Rotation | −20° to +20° | Angled phone camera shots |
| Gaussian Noise | σ = 5–20 | Low-quality sensor simulation |
| Random Shadow Overlay | 30–70% darkness | Tree / structure shade |
| Sharpening | — | Over-sharpened phone processing |

> **Max augmentation ratio:** 5× per original image (to prevent overfitting on very small classes).

---

### Step 5 — New Crop Integration (Phase 2)

**Script:** `merge_new_crops.py`

Three crops of high agricultural relevance to Bangladesh were added to the finalized dataset. These crops are primarily grown during the **rabi (winter) season** and are frequently affected by the listed diseases in the region.

**New crops added:**

| Crop | Rationale | Classes Added |
|---|---|---|
| **Cabbage** | Major rabi-season vegetable; fungal/bacterial diseases prevalent in humid winters | 4 |
| **Cauliflower** | Major rabi-season vegetable; significant post-harvest and leaf disease burden | 7 |
| **Guava** | Common year-round fruit tree; diverse disease presence | 5 |

**Pipeline for each new crop:**

1. Images were pooled from respective subfolders in the `Extra AgriVision Images` source using the `NEW_CLASS_MAP` lookup.
2. Stratified 70/15/15 split was applied per class.
3. New class folders were written directly into `AgriVision_Final/train`, `val`, and `test`.
4. Train classes below the per-crop minimum target (150 for Cabbage/Guava; 250 for Cauliflower) were augmented using the same augmentation pipeline as Step 4.
5. The updated dataset report was regenerated.

**Raw vs. Augmented image counts for new classes:**

| Class | Original Images | Augmented Added | Final Train Count |
|---|---|---|---|
| `Cabbage__Alternaria_Spot` | 49 | +101 | 150 |
| `Cabbage__Black_Rot` | 25 | +100 | 125 |
| `Cabbage__Downy_Mildew` | 21 | +84 | 105 |
| `Cabbage__Healthy_Leaf` | 47 | +103 | 150 |
| `Cauliflower__Alternaria_Disease` | 247 | +3 | 250 |
| `Cauliflower__Bacterial_Soft_Rot` | 140 | +110 | 250 |
| `Cauliflower__Bacterial_Spot` | 283 | 0 | 283 |
| `Cauliflower__Black_Spot` | 175 | +75 | 250 |
| `Cauliflower__Downy_Mildew` | 287 | 0 | 287 |
| `Cauliflower__Healthy` | 632 | 0 | 632 |
| `Cauliflower__Nutrient_Deficiency` | 107 | +143 | 250 |
| `Guava__Algal_Leaf_Spot` | 70 | +80 | 150 |
| `Guava__Dry_Leaf` | 36 | +114 | 150 |
| `Guava__Healthy` | 140 | +10 | 150 |
| `Guava__Insect_Pest_Disease` | 114 | +36 | 150 |
| `Guava__Red_Rust` | 62 | +88 | 150 |
| **TOTAL (new crops)** | **3,488** | **+1,047** | **4,535** |

---

### Step 6 — Research-Driven Class Trimming

Following completion of the merged dataset, a systematic review was conducted to eliminate classes that are either low-prevalence in Bangladesh, exhibit high visual overlap with other retained classes (confusion risk), or are outside the leaf-based classification scope of this project. Decisions are grounded in agricultural reports from **BARI (Bangladesh Agricultural Research Institute, 2025–2026)** and **FAO (2025)** crop disease incidence data.

**Classes dropped and research rationale:**

| Crop | Class Dropped | Research Basis |
|---|---|---|
| Chili | `Chili__Nutrition_Deficiency` | Symptom overlap with `Chili__Bacterial_Spot`; nutritional deficiency visually ambiguous — high inter-class confusion risk. |
| Tomato | `Tomato__Leaf_Mold` | Lower incidence in Bangladesh (<10% vs. Late Blight ~30%, BARI 2026); visual similarity to `Tomato__Bacterial_Spot` and `Tomato__Early_Blight` increases confusion risk. |

**Classes evaluated but not present in dataset (no action required):**

| Crop | Class Evaluated | Outcome |
|---|---|---|
| Chili | `Chili__White_Spot` | Not found in dataset — already excluded at restructuring stage |
| Eggplant | `Eggplant__Small_Leaf_Disease` | Not found in dataset |
| Gourd | `Gourd__Angular_Leaf_Spot` | Not found in dataset |
| Gourd | `Gourd__Early_Alternaria_Leaf_Blight` | Not found in dataset |
| Potato | `Potato__Common_Scab` | Not found in dataset |
| Potato | `Potato__Black_Scurf` | Not found in dataset |
| Tomato | `Tomato__Septoria_Leaf_Spot` | Not found in dataset |
| Cauliflower | `Cauliflower__Clubroot` | Not found in dataset |
| Cauliflower | `Cauliflower__Mosaic_Virus` | Not found in dataset |
| Guava | `Guava__Anthracnose` | Not found in dataset |

**Impact of trimming:**

| Metric | Before Trim | After Trim | Change |
|---|---|---|---|
| Total Classes | 47 | **45** | −2 |
| Total Images | 18,108 | **17,506** | −602 |
| Train Images | 14,096 | **13,596** | −500 |
| Val Images | 1,971 | **1,922** | −49 |
| Test Images | 2,041 | **1,988** | −53 |

**Result at end of Step 6:** 45 classes, 9 crops, 17,506 images.

---

## 3. Final Dataset Statistics

### Grand Summary

| Split | Image Count | Percentage |
|---|---|---|
| Train | **13,596** | 77.7% |
| Validation | **1,922** | 11.0% |
| Test | **1,988** | 11.4% |
| **Grand Total** | **17,506** | 100% |

> **Total Classes:** 45 &nbsp;|&nbsp; **Total Crops:** 9 &nbsp;|&nbsp; **Split Ratio:** 70 / 15 / 15

---

### Per-Crop Breakdown

| Crop | Status | Classes | Train | Val | Test | Total |
|---|---|---|---|---|---|---|
| Cabbage | NEW | 4 | 530 | 29 | 34 | **593** |
| Cauliflower | NEW | 7 | 2,202 | 397 | 409 | **3,008** |
| Chili | Existing | 4 | 1,000 | 129 | 134 | **1,263** |
| Eggplant | Existing | 5 | 2,003 | 413 | 420 | **2,836** |
| Gourd | Existing | 4 | 800 | 77 | 84 | **961** |
| Guava | NEW | 5 | 750 | 89 | 95 | **934** |
| Potato | Existing | 3 | 2,261 | 460 | 465 | **3,186** |
| Rice | Existing | 8 | 2,800 | 267 | 280 | **3,347** |
| Tomato | Existing | 5 | 1,250 | 61 | 67 | **1,378** |
| **TOTAL** | | **45** | **13,596** | **1,922** | **1,988** | **17,506** |

---

### Class-Level Detail

#### Cabbage — 4 Classes | 593 Images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Cabbage__Alternaria_Spot | 150 *(+101 aug)* | 10 | 11 | 171 |
| Cabbage__Black_Rot | 125 *(+100 aug)* | 5 | 7 | 137 |
| Cabbage__Downy_Mildew | 105 *(+84 aug)* | 4 | 5 | 114 |
| Cabbage__Healthy_Leaf | 150 *(+103 aug)* | 10 | 11 | 171 |
| **Subtotal** | **530** | **29** | **34** | **593** |

#### Cauliflower — 7 Classes | 3,008 Images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Cauliflower__Alternaria_Disease | 250 *(+3 aug)* | 52 | 54 | 356 |
| Cauliflower__Bacterial_Soft_Rot | 250 *(+110 aug)* | 30 | 31 | 311 |
| Cauliflower__Bacterial_Spot | 283 | 60 | 62 | 405 |
| Cauliflower__Black_Spot | 250 *(+75 aug)* | 37 | 39 | 326 |
| Cauliflower__Downy_Mildew | 287 | 61 | 62 | 410 |
| Cauliflower__Healthy | 632 | 135 | 137 | 904 |
| Cauliflower__Nutrient_Deficiency | 250 *(+143 aug)* | 22 | 24 | 296 |
| **Subtotal** | **2,202** | **397** | **409** | **3,008** |

#### Chili — 4 Classes | 1,263 Images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Chili__Bacterial_Spot | 250 | 28 | 29 | 307 |
| Chili__Cercospora_Leaf_Spot | 250 | 28 | 30 | 308 |
| Chili__Curl_Virus | 250 | 36 | 37 | 323 |
| Chili__Healthy_Leaf | 250 | 37 | 38 | 325 |
| ~~Chili__Nutrition_Deficiency~~ | ~~250~~ | ~~36~~ | ~~38~~ | ~~324~~ *(dropped — Step 6)* |
| **Subtotal** | **1,000** | **129** | **134** | **1,263** |

#### Eggplant — 5 Classes | 2,836 Images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Eggplant__Healthy_Leaf | 664 | 142 | 143 | 949 |
| Eggplant__Insect_Pest_Disease | 522 | 111 | 113 | 746 |
| Eggplant__Leaf_Spot_Disease | 218 | 46 | 48 | 312 |
| Eggplant__Mosaic_Virus_Disease | 399 | 85 | 86 | 570 |
| Eggplant__Wilt_Disease | 200 | 29 | 30 | 259 |
| **Subtotal** | **2,003** | **413** | **420** | **2,836** |

#### Gourd — 4 Classes | 961 Images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Gourd__Alternaria_Leaf_Blight | 200 | 26 | 28 | 254 |
| Gourd__Downy_Mildew | 200 | 6 | 8 | 214 |
| Gourd__Healthy_Leaf | 200 | 8 | 9 | 217 |
| Gourd__Mosaic_Virus | 200 | 37 | 39 | 276 |
| **Subtotal** | **800** | **77** | **84** | **961** |

#### Guava — 5 Classes | 934 Images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Guava__Algal_Leaf_Spot | 150 *(+80 aug)* | 15 | 15 | 180 |
| Guava__Dry_Leaf | 150 *(+114 aug)* | 7 | 9 | 166 |
| Guava__Healthy | 150 *(+10 aug)* | 30 | 30 | 210 |
| Guava__Insect_Pest_Disease | 150 *(+36 aug)* | 24 | 26 | 200 |
| Guava__Red_Rust | 150 *(+88 aug)* | 13 | 15 | 178 |
| **Subtotal** | **750** | **89** | **95** | **934** |

#### Potato — 3 Classes | 3,186 Images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Potato__Early_Blight | 921 | 197 | 199 | 1,317 |
| Potato__Healthy_Leaf | 250 | 30 | 31 | 311 |
| Potato__Late_Blight | 1,090 | 233 | 235 | 1,558 |
| **Subtotal** | **2,261** | **460** | **465** | **3,186** |

#### Rice — 8 Classes | 3,347 Images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Rice__Bacterial_Leaf_Blight | 350 | 32 | 34 | 416 |
| Rice__Brown_Spot | 350 | 35 | 37 | 422 |
| Rice__Healthy_Leaf | 350 | 31 | 33 | 414 |
| Rice__Leaf_Blast | 350 | 37 | 38 | 425 |
| Rice__Leaf_Scald | 350 | 33 | 35 | 418 |
| Rice__Narrow_Brown_Leaf_Spot | 350 | 30 | 32 | 412 |
| Rice__Rice_Hispa | 350 | 33 | 34 | 417 |
| Rice__Sheath_Blight | 350 | 36 | 37 | 423 |
| **Subtotal** | **2,800** | **267** | **280** | **3,347** |

#### Tomato — 5 Classes | 1,378 Images

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Tomato__Bacterial_Spot | 250 | 16 | 17 | 283 |
| Tomato__Early_Blight | 250 | 12 | 13 | 275 |
| Tomato__Healthy_Leaf | 250 | 9 | 10 | 269 |
| Tomato__Late_Blight | 250 | 16 | 18 | 284 |
| ~~Tomato__Leaf_Mold~~ | ~~250~~ | ~~13~~ | ~~15~~ | ~~278~~ *(dropped — Step 6)* |
| Tomato__Mosaic_Virus | 250 | 8 | 9 | 267 |
| **Subtotal** | **1,250** | **61** | **67** | **1,378** |

---

## 4. Augmentation Details

Offline augmentation was applied exclusively to the **train split** to avoid data leakage into validation or test sets. Each augmented image received **2–4 randomly selected transforms** applied sequentially.

### Augmentation Techniques

| Technique | Parameter | Purpose |
|---|---|---|
| Gaussian Blur | radius 0.5–2.0 | Simulates defocus and camera hand-shake |
| Brightness Jitter | factor 0.6–1.4× | Models outdoor lighting variation |
| Contrast Jitter | factor 0.6–1.5× | Models auto-exposure compensation |
| Color / Saturation Jitter | factor 0.7–1.4× | Models white balance differences |
| Horizontal Flip | — | Models orientation-agnostic capture |
| Rotation | −20° to +20° | Models angled phone camera shots |
| Gaussian Noise | σ = 5–20 | Simulates low-quality sensor noise |
| Random Shadow Overlay | 30–70% opacity | Simulates partial shading from trees or structures |
| Sharpening | — | Simulates over-sharpened phone camera processing |

### Augmentation Safety Constraints

- **Max augmentation ratio:** 5× the original image count per class (prevents mode collapse on tiny classes).
- **Val/Test splits:** Untouched in all augmentation passes.
- **Seed:** Fixed at 42 for full reproducibility.

### Total Augmentation Summary

| Phase | Classes Augmented | Images Added |
|---|---|---|
| Phase 1 (Steps 3–4, original 31 classes) | Selected low-support classes | Varies per class |
| Phase 2 (Step 5, new 16 classes) | 13 of 16 new classes | **1,047 images** |

---

## 5. New Crops Added — Detailed Accounting

**Total new raw images collected:** 3,488

**Total augmented images added (train only):** 1,047

**Total new images in final dataset (raw + aug):** 4,535

### Source Folder to Target Class Mapping — New Crops

| Source Path (relative to Extra AgriVision Images) | Target Class |
|---|---|
| `Cabbage/Cabbage_Altenaria_spot` | `Cabbage__Alternaria_Spot` |
| `Cabbage/Cabbage - Black_rot` | `Cabbage__Black_Rot` |
| `Cabbage/Cabbage downy mildew` | `Cabbage__Downy_Mildew` |
| `Cabbage/Cabbage - Healthy` | `Cabbage__Healthy_Leaf` |
| `Cauliflower Dataset/Cauliflower Categories/Alternaria Brassicae` | `Cauliflower__Alternaria_Disease` |
| `Cauliflower Dataset/Leaves Categories/Alternaria Leaf Spot` | `Cauliflower__Alternaria_Disease` |
| `Cauliflower Dataset/Cauliflower Categories/Bacterial Soft Rot` | `Cauliflower__Bacterial_Soft_Rot` |
| `Cauliflower Dataset/Cauliflower Categories/Bacterial Spot` | `Cauliflower__Bacterial_Spot` |
| `Cauliflower Dataset/Cauliflower Categories/Black Spot` | `Cauliflower__Black_Spot` |
| `Cauliflower Dataset/Leaves Categories/Downy Mildew` | `Cauliflower__Downy_Mildew` |
| `Cauliflower Dataset/Cauliflower Categories/Healthy` + `Leaves Categories/Healthy` | `Cauliflower__Healthy` |
| `Cauliflower Dataset/Cauliflower Categories/Purple Tinges` | `Cauliflower__Nutrient_Deficiency` |
| `Guava leaves.../Algal leaves spot` | `Guava__Algal_Leaf_Spot` |
| `Guava leaves.../Dry leaves` | `Guava__Dry_Leaf` |
| `Guava leaves.../Healthly fruit` + `Healthly leaves` | `Guava__Healthy` |
| `Guava leaves.../Insects eatten class` | `Guava__Insect_Pest_Disease` |
| `Guava leaves.../Red rust` | `Guava__Red_Rust` |

---

## 6. Dataset Directory Structure

```
AgriVision_Final/
├── train/
│   ├── Cabbage__Alternaria_Spot/        (150 images)
│   ├── Cabbage__Black_Rot/              (125 images)
│   ├── Cabbage__Downy_Mildew/           (105 images)
│   ├── Cabbage__Healthy_Leaf/           (150 images)
│   ├── Cauliflower__Alternaria_Disease/ (250 images)
│   ├── Cauliflower__Bacterial_Soft_Rot/ (250 images)
│   ├── Cauliflower__Bacterial_Spot/     (283 images)
│   ├── Cauliflower__Black_Spot/         (250 images)
│   ├── Cauliflower__Downy_Mildew/       (287 images)
│   ├── Cauliflower__Healthy/            (632 images)
│   ├── Cauliflower__Nutrient_Deficiency/(250 images)
│   ├── Chili__Bacterial_Spot/           (250 images)
│   ├── Chili__Cercospora_Leaf_Spot/     (250 images)
│   ├── Chili__Curl_Virus/               (250 images)
│   ├── Chili__Healthy_Leaf/             (250 images)
│   ├── Eggplant__Healthy_Leaf/          (664 images)
│   ├── Eggplant__Insect_Pest_Disease/   (522 images)
│   ├── Eggplant__Leaf_Spot_Disease/     (218 images)
│   ├── Eggplant__Mosaic_Virus_Disease/  (399 images)
│   ├── Eggplant__Wilt_Disease/          (200 images)
│   ├── Gourd__Alternaria_Leaf_Blight/   (200 images)
│   ├── Gourd__Downy_Mildew/             (200 images)
│   ├── Gourd__Healthy_Leaf/             (200 images)
│   ├── Gourd__Mosaic_Virus/             (200 images)
│   ├── Guava__Algal_Leaf_Spot/          (150 images)
│   ├── Guava__Dry_Leaf/                 (150 images)
│   ├── Guava__Healthy/                  (150 images)
│   ├── Guava__Insect_Pest_Disease/      (150 images)
│   ├── Guava__Red_Rust/                 (150 images)
│   ├── Potato__Early_Blight/            (921 images)
│   ├── Potato__Healthy_Leaf/            (250 images)
│   ├── Potato__Late_Blight/             (1,090 images)
│   ├── Rice__Bacterial_Leaf_Blight/     (350 images)
│   ├── Rice__Brown_Spot/                (350 images)
│   ├── Rice__Healthy_Leaf/              (350 images)
│   ├── Rice__Leaf_Blast/                (350 images)
│   ├── Rice__Leaf_Scald/                (350 images)
│   ├── Rice__Narrow_Brown_Leaf_Spot/    (350 images)
│   ├── Rice__Rice_Hispa/                (350 images)
│   ├── Rice__Sheath_Blight/             (350 images)
│   ├── Tomato__Bacterial_Spot/          (250 images)
│   ├── Tomato__Early_Blight/            (250 images)
│   ├── Tomato__Healthy_Leaf/            (250 images)
│   ├── Tomato__Late_Blight/             (250 images)
│   └── Tomato__Mosaic_Virus/            (250 images)
├── val/   (same 45 class folders)
└── test/  (same 45 class folders)
```

---

## 7. Class Mapping Reference

All class names follow the convention: `Crop__DiseaseName`

| # | Class Name | Crop | Category |
|---|---|---|---|
| 1 | `Cabbage__Alternaria_Spot` | Cabbage | Disease |
| 2 | `Cabbage__Black_Rot` | Cabbage | Disease |
| 3 | `Cabbage__Downy_Mildew` | Cabbage | Disease |
| 4 | `Cabbage__Healthy_Leaf` | Cabbage | Healthy |
| 5 | `Cauliflower__Alternaria_Disease` | Cauliflower | Disease |
| 6 | `Cauliflower__Bacterial_Soft_Rot` | Cauliflower | Disease |
| 7 | `Cauliflower__Bacterial_Spot` | Cauliflower | Disease |
| 8 | `Cauliflower__Black_Spot` | Cauliflower | Disease |
| 9 | `Cauliflower__Downy_Mildew` | Cauliflower | Disease |
| 10 | `Cauliflower__Healthy` | Cauliflower | Healthy |
| 11 | `Cauliflower__Nutrient_Deficiency` | Cauliflower | Deficiency |
| 12 | `Chili__Bacterial_Spot` | Chili | Disease |
| 13 | `Chili__Cercospora_Leaf_Spot` | Chili | Disease |
| 14 | `Chili__Curl_Virus` | Chili | Disease |
| 15 | `Chili__Healthy_Leaf` | Chili | Healthy |
| 16 | `Eggplant__Healthy_Leaf` | Eggplant | Healthy |
| 17 | `Eggplant__Insect_Pest_Disease` | Eggplant | Disease |
| 18 | `Eggplant__Leaf_Spot_Disease` | Eggplant | Disease |
| 19 | `Eggplant__Mosaic_Virus_Disease` | Eggplant | Disease |
| 20 | `Eggplant__Wilt_Disease` | Eggplant | Disease |
| 21 | `Gourd__Alternaria_Leaf_Blight` | Gourd | Disease |
| 22 | `Gourd__Downy_Mildew` | Gourd | Disease |
| 23 | `Gourd__Healthy_Leaf` | Gourd | Healthy |
| 24 | `Gourd__Mosaic_Virus` | Gourd | Disease |
| 25 | `Guava__Algal_Leaf_Spot` | Guava | Disease |
| 26 | `Guava__Dry_Leaf` | Guava | Disease |
| 27 | `Guava__Healthy` | Guava | Healthy |
| 28 | `Guava__Insect_Pest_Disease` | Guava | Disease |
| 29 | `Guava__Red_Rust` | Guava | Disease |
| 30 | `Potato__Early_Blight` | Potato | Disease |
| 31 | `Potato__Healthy_Leaf` | Potato | Healthy |
| 32 | `Potato__Late_Blight` | Potato | Disease |
| 33 | `Rice__Bacterial_Leaf_Blight` | Rice | Disease |
| 34 | `Rice__Brown_Spot` | Rice | Disease |
| 35 | `Rice__Healthy_Leaf` | Rice | Healthy |
| 36 | `Rice__Leaf_Blast` | Rice | Disease |
| 37 | `Rice__Leaf_Scald` | Rice | Disease |
| 38 | `Rice__Narrow_Brown_Leaf_Spot` | Rice | Disease |
| 39 | `Rice__Rice_Hispa` | Rice | Disease |
| 40 | `Rice__Sheath_Blight` | Rice | Disease |
| 41 | `Tomato__Bacterial_Spot` | Tomato | Disease |
| 42 | `Tomato__Early_Blight` | Tomato | Disease |
| 43 | `Tomato__Healthy_Leaf` | Tomato | Healthy |
| 44 | `Tomato__Late_Blight` | Tomato | Disease |
| 45 | `Tomato__Mosaic_Virus` | Tomato | Disease |

**Breakdown by category:** 35 Disease classes · 9 Healthy classes · 1 Deficiency class (Nutrient)

---

## 8. Training Configuration Recommendations

The following configuration is recommended for baseline model training on this dataset, targeting Google Colab (T4 GPU) or Kaggle (P100 GPU) environments.

| Parameter | Recommended Value |
|---|---|
| **Model Architecture** | EfficientNet-B2 or ResNet-50 (pretrained on ImageNet) |
| **Input Resolution** | 224×224 (or 260×260 for EfficientNet-B2) |
| **Batch Size** | 32 (T4) · 64 (P100) |
| **Epochs** | 30–50 with early stopping (patience = 7) |
| **Optimizer** | AdamW (lr = 1e-4, weight_decay = 1e-4) |
| **LR Scheduler** | CosineAnnealingLR or ReduceLROnPlateau |
| **Loss Function** | CrossEntropyLoss with inverse-frequency class weights |
| **Online Augmentation** | RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation |
| **Estimated Training Time** | ~2–4 hours (T4) · ~1.5–3 hours (P100) |

> **Important:** Inverse-frequency class weighting is strongly recommended to compensate for the imbalance between large classes (e.g., `Potato__Late_Blight`: ~1,090 train images) and small augmented classes (e.g., Cabbage/Guava: ~105–150 train images). This significantly improves recall on minority classes.

### Recommended Online Augmentation Pipeline (Training Only)

```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

---

## 9. Known Limitations & Notes

| Issue | Description |
|---|---|
| **Class Imbalance** | Significant imbalance exists between crops. Potato Late Blight (~1,090 train) vs. Cabbage Downy Mildew (~105 train). Mitigated by class weighting and offline augmentation, but remains a consideration during evaluation. |
| **Augmented Data Provenance** | Augmented images in the train split are synthetic derivatives. They are not counted as independent real-world samples. |
| **Cabbage Downy Mildew (small class)** | Only 21 original images available; train set reaches 105 after augmentation, still the smallest class in the dataset. Additional real data collection is recommended for this class if accuracy is unsatisfactory. |
| **Guava Dry Leaf (small class)** | Only 36 original images; augmented to 150 in train. Similar caution applies. |
| **Val/Test Sizes** | Some val/test splits are very small for certain Cabbage and Tomato classes due to the limited original image counts. Evaluation metrics on these classes should be interpreted with caution. |
| **Reproducibility** | All splits and augmentations are deterministic under seed = 42; re-running any script will produce identical output. |

---

*Report generated: March 3, 2026*
*Dataset version: AgriVision Final v3 (Post Class Trimming)*
*Output directory: `D:\499A Dataset\AgriVision\AgriVision`*
