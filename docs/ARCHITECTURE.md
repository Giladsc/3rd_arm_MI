# System Architecture Overview

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-SUBJECT ANALYSIS SYSTEM                │
└─────────────────────────────────────────────────────────────────┘

                          ┌──────────────────┐
                          │  Subject List    │
                          │  Definition      │
                          └────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  run_group_analysis()       │
                    │  (Orchestrator)             │
                    └──────────────┬──────────────┘
                                   │
            ┌──────────────┬────────┴────────┬──────────────┐
            │              │                 │              │
    ┌───────▼─────┐ ┌──────▼──────┐ ┌───────▼─────┐ ┌──────▼──────┐
    │  Subject 1  │ │  Subject 2  │ │  Subject 3  │ │  Subject N  │
    └───────┬─────┘ └──────┬──────┘ └───────┬─────┘ └──────┬──────┘
            │              │               │              │
            │ process_single_subject()     │              │
            │              │               │              │
    ┌───────▼──────────────▼───────────────▼──────────────▼────────┐
    │                    Single Subject Pipeline                    │
    ├───────────────────────────────────────────────────────────────┤
    │  1. Load XDF files                                            │
    │  2. Preprocessing (EEG_Preprocessing)                         │
    │  3. Balance epochs                                            │
    │  4. Split train/validation                                    │
    │  5. Augment data                                              │
    │  6. Train classifier                                          │
    │  7. Make predictions                                          │
    │  8. Calculate metrics (accuracy, F1, precision, recall)       │
    │  9. Return result dictionary                                  │
    └───────────────────────────────────────────────────────────────┘
            │              │               │              │
            └──────────────┴───────────────┴──────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │  SubjectPerformanceTracker          │
        │  (Aggregator)                       │
        ├──────────────────────────────────────┤
        │  • add_subject()                     │
        │  • compute_group_statistics()        │
        │  • print_summary()                   │
        │  • save_results()                    │
        │  • get_dataframe()                   │
        └──────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
    ┌───▼────────────────┐      ┌──────────────▼──────────┐
    │  JSON Output       │      │  CSV Output              │
    │  (Complete)        │      │  (Summary Table)         │
    └────────────────────┘      └──────────────────────────┘
        • All metrics          • Subject names
        • Confusion matrices   • Accuracy/F1/Precision
        • Statistics           • Total epochs
        • Timestamps           • Group means


```

## Data Flow

```
XDF FILES
   │
   ▼
┌─────────────────────────────────────┐
│   EEG_Preprocessing()               │
│   • Filter & montage setup          │
│   • ICA/Autoreject                  │
│   • Re-reference (CSD/AvgRef)       │
│     └ optional FCz reconstruction   │
│       (params_dict['AddRefChannel'])│
└────────────┬────────────────────────┘
             │
             ▼
        EPOCHS
             │
    ┌────────┴────────┐
    ▼                 ▼
TRAIN         VALIDATION
SET            SET
    │                 │
    ▼                 ▼
AUGMENT          CROP
DATA         (window size)
    │                 │
    ▼                 ▼
CLASSIFIER      PREDICTIONS
 TRAINING            │
    │                ▼
    └───────────────▶METRICS
                   (Accuracy, F1, etc.)
                     │
                     ▼
            ┌─────────────────┐
            │ Result Dict per │
            │    Subject      │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Aggregator     │
            │  (Tracker)      │
            └────────┬────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      JSON                  CSV
      Output               Output
```

## Function Call Hierarchy

```
run_group_analysis()
├── process_single_subject()  [for each subject]
│   ├── read_raw_xdf()
│   ├── EEG_Preprocessing()
│   ├── remap_epoch_events_to_standard()
│   ├── mne.concatenate_epochs()
│   ├── balance_epochs_by_subsampling()
│   ├── Split_training_validation()
│   ├── crop_the_data()
│   ├── augment_and_oversample_data_v4()
│   ├── classifier_training()
│   ├── clf.predict()
│   ├── calculate_performance_metrics()
│   │   ├── accuracy_score()
│   │   ├── f1_score()
│   │   ├── precision_score()
│   │   ├── recall_score()
│   │   └── confusion_matrix()
│   └── return result_dict
│
├── performance_tracker.add_subject()  [for each result]
├── performance_tracker.compute_group_statistics()
├── performance_tracker.print_summary()
└── performance_tracker.save_results()
    └── json.dump() to group_results.json
```

## Class Relationships

```
┌──────────────────────────────────────┐
│ SubjectPerformanceTracker            │
├──────────────────────────────────────┤
│ Properties:                          │
│  • subjects_data: dict               │
│  • group_stats: dict                 │
│  • save_path: Path                   │
├──────────────────────────────────────┤
│ Methods:                             │
│  • add_subject(name, metrics)        │
│  • compute_group_statistics()        │
│  • print_summary()                   │
│  • save_results()                    │
│  • get_dataframe() → DataFrame       │
└──────────────────────────────────────┘
         ▲
         │ returns
         │
    used by run_group_analysis()


Result Dict Structure:
┌────────────────────────────────────┐
│ {                                  │
│   'subject_name': str              │
│   'epochs': MNE Epochs             │
│   'classifier': sklearn model      │
│   'predictions': np.array          │
│   'y_true': np.array               │
│   'metrics': {                     │
│     'accuracy': float              │
│     'f1_score': float              │
│     'precision': float             │
│     'recall': float                │
│     'epochs': int                  │
│     'confusion_matrix': np.array   │
│   }                                │
│   'train_inds': np.array           │
│   'validation_inds': np.array      │
│ }                                  │
└────────────────────────────────────┘
```

## Reference Handling and the FCz Channel

The amplifier records with **FCz as the online reference**, so FCz is a physical
electrode but never appears in the XDF/LSL stream — the data carries only the 64
differential channels. `Montages/CACS-64_REF.bvef` names that position `REF`
(Theta=23, Phi=90).

`params_dict['AddRefChannel']` (**default `False`**) reconstructs it. When enabled,
FCz is re-added as a zero-filled channel *before* average referencing, so the
average reference leaves it holding `-(sum of every other channel)`, i.e.
`-(64/65) * mean(recorded channels)` — the estimated potential at the reference
site. `FC+C+CP+P` then yields **30 channels instead of 29**.

Three functions in `src/preprocessing.py` own this, and are called by **both** the
offline pipeline and the live-stream loop so the two cannot drift:

| Function | Role |
|---|---|
| `load_montage(current_path)` | Reads the .bvef and renames `REF` → `FCz`. The montage file itself is never modified. |
| `apply_montage_and_reference(inst, montage, params_dict)` | montage + optional FCz reconstruction + bad-electrode drop + average reference. Returns `(inst, elecs_to_drop)`. |
| `select_electrodes(inst, params_dict, elecs_to_drop)` | The pick list, single-sourced. Defines the channel *order* the classifier sees. |

### The step order is load-bearing

Inside `apply_montage_and_reference`: **`add_reference_channels` must run before
`set_montage`, and both before the average reference.**

- `add_reference_channels` refuses to run at all once an average-reference
  projection is active.
- It positions the new channel from an EEG dig point with `ident==0`. A
  bvef-derived montage has none, so if `set_montage` has already run, FCz's
  location is left **NaN** — silently, since MNE 1.6 emits no catchable warning.
  You only discover it when `compute_current_source_density` dies with
  `Zero or infinite position found in chs`.

The regression check is therefore on the *location*, not on a warning: after
preprocessing, `raw.info['chs'][idx]['loc'][:3]` for FCz must be finite and
non-zero (it should be `(0, 37.1, 87.4)` mm).

### Why this is safe online

Average referencing is per-time-sample and purely spatial, so the reconstructed
FCz is **bit-identical** whether computed over a whole recording or a 55-sample
live chunk — verified against `BA_MI1.xdf` with `np.array_equal`. What must match
between training and inference is the channel *set* at referencing time, which is
exactly what calling the shared helper from both places guarantees.

### Caveats

- **FCz is not an independent measurement.** It is the common-mode estimate, so
  against the *full* 64-channel set it adds no information to a linear model. It
  adds real information only to a channel subset, where it carries the aggregate
  of the electrodes the subset excludes.
- **Rank.** After average referencing the 65-channel set is exactly rank 64
  (`FCz = -Σ(others)`). Picking a 30-of-65 subset is generically full rank and
  safe. Anything consuming the *full* set is not: `CSP(reg=None)`,
  `Covariances(estimator='scm'|'lwf')`, ICA (pass `n_components ≤ 64`), and
  `compute_covariance` (pass `rank='info'`).
- **Artifact compatibility.** Existing `Models/*.joblib` and the `TFRs*/` caches
  were built on 29 channels. Enabling the flag requires retraining, and model,
  channel list and `params_dict` must be re-dumped **together** — the live
  notebook loads all three from pickles and asserts they agree.
- **ICA breaks the identity.** After component removal `FCz = -Σ(others)` no
  longer holds unless the average reference is re-applied post-ICA. Keep the flag
  off for the ICA path.

Currently enabled only in `notebooks/Main_Experiment.ipynb`. `Main.ipynb`,
`Main2.ipynb`, `Main_Experiment_ICA.ipynb`, `TFR_Analysis.ipynb` and
`src/tfr_batch.py` do not set it and are bit-for-bit unaffected.

## Processing Pipeline Detail

```
For Each Subject:
┌────────────────────────────────────────────────────────┐
│                                                        │
│  1. LOAD PHASE                                         │
│     ├─ Find XDF files matching pattern                │
│     ├─ read_raw_xdf() for each file                   │
│     └─ List of Raw objects                            │
│                                                        │
│  2. PREPROCESSING PHASE                                │
│     ├─ EEG_Preprocessing() on each raw                │
│     │  ├─ Filtering                                   │
│     │  ├─ ICA/Autoreject                              │
│     │  ├─ Re-referencing                              │
│     │  └─ Epoching                                    │
│     ├─ remap_epoch_events_to_standard()               │
│     ├─ mne.concatenate_epochs()                       │
│     └─ Single Epochs object                           │
│                                                        │
│  3. BALANCE PHASE                                      │
│     ├─ balance_epochs_by_subsampling()                │
│     └─ Balanced Epochs object                         │
│                                                        │
│  4. SPLIT PHASE                                        │
│     ├─ Split_training_validation()                    │
│     ├─ train_inds: indices for training               │
│     └─ validation_inds: indices for testing           │
│                                                        │
│  5. PREPARATION PHASE                                  │
│     ├─ crop_the_data()                                │
│     ├─ train_set_data, train_set_labels               │
│     └─ validation_set_data, validation_set_labels     │
│                                                        │
│  6. AUGMENTATION PHASE                                 │
│     ├─ augment_and_oversample_data_v4()               │
│     ├─ augmented_x: expanded features                 │
│     └─ augmented_y: expanded labels                   │
│                                                        │
│  7. TRAINING PHASE                                     │
│     ├─ classifier_training()                          │
│     ├─ Pipeline: CSP + LDA/SVM                        │
│     └─ clf: trained classifier                        │
│                                                        │
│  8. PREDICTION PHASE                                   │
│     ├─ clf.predict(validation_set_data)               │
│     ├─ predictions: model outputs                     │
│     └─ y_true: ground truth labels                    │
│                                                        │
│  9. EVALUATION PHASE                                   │
│     ├─ calculate_performance_metrics()                │
│     ├─ accuracy, f1, precision, recall                │
│     ├─ confusion_matrix                               │
│     └─ metrics_dict                                   │
│                                                        │
│  10. RETURN PHASE                                      │
│      └─ result_dict with all info                     │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Typical Output Example

```
======================================================================
INDIVIDUAL SUBJECT REPORT
======================================================================

Subject: Tomer
Generated: 2026-02-05T14:30:45.123456

PERFORMANCE METRICS:
--------------------
Accuracy:  0.8234
F1 Score:  0.8156
Precision: 0.8312
Recall:    0.8089
Total Epochs: 342

CONFUSION MATRIX:
-----------------
[[180  12   8   2]
 [ 15  92   5   0]
 [ 10   6  85   3]
 [  3   0   1  18]]

======================================================================


============================================================
GROUP PERFORMANCE SUMMARY
============================================================
Number of subjects: 3
Total epochs: 1256

--- ACCURACY ---
  Mean: 0.7956
  Std:  0.0412
  Range: [0.7623, 0.8456]

--- F1 SCORE ---
  Mean: 0.7891
  Std:  0.0458
  Range: [0.7410, 0.8523]

Subjects included:
  1. Tomer
  2. Noam
  3. Subject3
============================================================
```

## Integration Points with Existing Code

```
Existing Code          │    New System
────────────────────────────────────────
params_dict ──────────▶ run_group_analysis()
                            │
read_raw_xdf ────────────── process_single_subject()
                            │
EEG_Preprocessing ─────────┤
                            │
classifier_training ──────┤
                            │
SubjectPerformanceTracker ─▶ save_results()
                            │
                        group_results.json
                        group_performance_summary.csv
```

## File Structure

```
3rd_arm_MI/
├── Main.ipynb                              (Modified - new cells added)
├── Recordings/
│   ├── Tomer_*.xdf
│   ├── Noam_*.xdf
│   └── ...
├── Models/
│   └── (saved classifiers)
├── group_results.json                      (Generated - new)
├── group_performance_summary.csv           (Generated - new)
├── MULTI_SUBJECT_ANALYSIS_GUIDE.md         (New)
├── MULTI_SUBJECT_EXAMPLES.md               (New)
├── CHANGES_SUMMARY.md                      (New)
└── QUICK_REFERENCE.md                      (New)
```

---

This architecture provides a modular, extensible system for multi-subject analysis
while maintaining compatibility with your existing preprocessing and classification code.
