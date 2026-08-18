# Multi-Subject Analysis - Quick Reference Card

## 🚀 Quick Start (Copy & Paste)

```python
# 1. Define your subjects
subjects = [
    {'name': 'Tomer', 'xdf_pattern': 'Tomer'},
    {'name': 'Noam', 'xdf_pattern': 'Noam'},
]

# 2. Run analysis
tracker = SubjectPerformanceTracker()
results = run_group_analysis(subjects, params_dict, tracker)

# 3. View results
tracker.print_summary()
plot_group_performance_comparison(tracker)
export_group_results_to_csv(tracker)
```

## 📊 Common Operations

### View Group Summary
```python
tracker.print_summary()
```
Output: Mean/std/min/max for accuracy, F1, etc.

### Get Results as Table
```python
df = tracker.get_dataframe()
print(df)
```
Output: pandas DataFrame with all subjects

### Export to CSV
```python
export_group_results_to_csv(tracker)
```
Output: CSV file with summary metrics

### Plot Comparisons
```python
plot_group_performance_comparison(tracker)
```
Output: 4-panel figure with all metrics

### Get One Subject's Result
```python
subject_result = get_subject_result(results, 'Tomer')
print(subject_result['metrics'])
```

### Compare Two Subjects
```python
compare_two_subjects(results, 'Tomer', 'Noam')
```
Output: Side-by-side metric comparison

### Save Subject Report
```python
create_individual_subject_report(subject_result)
```
Output: Text file with metrics and confusion matrix

### Load Saved Results
```python
saved = load_group_results()
```
Output: Dictionary from group_results.json

## 📁 Output Files

| File | Format | Content |
|------|--------|---------|
| `group_results.json` | JSON | All metrics + statistics |
| `group_performance_summary.csv` | CSV | Subject scores + group mean |

## 🔧 Customize

### Change Save Location
```python
tracker = SubjectPerformanceTracker(
    save_path=current_path / 'my_results.json'
)
```

### Skip Auto-Save
```python
run_group_analysis(subjects, params_dict, tracker, save_results=False)
```

### Custom Parameters per Subject
```python
params = copy.deepcopy(params_dict)
params['LowPass'] = 8
result = process_single_subject('SubjectName', 'pattern', params)
```

## 📈 Key Metrics Tracked

- **Accuracy** - Overall correct predictions
- **F1 Score** - Harmonic mean of precision & recall
- **Precision** - True positives / all positives
- **Recall** - True positives / true labels
- **Epochs** - Number of samples used
- **Confusion Matrix** - Per-class performance

## 🎯 Typical Workflow

```
1. Define subject_list
   ↓
2. Initialize tracker = SubjectPerformanceTracker()
   ↓
3. Run run_group_analysis(subjects, params, tracker)
   ↓
4. Print tracker.print_summary()
   ↓
5. Visualize plot_group_performance_comparison(tracker)
   ↓
6. Export export_group_results_to_csv(tracker)
   ↓
7. Analyze DataFrame for insights
```

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| "No XDF files found" | Check `xdf_pattern` matches filename |
| Memory errors | Reduce augmentation params or process fewer subjects |
| Slow processing | Normal - takes 10-30 min per subject |
| Missing files | Ensure Recordings/ folder exists with XDF files |
| `Zero or infinite position found in chs` (CSD) | FCz has a NaN location: `set_montage` ran *before* `add_reference_channels`. See `apply_montage_and_reference` — that order is load-bearing. |
| Live stream: `live channels != model channels` | The model was trained with a different `AddRefChannel` setting than the loaded `params_dict`. Re-dump model + picks + params_dict together. |
| Group stacking fails on channel count | Some subjects at 29 channels, some at 30. Do not mix models trained with and without `AddRefChannel`. |

## 🔌 FCz / online reference

The amplifier's online reference (FCz) is not in the recorded data.
`params_dict['AddRefChannel'] = True` rebuilds it as a real channel before average
referencing, taking `FC+C+CP+P` from **29 to 30 channels**.

```python
params_dict['PerformAvgRef'] = True    # required
params_dict['AddRefChannel'] = True    # default False everywhere else
# and add 'FCz' to the 'FC' entry of Electorde_Groups
```

**Default is off**, so notebooks that don't set it — and every existing
`Models/*.joblib` and `TFRs*/` cache — are bit-for-bit unaffected. Turning it on
requires retraining. Full detail in [ARCHITECTURE.md](ARCHITECTURE.md#reference-handling-and-the-fcz-channel).

## 🎯 Per-class centering (`CenterByClass`)

```python
params_dict['CenterByClass'] = True    # default; the historical behaviour
```

`EEG_Preprocessing` subtracts from each epoch the mean of all epochs of **that
epoch's own class**, computed per XDF file. It is **label-dependent** — you must
know a trial's class to centre it — so the live loop cannot apply it, and a model
trained with it on is served uncentered data online. It also removes the per-class
evoked response, which is what makes everything downstream *induced* power.

Set it to `False` to leave the evoked response in. The benchmark stack
(`Windowed_`/`Session_`/`Full_Epoch_Analysis.ipynb`) ignores the value in its params
cell: it builds **both** a centered and an uncentered copy of every subject's epochs
and pairs them into `<train>2<test>` modes, which is where the cost of this transform
at inference time is measured. Each notebook runs its own set:

| notebook | modes |
|---|---|
| `Windowed_Analysis` | `c2c`, `c2u`, `u2c`, `u2u` — the full 2×2 train/test matrix |
| `Session_Analysis` | `c2c`, `c2u`, `u2u` — `c2u` is the leak-free one under the `cross` split |
| `Full_Epoch_Analysis` | `c2c` only by default; `DIAGNOSTIC_MODES` adds `g2g`/`g2u`/`l2l`/`l2u`/`c2u`/`u2u` |

The letter is the centering recipe each side used: `c` = class means per XDF file
(what `CenterByClass=True` produces), `g` = global pooled over the subject,
`l` = fold-local (training trials only), `u` = not centered.

## 💡 Tips

- Use `verbose=True` for detailed processing output
- Save results with `save_results=True` for later analysis
- Convert to DataFrame with `get_dataframe()` for advanced analysis
- Use `copy.deepcopy(params_dict)` to avoid modifying original
- Check `group_results.json` for complete metric details

## 📚 Documentation

- **MULTI_SUBJECT_ANALYSIS_GUIDE.md** - Full reference
- **MULTI_SUBJECT_EXAMPLES.md** - Code examples
- **CHANGES_SUMMARY.md** - What was added

## 🔍 Inspect Results

```python
# All results as DataFrame
df = tracker.get_dataframe()

# One subject
result = get_subject_result(results, 'Tomer')

# Metrics only
metrics = result['metrics']

# Classifier model
model = result['classifier']

# Predictions
predictions = result['predictions']

# Ground truth
y_true = result['y_true']

# Confusion matrix
cm = result['metrics']['confusion_matrix']
```

## 📊 Simple Analysis

```python
# Get average performance
df = tracker.get_dataframe()
mean_acc = df['accuracy'].mean()
std_acc = df['accuracy'].std()
print(f"Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

# Find best/worst performer
best = df.loc[df['accuracy'].idxmax()]
worst = df.loc[df['accuracy'].idxmin()]
print(f"Best: {best['subject']} ({best['accuracy']:.3f})")
print(f"Worst: {worst['subject']} ({worst['accuracy']:.3f})")
```

---

**Version**: 1.0  
**Added**: February 2026  
**Location**: Main.ipynb cells after line 1681
