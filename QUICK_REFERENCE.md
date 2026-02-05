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
