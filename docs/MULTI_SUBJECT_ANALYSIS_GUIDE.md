# Multi-Subject Performance Analysis System

## Overview

A comprehensive system has been added to your notebook to automate running multiple subjects, tracking their performance scores, and aggregating results across the group.

## Key Features

✅ **Automated Single-Subject Pipeline** - Processes one subject end-to-end  
✅ **Batch Group Processing** - Run multiple subjects sequentially  
✅ **Performance Tracking** - Automatically collects accuracy, F1, precision, recall  
✅ **Statistical Aggregation** - Computes mean, std, min, max across group  
✅ **JSON Export** - Saves all results to structured JSON file  
✅ **CSV Export** - Exports summary metrics for further analysis  
✅ **Visualization** - Creates comparison plots across subjects  
✅ **Individual Reports** - Generates per-subject report files  

## Quick Start

### Step 1: Define Your Subject List

```python
subject_list = [
    {'name': 'Tomer', 'xdf_pattern': 'Tomer'},
    {'name': 'Noam', 'xdf_pattern': 'Noam'},
    {'name': 'Subject3', 'xdf_pattern': 'Subject3'},
]
```

The `'xdf_pattern'` is used to find matching XDF files in the `Recordings/` folder.

### Step 2: Run the Analysis

```python
# Initialize tracker
group_tracker = SubjectPerformanceTracker()

# Run analysis for all subjects
all_results = run_group_analysis(
    subject_list, 
    params_dict,  # Your existing parameter dictionary
    group_tracker,
    save_results=True  # Saves to group_results.json
)
```

### Step 3: View Results

```python
# Print summary statistics
group_tracker.print_summary()

# Visualize performance comparison
plot_group_performance_comparison(group_tracker)

# Export to CSV
export_group_results_to_csv(group_tracker)
```

## Data Structure

### Results Dictionary (all_results)

Each subject processing returns:
```python
{
    'subject_name': str,
    'epochs': MNE Epochs object,
    'classifier': trained sklearn classifier,
    'predictions': np.array of predictions,
    'y_true': np.array of ground truth labels,
    'metrics': {
        'accuracy': float,
        'f1_score': float,
        'precision': float,
        'recall': float,
        'epochs': int,
        'confusion_matrix': np.array
    },
    'train_inds': np.array,
    'validation_inds': np.array
}
```

### Group Statistics

Computed after all subjects are processed:
```python
{
    'n_subjects': int,
    'accuracy': {
        'mean': float,
        'std': float,
        'min': float,
        'max': float
    },
    'f1_score': {
        'mean': float,
        'std': float,
        'min': float,
        'max': float
    },
    'total_epochs': int,
    'subjects': list of subject names
}
```

## Available Functions

### Core Processing

| Function | Purpose |
|----------|---------|
| `process_single_subject()` | Run pipeline for one subject |
| `run_group_analysis()` | Orchestrate multi-subject processing |

### Metrics & Calculation

| Function | Purpose |
|----------|---------|
| `calculate_performance_metrics()` | Compute accuracy, F1, precision, recall from predictions |

### Analysis & Reporting

| Function | Purpose |
|----------|---------|
| `SubjectPerformanceTracker` | Main class for tracking and aggregation |
| `plot_group_performance_comparison()` | Create comparison bar charts |
| `export_group_results_to_csv()` | Save results to CSV |
| `get_subject_result()` | Retrieve specific subject's data |
| `compare_two_subjects()` | Side-by-side subject comparison |
| `create_individual_subject_report()` | Generate text report for subject |
| `load_group_results()` | Load previously saved JSON results |

## Output Files

When `save_results=True`, these files are created in your project directory:

```
3rd_arm_MI/
├── group_results.json              # Complete results in JSON
├── group_performance_summary.csv    # Summary metrics in CSV
└── {subject_name}_report.txt        # Individual reports (optional)
```

## Example: Complete Workflow

```python
# 1. Setup
subject_list = [
    {'name': 'Subject1', 'xdf_pattern': 'Subject1'},
    {'name': 'Subject2', 'xdf_pattern': 'Subject2'},
]
tracker = SubjectPerformanceTracker()

# 2. Run analysis
results = run_group_analysis(subject_list, params_dict, tracker)

# 3. View group stats
tracker.print_summary()

# 4. Create visualizations
plot_group_performance_comparison(tracker)

# 5. Export results
df = export_group_results_to_csv(tracker)

# 6. Individual analysis
subject1_result = get_subject_result(results, 'Subject1')
compare_two_subjects(results, 'Subject1', 'Subject2')
create_individual_subject_report(subject1_result)

# 7. Load results later
saved_results = load_group_results('group_results.json')
```

## Customization

### Modifying Parameters per Subject

```python
def run_group_analysis_custom(subject_list, base_params, tracker):
    for subject_info in subject_list:
        params = copy.deepcopy(base_params)
        
        # Customize parameters for this subject
        if subject_info['name'] == 'SpecialSubject':
            params['LowPass'] = 8
            params['HighPass'] = 30
        
        result = process_single_subject(
            subject_info['name'],
            subject_info['xdf_pattern'],
            params
        )
        tracker.add_subject(subject_info['name'], result['metrics'])
```

### Adding Custom Metrics

Modify `calculate_performance_metrics()` to include custom metrics:

```python
def calculate_performance_metrics(y_true, y_pred, confusion_matrices_list=None):
    metrics_dict = {...}
    # Add custom metrics
    metrics_dict['sensitivity'] = recall_score(y_true, y_pred, average='weighted')
    metrics_dict['specificity'] = ...
    return metrics_dict
```

## Troubleshooting

### Issue: "No XDF files found"
- Check that XDF files are in `Recordings/` folder
- Verify `'xdf_pattern'` matches your filenames exactly
- File names should be like: `SubjectName_xxx.xdf`

### Issue: Processing takes too long
- Check for large XDF files (can take 10-30 min per subject)
- Consider processing fewer subjects or using a subset of data
- Monitor memory usage with `psutil.virtual_memory()`

### Issue: Memory errors
- Set `params_dict['augmentation_params']` to smaller values
- Process subjects one at a time instead of in batch
- Clear variables between runs: `del epochs, raw, clf`

## Advanced Usage

### Parallel Processing (Future Enhancement)

Currently processes subjects sequentially. For parallel processing:

```python
from concurrent.futures import ProcessPoolExecutor

def process_subject_task(subject_info, params):
    return process_single_subject(
        subject_info['name'],
        subject_info['xdf_pattern'],
        params
    )

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(
        process_subject_task,
        subject_list,
        [params_dict] * len(subject_list)
    ))
```

### Statistical Analysis

```python
from scipy import stats

df = tracker.get_dataframe()

# Correlation with external variables
corr = df['accuracy'].corr(df['epochs'])

# Significance testing
t_stat, p_val = stats.ttest_1samp(df['accuracy'], 0.5)

# Group comparison
group1_acc = df[df['subject'].isin(['S1', 'S2'])]['accuracy']
group2_acc = df[df['subject'].isin(['S3', 'S4'])]['accuracy']
t_stat, p_val = stats.ttest_ind(group1_acc, group2_acc)
```

## Notes

- All results are saved with timestamps for reproducibility
- Parameters are logged with each subject's results
- Group statistics are automatically computed after processing
- Results can be loaded and re-analyzed without re-running

## Support

For issues or questions:
1. Check output messages during processing
2. Review individual subject reports
3. Check `group_results.json` for detailed metrics
4. Examine confusion matrices in the results
