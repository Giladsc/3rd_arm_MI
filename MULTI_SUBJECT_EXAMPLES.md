# Multi-Subject Analysis - Code Examples

## Example 1: Basic Group Analysis

```python
# Define subjects
subject_list = [
    {'name': 'Tomer', 'xdf_pattern': 'Tomer'},
    {'name': 'Noam', 'xdf_pattern': 'Noam'},
]

# Initialize tracker
tracker = SubjectPerformanceTracker(save_path=current_path / 'my_group_results.json')

# Run analysis
all_results = run_group_analysis(
    subject_list, 
    params_dict,  # Your existing params_dict
    tracker,
    save_results=True
)

# Display results
tracker.print_summary()
plot_group_performance_comparison(tracker)
```

## Example 2: Process One Subject at a Time

```python
# Process first subject
result_tomer = process_single_subject(
    'Tomer',
    'Tomer',  # XDF file pattern
    params_dict
)

# Check metrics
print(f"Tomer Accuracy: {result_tomer['metrics']['accuracy']:.4f}")
print(f"Tomer F1 Score: {result_tomer['metrics']['f1_score']:.4f}")

# Add to tracker
tracker.add_subject('Tomer', result_tomer['metrics'])
```

## Example 3: Compare Individual Subjects

```python
# Get a specific subject's result
tomer_result = get_subject_result(all_results, 'Tomer')

# Print their metrics
print(f"Accuracy: {tomer_result['metrics']['accuracy']:.4f}")
print(f"Confusion Matrix:\n{tomer_result['metrics']['confusion_matrix']}")

# Compare two subjects
compare_two_subjects(all_results, 'Tomer', 'Noam')

# Generate report
create_individual_subject_report(tomer_result)
```

## Example 4: Export and Analyze Results

```python
# Export to CSV
df = export_group_results_to_csv(tracker, filename='my_results.csv')

# View as table
print(df)

# Filter high performers
high_performers = df[df['accuracy'] > 0.7]
print(f"High performers: {high_performers['subject'].tolist()}")

# Calculate group mean ± std
mean_acc = df['accuracy'].mean()
std_acc = df['accuracy'].std()
print(f"Group Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
```

## Example 5: Custom Parameters per Subject

```python
def process_group_with_custom_params():
    subject_configs = [
        {
            'name': 'Tomer',
            'xdf_pattern': 'Tomer',
            'custom_params': {'LowPass': 6, 'HighPass': 28}
        },
        {
            'name': 'Noam',
            'xdf_pattern': 'Noam',
            'custom_params': {'LowPass': 8, 'HighPass': 35}
        },
    ]
    
    tracker = SubjectPerformanceTracker()
    all_results = []
    
    for config in subject_configs:
        # Copy base params and apply custom settings
        params = copy.deepcopy(params_dict)
        params.update(config['custom_params'])
        
        # Process subject
        result = process_single_subject(
            config['name'],
            config['xdf_pattern'],
            params
        )
        
        if result:
            all_results.append(result)
            tracker.add_subject(config['name'], result['metrics'])
    
    tracker.compute_group_statistics()
    tracker.print_summary()
    
    return all_results, tracker

results, tracker = process_group_with_custom_params()
```

## Example 6: Reload and Reanalyze Saved Results

```python
# Load previously saved results
saved_results = load_group_results(current_path / 'group_results.json')

# Reconstruct tracker from saved data
tracker_loaded = SubjectPerformanceTracker()

for subject_name, data in saved_results['subjects_data'].items():
    tracker_loaded.add_subject(subject_name, data['metrics'])

tracker_loaded.compute_group_statistics()
tracker_loaded.print_summary()

# Convert to DataFrame for analysis
df = tracker_loaded.get_dataframe()
```

## Example 7: Visualize and Compare Performance

```python
# Create comparison visualizations
fig, axes = plot_group_performance_comparison(tracker)

# Additional custom plot: Accuracy vs Epochs
df = tracker.get_dataframe()
plt.figure(figsize=(10, 6))
plt.scatter(df['epochs'], df['accuracy'], s=100, alpha=0.6)
for idx, row in df.iterrows():
    plt.annotate(row['subject'], (row['epochs'], row['accuracy']))
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Epochs')
plt.grid(True, alpha=0.3)
plt.show()
```

## Example 8: Batch Analysis with Error Handling

```python
def robust_group_analysis(subject_list, params_dict, tracker):
    """Process subjects with error handling and logging"""
    
    results = []
    failed_subjects = []
    
    for i, subject_info in enumerate(subject_list, 1):
        try:
            print(f"\n[{i}/{len(subject_list)}] Processing {subject_info['name']}...")
            
            result = process_single_subject(
                subject_info['name'],
                subject_info['xdf_pattern'],
                params_dict,
                verbose=True
            )
            
            if result is not None:
                results.append(result)
                tracker.add_subject(subject_info['name'], result['metrics'])
                print(f"✓ {subject_info['name']} completed successfully")
            else:
                failed_subjects.append(subject_info['name'])
                print(f"✗ {subject_info['name']} returned None")
                
        except Exception as e:
            failed_subjects.append(subject_info['name'])
            print(f"✗ {subject_info['name']} failed with error: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print(f"PROCESSING COMPLETE")
    print(f"Successful: {len(results)}/{len(subject_list)}")
    if failed_subjects:
        print(f"Failed: {', '.join(failed_subjects)}")
    print("="*60)
    
    tracker.compute_group_statistics()
    tracker.print_summary()
    
    return results

# Use it
all_results = robust_group_analysis(subject_list, params_dict, tracker)
```

## Example 9: Statistical Group Analysis

```python
from scipy import stats
import numpy as np

# Get results as DataFrame
df = tracker.get_dataframe()

# Summary statistics
print("=== GROUP STATISTICS ===")
print(f"N subjects: {len(df)}")
print(f"\nAccuracy:")
print(f"  Mean: {df['accuracy'].mean():.4f}")
print(f"  Median: {df['accuracy'].median():.4f}")
print(f"  Std: {df['accuracy'].std():.4f}")
print(f"  SEM: {df['accuracy'].sem():.4f}")

print(f"\nF1 Score:")
print(f"  Mean: {df['f1_score'].mean():.4f}")
print(f"  Median: {df['f1_score'].median():.4f}")
print(f"  Range: [{df['f1_score'].min():.4f}, {df['f1_score'].max():.4f}]")

# Test if accuracy differs from chance level (assuming 2-class, chance = 0.5)
t_stat, p_val = stats.ttest_1samp(df['accuracy'], 0.5)
print(f"\nT-test vs chance (0.5):")
print(f"  t = {t_stat:.4f}, p = {p_val:.4f}")

# Correlation analysis
corr = df['accuracy'].corr(df['f1_score'])
print(f"\nCorrelation (Accuracy vs F1): {corr:.4f}")
```

## Example 10: Extract and Visualize Confusion Matrices

```python
def plot_group_confusion_matrices(all_results):
    """Create a grid of confusion matrices for all subjects"""
    
    n_subjects = len(all_results)
    n_cols = 3
    n_rows = (n_subjects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, result in enumerate(all_results):
        cm = result['metrics']['confusion_matrix']
        subject_name = result['subject_name']
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = axes[idx].imshow(cm_norm, cmap='Blues')
        axes[idx].set_title(f'{subject_name}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = axes[idx].text(j, i, f'{cm_norm[i, j]:.2f}',
                                     ha="center", va="center",
                                     color="white" if cm_norm[i, j] > 0.5 else "black")
        
        plt.colorbar(im, ax=axes[idx])
    
    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Use it
plot_group_confusion_matrices(all_results)
```

## Notes

- Each example is independent and can be run separately
- Customize `params_dict` values as needed for your analysis
- Check file paths match your directory structure
- Results are saved with timestamps for reproducibility
- All metrics are automatically tracked and aggregated
