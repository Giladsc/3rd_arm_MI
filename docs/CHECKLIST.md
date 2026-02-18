# ✅ Setup Verification & Getting Started Checklist

## Pre-Flight Checklist ✈️

Before running your first analysis, verify everything is in place:

### Notebook Changes
- [ ] Opened Main.ipynb
- [ ] Located cells after line 1681 (new cells are here)
- [ ] Can see "Multi-Subject Analysis" section header
- [ ] Cells are not showing syntax errors (no red underlining)

### Documentation Files
- [ ] START_HERE.md exists in project directory
- [ ] QUICK_REFERENCE.md exists in project directory  
- [ ] MULTI_SUBJECT_ANALYSIS_GUIDE.md exists in project directory
- [ ] MULTI_SUBJECT_EXAMPLES.md exists in project directory
- [ ] README_MULTISUBJECT.md exists in project directory
- [ ] ARCHITECTURE.md exists in project directory

### Python Environment
- [ ] Can import main notebook without errors
- [ ] All existing functions still work
- [ ] Can see existing variables in kernel

### Data Files
- [ ] XDF files exist in Recordings/ folder
- [ ] Subject naming is consistent
- [ ] params_dict is available in kernel
- [ ] Previous analyses have been run successfully

---

## First-Time Setup (30 minutes)

### Step 1: Read Documentation (5 min)
- [ ] Open **START_HERE.md** (Visual summary)
- [ ] Open **QUICK_REFERENCE.md** (Quick start guide)
- [ ] Read the "Quick Start" section only
- ✅ Now you know what to do

### Step 2: Prepare Your Subject List (5 min)
- [ ] List your subjects: _________________
- [ ] Note their XDF file patterns: _________________
- [ ] Example: {'name': 'Tomer', 'xdf_pattern': 'Tomer'}
- [ ] Create your subject_list variable

### Step 3: Run Quick Test (10 min)
- [ ] Copy Quick Start code from QUICK_REFERENCE.md
- [ ] Paste into new notebook cell
- [ ] Update subject list with 1-2 test subjects
- [ ] Run the cell
- [ ] Monitor output (should show ✓ messages)
- ✅ First subject processed successfully

### Step 4: View Results (5 min)
- [ ] Run: `tracker.print_summary()`
- [ ] See group statistics printed
- [ ] Run: `plot_group_performance_comparison(tracker)`
- [ ] See beautiful comparison charts
- ✅ Results are displayed

### Step 5: Export Results (5 min)
- [ ] Run: `export_group_results_to_csv(tracker)`
- [ ] Check project directory for CSV file
- [ ] Open in Excel (optional)
- ✅ Results exported successfully

---

## Running Your Analysis

### Before You Start
- [ ] All subjects have XDF files in Recordings/
- [ ] XDF file names contain the pattern you specified
- [ ] params_dict is available and configured
- [ ] You have 30-60 minutes per 2-3 subjects
- [ ] Enough disk space (~500 MB per subject)

### Starting Analysis
```python
# This is what you'll run:

# 1. Define subjects
subject_list = [
    {'name': 'SubjectA', 'xdf_pattern': 'SubjectA'},
    {'name': 'SubjectB', 'xdf_pattern': 'SubjectB'},
]

# 2. Initialize tracker
tracker = SubjectPerformanceTracker()

# 3. Run analysis (takes 20-60 min)
results = run_group_analysis(subject_list, params_dict, tracker)

# 4. View results (instant)
tracker.print_summary()

# 5. Visualize (instant)
plot_group_performance_comparison(tracker)

# 6. Export (instant)
export_group_results_to_csv(tracker)
```

### During Processing
- [ ] Watch for ✓ checkmarks (= completed)
- [ ] Watch for progress numbers [Subject 1/3], [Subject 2/3]
- [ ] If error occurs, check troubleshooting section
- [ ] Processing can take 10-30 minutes per subject
- ✅ Wait for "PROCESSING COMPLETE" message

### After Processing
- [ ] Check console for group statistics
- [ ] Look at bar charts (accuracy, F1, etc.)
- [ ] Review CSV file in project directory
- [ ] Check group_results.json for detailed data
- ✅ Success! Your analysis is complete

---

## File Location Reference

### Notebook Location
```
c:\Users\CensorLab\3rd_arm_MI\Main.ipynb
│
└─ New cells added after line 1681
   (Can scroll down to find them)
```

### Documentation Location
```
c:\Users\CensorLab\3rd_arm_MI\
├─ START_HERE.md                          ← Visual overview
├─ QUICK_REFERENCE.md                     ← Cheat sheet
├─ CHANGES_SUMMARY.md                     ← What changed
├─ MULTI_SUBJECT_ANALYSIS_GUIDE.md        ← Full reference
├─ MULTI_SUBJECT_EXAMPLES.md              ← Code examples
├─ README_MULTISUBJECT.md                 ← Navigation
├─ ARCHITECTURE.md                        ← System design
└─ IMPLEMENTATION_COMPLETE.md             ← Summary
```

### Output Location (After Running)
```
c:\Users\CensorLab\3rd_arm_MI\
├─ group_results.json                     ← Complete results
├─ group_performance_summary.csv          ← Summary table
└─ [optional subject reports]             ← Individual summaries
```

---

## Troubleshooting Guide

### Issue: "No XDF files found"

**Cause:** XDF file pattern doesn't match filenames

**Check:**
- [ ] XDF files exist in Recordings/ folder
- [ ] File names contain your pattern (e.g., 'Tomer')
- [ ] Case matches (case-sensitive!)
- [ ] No extra spaces in pattern

**Fix:**
```python
# Show actual file names
import pathlib
recording_path = pathlib.Path('Recordings')
files = list(recording_path.glob('*.xdf'))
print([f.name for f in files])

# Update pattern to match
```

### Issue: "Processing seems stuck"

**Cause:** Normal - processing takes 10-30 min per subject

**Check:**
- [ ] Watch console for processing messages
- [ ] Subject processing shows [1/3], [2/3], etc.
- [ ] No error messages appearing
- [ ] Still seeing output (not frozen)

**Solution:** Wait! It's normal for large files to take time.

### Issue: Memory error during processing

**Cause:** Not enough RAM for large datasets

**Solution:**
- [ ] Close other applications
- [ ] Reduce augmentation params:
  ```python
  params['augmentation_params'] = {'win_len': 0, 'win_step': 0.5}
  ```
- [ ] Process fewer subjects at once

### Issue: Results not saved

**Cause:** save_results parameter not set

**Fix:**
```python
# Make sure save_results=True
results = run_group_analysis(
    subject_list, 
    params_dict, 
    tracker,
    save_results=True  # ← Add this
)
```

### Issue: Can't find functions

**Cause:** Cells not executed or still loading

**Check:**
- [ ] All new cells have been executed
- [ ] No errors in cell output
- [ ] Kernel is still active
- [ ] No kernel restart needed

**Fix:**
```python
# Test if functions exist
print(SubjectPerformanceTracker)
print(run_group_analysis)
```

---

## Common Questions

### Q: How long does analysis take?
**A:** 10-30 minutes per subject, depending on data size and computer speed.

### Q: Where are the results?
**A:** Two files appear in your project directory:
- `group_results.json` - Complete data
- `group_performance_summary.csv` - Summary table

### Q: Can I process one subject at a time?
**A:** Yes! Use `process_single_subject()` function

### Q: Can I run subjects in parallel?
**A:** Currently sequential. See MULTI_SUBJECT_EXAMPLES.md Example 8 for advanced patterns.

### Q: How do I reload previous results?
**A:** Use `load_group_results('group_results.json')`

### Q: Can I customize parameters per subject?
**A:** Yes! See MULTI_SUBJECT_EXAMPLES.md Example 5

### Q: What if my subjects have different parameters?
**A:** See MULTI_SUBJECT_EXAMPLES.md Example 5 for custom parameter handling

---

## Performance Expectations

### Accuracy per Subject
- Typically ranges from 0.60 to 0.95
- Depends on: number of epochs, SNR, subject variability
- Group mean typically 0.75-0.85

### Processing Speed
- Loading: ~1 minute per subject
- Preprocessing: ~5-10 minutes per subject
- Training: ~2-5 minutes per subject
- Prediction: ~1 minute per subject
- Total: ~10-30 minutes per subject

### File Sizes
- per subject results: ~100-500 MB
- Saved JSON: ~50-100 MB
- CSV export: ~10-50 KB

---

## Success Verification

### After First Run, You Should See:

✅ Console Output:
```
==================================================
Subject 1/2 Processing SubjectA...
  ✓ Loaded 342 epochs
  ✓ Train set: 273 samples
  ✓ Validation set: 69 samples
  ✓ Augmented to 819 samples
  ✓ Classifier trained
  ✓ Accuracy: 0.8456
  ✓ F1 Score: 0.8234
...
✓ SubjectA processing complete!
==================================================
```

✅ Summary Output:
```
============================================================
GROUP PERFORMANCE SUMMARY
============================================================
Number of subjects: 2
Total epochs: 611

--- ACCURACY ---
  Mean: 0.8145
  Std:  0.0312
  Range: [0.7833, 0.8456]
...
```

✅ Files Created:
- `group_results.json` ✅
- `group_performance_summary.csv` ✅

✅ Visualizations:
- 4-panel bar chart appears ✅

---

## Next Steps After First Run

### Immediate (Same Day)
- [ ] Review the summary statistics
- [ ] Look at the visualization
- [ ] Export CSV and open in Excel
- [ ] Share results with colleagues

### Short-term (Next Days)
- [ ] Read MULTI_SUBJECT_EXAMPLES.md for advanced usage
- [ ] Try individual subject comparison
- [ ] Create custom reports
- [ ] Export to new format (if needed)

### Long-term (Next Weeks)
- [ ] Process all subjects
- [ ] Run statistical analysis
- [ ] Create publication-ready figures
- [ ] Archive results with timestamps

---

## Documentation Reading Order

### For Immediate Use (15 min)
1. ✅ START_HERE.md (this gives overview)
2. ✅ QUICK_REFERENCE.md (copy Quick Start)
3. ✅ Run your first analysis!

### For Understanding (30 min)
1. ✅ CHANGES_SUMMARY.md (what was added)
2. ✅ QUICK_REFERENCE.md (operations)
3. ✅ MULTI_SUBJECT_EXAMPLES.md (see examples)

### For Mastery (1 hour)
1. ✅ MULTI_SUBJECT_ANALYSIS_GUIDE.md (complete ref)
2. ✅ ARCHITECTURE.md (system design)
3. ✅ MULTI_SUBJECT_EXAMPLES.md (advanced examples)
4. ✅ QUICK_REFERENCE.md (as needed lookup)

### For Deep Dive (2 hours)
1. ✅ Read all documentation files
2. ✅ Study source code in cells
3. ✅ Try all examples
4. ✅ Create custom modifications

---

## You're Ready! 🚀

This checklist confirms:
- ✅ System is installed
- ✅ Documentation is available
- ✅ You understand the workflow
- ✅ You know where to find help
- ✅ You can run your first analysis

### Start Now:
1. Open **START_HERE.md** 
2. Follow "Quick Start" code
3. Run in your notebook
4. View results

### Questions? 
→ Check **README_MULTISUBJECT.md** for which doc has your answer

---

**Date Completed:** _______________  
**By:** _______________  
**First Analysis Run:** _______________  
**Notes:** _______________
