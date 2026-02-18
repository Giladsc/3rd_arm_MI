# Multi-Subject Analysis System - Complete Documentation Index

## 📚 Documentation Files Overview

### 1. **QUICK_REFERENCE.md** ⚡
   - **Best for:** Getting started quickly
   - **Contains:** Copy-paste code, common operations, tips
   - **Read time:** 5 minutes
   - **Key sections:**
     - Quick start (3 lines of code)
     - Common operations
     - Quick troubleshooting

### 2. **CHANGES_SUMMARY.md** 📋
   - **Best for:** Understanding what was added
   - **Contains:** All new cells, features, functions
   - **Read time:** 10 minutes
   - **Key sections:**
     - What was added to the notebook
     - Key features
     - How to use
     - Data structures

### 3. **MULTI_SUBJECT_ANALYSIS_GUIDE.md** 📖
   - **Best for:** Complete reference documentation
   - **Contains:** Detailed explanations, all functions, troubleshooting
   - **Read time:** 20-30 minutes
   - **Key sections:**
     - Complete workflow
     - All available functions
     - Output file formats
     - Customization examples
     - Troubleshooting guide
     - Advanced usage patterns

### 4. **MULTI_SUBJECT_EXAMPLES.md** 💻
   - **Best for:** Learning by example
   - **Contains:** 10 different usage scenarios with full code
   - **Read time:** 15-20 minutes
   - **Includes:**
     - Basic group analysis
     - Single subject processing
     - Subject comparison
     - Custom parameters
     - Statistical analysis
     - Visualization examples

### 5. **ARCHITECTURE.md** 🏗️
   - **Best for:** Understanding system design
   - **Contains:** Diagrams, data flows, function hierarchies
   - **Read time:** 10-15 minutes
   - **Sections:**
     - Component diagrams
     - Data flow visualization
     - Function call hierarchy
     - Processing pipeline details
     - Integration with existing code

## 🎯 Quick Navigation Guide

### "I want to..."

**...get started in 5 minutes**
→ Read QUICK_REFERENCE.md → Copy the Quick Start code

**...understand what was added**
→ Read CHANGES_SUMMARY.md → First 2 sections

**...see code examples**
→ Read MULTI_SUBJECT_EXAMPLES.md → Pick an example

**...find a specific function**
→ Read MULTI_SUBJECT_ANALYSIS_GUIDE.md → Function Reference section

**...troubleshoot an issue**
→ Read MULTI_SUBJECT_ANALYSIS_GUIDE.md → Troubleshooting section

**...understand the architecture**
→ Read ARCHITECTURE.md → All sections

**...customize for my use case**
→ Read MULTI_SUBJECT_EXAMPLES.md → Example 5 or 8

**...export and analyze results**
→ Read MULTI_SUBJECT_EXAMPLES.md → Examples 4 & 9

## 📊 Feature Matrix

| Feature | Where to Find | Example |
|---------|---------------|---------|
| Basic usage | QUICK_REFERENCE | Quick Start section |
| Run multiple subjects | CHANGES_SUMMARY | Step 2 |
| View results | QUICK_REFERENCE | "View Group Summary" |
| Create plots | QUICK_REFERENCE | "Plot Comparisons" |
| Export to CSV | QUICK_REFERENCE | "Export to CSV" |
| Custom parameters | MULTI_SUBJECT_EXAMPLES | Example 5 |
| Statistical analysis | MULTI_SUBJECT_EXAMPLES | Example 9 |
| Error handling | MULTI_SUBJECT_EXAMPLES | Example 8 |
| Reload results | MULTI_SUBJECT_EXAMPLES | Example 6 |
| Advanced visualization | MULTI_SUBJECT_EXAMPLES | Example 10 |

## 🔑 Key Concepts

### Main Classes
- **SubjectPerformanceTracker** - Manages tracking and aggregation
  - See: MULTI_SUBJECT_ANALYSIS_GUIDE.md (Core Processing section)

### Main Functions
- **run_group_analysis()** - Orchestrates multi-subject processing
  - See: QUICK_REFERENCE.md (Quick Start)
- **process_single_subject()** - Processes one subject end-to-end
  - See: MULTI_SUBJECT_EXAMPLES.md (Example 2)
- **calculate_performance_metrics()** - Computes metrics
  - See: MULTI_SUBJECT_ANALYSIS_GUIDE.md (Metrics & Calculation)

### Output Structures
- **Subject Result Dict** - Per-subject data
  - See: CHANGES_SUMMARY.md (Data Structure section)
- **Group Statistics** - Aggregated data
  - See: CHANGES_SUMMARY.md (Data Structure section)

## 🚀 Recommended Reading Order

**For First-Time Users:**
1. QUICK_REFERENCE.md (5 min) - Get overview
2. CHANGES_SUMMARY.md (10 min) - Understand what's new
3. MULTI_SUBJECT_EXAMPLES.md Example 1 (5 min) - Run first example
4. QUICK_REFERENCE.md again - Look up specific operations

**For Advanced Users:**
1. CHANGES_SUMMARY.md (10 min) - What's new
2. ARCHITECTURE.md (15 min) - System design
3. MULTI_SUBJECT_EXAMPLES.md Examples 5,8,9 (15 min) - Advanced patterns
4. MULTI_SUBJECT_ANALYSIS_GUIDE.md (20 min) - Reference as needed

**For Troubleshooting:**
1. QUICK_REFERENCE.md - Common issues table
2. MULTI_SUBJECT_ANALYSIS_GUIDE.md - Troubleshooting section
3. MULTI_SUBJECT_EXAMPLES.md Example 8 - Error handling

## 📝 File Locations

All documentation files are in your project root:
```
c:\Users\CensorLab\3rd_arm_MI\
├── QUICK_REFERENCE.md
├── CHANGES_SUMMARY.md
├── MULTI_SUBJECT_ANALYSIS_GUIDE.md
├── MULTI_SUBJECT_EXAMPLES.md
├── ARCHITECTURE.md
└── README_MULTISUBJECT.md  (this file)
```

The modified notebook is:
```
c:\Users\CensorLab\3rd_arm_MI\Main.ipynb (cells added after line 1681)
```

## 🔗 Cross-References

### To find information about...

**SubjectPerformanceTracker class:**
- Overview: CHANGES_SUMMARY.md
- Methods: MULTI_SUBJECT_ANALYSIS_GUIDE.md
- Examples: MULTI_SUBJECT_EXAMPLES.md (all examples)

**process_single_subject() function:**
- Details: CHANGES_SUMMARY.md
- Examples: MULTI_SUBJECT_EXAMPLES.md Example 2
- Troubleshooting: MULTI_SUBJECT_ANALYSIS_GUIDE.md

**run_group_analysis() function:**
- Details: CHANGES_SUMMARY.md
- Examples: MULTI_SUBJECT_EXAMPLES.md Example 1
- Advanced: MULTI_SUBJECT_EXAMPLES.md Example 8

**Visualization functions:**
- Details: MULTI_SUBJECT_ANALYSIS_GUIDE.md
- Examples: MULTI_SUBJECT_EXAMPLES.md Examples 7, 10

**Export/Save functions:**
- Details: MULTI_SUBJECT_ANALYSIS_GUIDE.md
- Examples: MULTI_SUBJECT_EXAMPLES.md Examples 4, 6

**Data structures:**
- Details: CHANGES_SUMMARY.md (Data Structure)
- Visual: ARCHITECTURE.md (Class Relationships)

**Integration with existing code:**
- Overview: CHANGES_SUMMARY.md (Integration)
- Diagram: ARCHITECTURE.md (Integration Points)

## ✅ Verification Checklist

After setup, verify:
- [ ] Can import all new functions (no error messages)
- [ ] Can run `run_group_analysis()` with a subject
- [ ] Can create `SubjectPerformanceTracker()`
- [ ] Can visualize with `plot_group_performance_comparison()`
- [ ] Can export CSV with `export_group_results_to_csv()`
- [ ] Can load results with `load_group_results()`

## 🎓 Learning Path

**Complete Beginner** (30 minutes total):
1. Read QUICK_REFERENCE.md (5 min)
2. Run MULTI_SUBJECT_EXAMPLES.md Example 1 (10 min)
3. Read CHANGES_SUMMARY.md (10 min)
4. Skim MULTI_SUBJECT_ANALYSIS_GUIDE.md (5 min)

**Experienced User** (20 minutes total):
1. Read CHANGES_SUMMARY.md (10 min)
2. Scan MULTI_SUBJECT_EXAMPLES.md Examples 5,8 (10 min)

**Developer** (45 minutes total):
1. Read ARCHITECTURE.md (15 min)
2. Read CHANGES_SUMMARY.md (10 min)
3. Study MULTI_SUBJECT_ANALYSIS_GUIDE.md (15 min)
4. Review MULTI_SUBJECT_EXAMPLES.md (5 min)

## 📞 Support Resources

| Question | Resource |
|----------|----------|
| "How do I start?" | QUICK_REFERENCE.md Quick Start |
| "What functions are available?" | MULTI_SUBJECT_ANALYSIS_GUIDE.md Function Reference |
| "Show me examples" | MULTI_SUBJECT_EXAMPLES.md |
| "How does it work?" | ARCHITECTURE.md |
| "Something isn't working" | MULTI_SUBJECT_ANALYSIS_GUIDE.md Troubleshooting |
| "I need help customizing" | MULTI_SUBJECT_EXAMPLES.md Example 5 |

## 🔄 Version Information

- **Version:** 1.0
- **Date:** February 2026
- **Status:** Production Ready
- **Python Version:** 3.7+
- **Dependencies:** MNE, scikit-learn, pandas, numpy, matplotlib

## 📈 What You Can Do Now

✅ Process multiple subjects automatically  
✅ Track performance across subjects  
✅ Aggregate group statistics  
✅ Compare subjects side-by-side  
✅ Export results to CSV/JSON  
✅ Visualize group performance  
✅ Create individual reports  
✅ Reload and reanalyze results  

---

**Start here:** Open QUICK_REFERENCE.md and copy the Quick Start code!

**Questions?** Check the relevant documentation file above.

**Found a bug?** Review MULTI_SUBJECT_ANALYSIS_GUIDE.md Troubleshooting section.
