ann1 = stream_annotations
ann2 = original_ann

# Shift all onsets in ann1 back by 2 seconds
ann1_shifted = mne.Annotations(
    onset=ann1.onset - 2.0,       # subtract 2s from each onset
    duration=ann1.duration.copy(),  # keep durations the same
    description=ann1.description.copy(),  # keep labels the same
    orig_time=ann1.orig_time      # preserve original time reference
)


def expand_triggers_to_events(ann, raw=None, recording_end=None, use_raw_timebase=True):
    """
    Turn trigger annotations (point markers) into event intervals:
    each trigger's label lasts until the next trigger onset.

    Parameters
    ----------
    ann : mne.Annotations
        Trigger-like annotations (often duration=0).
    raw : mne.io.BaseRaw or None
        If given, we'll use raw.times[-1] as the end of recording and (optionally)
        align orig_time to raw.info['meas_date'].
    recording_end : float or None
        End time in seconds (from recording start). If None, inferred from `raw`.
    use_raw_timebase : bool
        If True and `raw` is provided, set orig_time to raw.info['meas_date'] so
        plotting aligns perfectly with `raw`.

    Returns
    -------
    events_ann : mne.Annotations
        Annotations whose durations now extend to the next trigger (last to end).
    """
    if recording_end is None:
        if raw is None:
            raise ValueError("Provide either `raw` or `recording_end` (seconds).")
        recording_end = float(raw.times[-1])

    on = np.asarray(ann.onset, float)
    desc = np.asarray(ann.description, dtype=object)

    # Sort by onset to build contiguous intervals
    order = np.argsort(on, kind="mergesort")
    on = on[order]
    desc = desc[order]

    if len(on) == 0:
        # Return an empty annotations aligned to raw (if provided) or keep original timebase
        base_time = raw.info['meas_date'] if (use_raw_timebase and raw is not None) else ann.orig_time
        return mne.Annotations([], [], [], orig_time=base_time)

    # Each trigger holds until the next trigger; last goes to recording_end
    next_on = np.r_[on[1:], recording_end]
    dur = next_on - on

    # Drop any non-positive durations (can happen with duplicate/same-time triggers)
    keep = dur > 0
    on, dur, desc = on[keep], dur[keep], desc[keep]

    base_time = raw.info['meas_date'] if (use_raw_timebase and raw is not None) else ann.orig_time
    return mne.Annotations(on, dur, desc.tolist(), orig_time=base_time)

event_ann = expand_triggers_to_events(ann1_shifted,Raw_for_analysis)


def _to_intervals(ann, include_desc=None):
    """Convert event-style annotations to (start, end, desc) tuples."""
    on = np.asarray(ann.onset, float)
    du = np.asarray(ann.duration, float)
    desc = np.asarray(ann.description, dtype=object)
    if include_desc is not None:
        mask = np.isin(desc, list(include_desc))
        on, du, desc = on[mask], du[mask], desc[mask]
    return [(float(s), float(s + d), str(dsc)) for s, d, dsc in zip(on, du, desc)]

def _intersections_by_desc(iv1, iv2):
    """Return {desc: [(start, end), ...]} where intervals overlap AND labels match."""
    by1, by2 = {}, {}
    for s, e, d in iv1: by1.setdefault(d, []).append((s, e))
    for s, e, d in iv2: by2.setdefault(d, []).append((s, e))
    common = set(by1).intersection(by2)
    out = {d: [] for d in common}
    for d in common:
        a = sorted(by1[d]); b = sorted(by2[d])
        i = j = 0
        while i < len(a) and j < len(b):
            s1, e1 = a[i]; s2, e2 = b[j]
            sI, eI = max(s1, s2), min(e1, e2)
            if eI > sI:
                out[d].append((sI, eI))
            if e1 <= e2: i += 1
            else:        j += 1
    return out

def compute_matching_annotations_events(raw, ann1_events, ann2_events, include_desc=None, min_len=0.0):
    """
    Build Annotations with description='match:<desc>' where ann1 & ann2
    have the SAME label and their event intervals overlap.
    """
    iv1 = _to_intervals(ann1_events, include_desc)
    iv2 = _to_intervals(ann2_events, include_desc)
    overlaps = _intersections_by_desc(iv1, iv2)

    on, du, ds = [], [], []
    for d, segs in overlaps.items():
        for s, e in segs:
            if (e - s) >= float(min_len):
                on.append(s); du.append(e - s); ds.append(f"match:{d}")

    if not on:
        return mne.Annotations([], [], [], orig_time=raw.info['meas_date'])
    return mne.Annotations(on, du, ds, orig_time=raw.info['meas_date'])


# Choose labels to compare (or set to None to include all)
include = {"ActiveRest", "ClosePalm"}  # or None

matches = compute_matching_annotations_events(
    Raw_for_analysis,
    event_ann,
    ann2,
    include_desc=include,
    min_len=0.02  # ignore overlaps < 20 ms if you want
)

# Option A: plot in MNE browser (auto-shaded)
raw_match = Raw_for_analysis.copy()
raw_match.set_annotations(matches)
raw_match.plot()

# Option B: if you want guaranteed green shading, reuse your custom overlay:
# fig = plot_raw_with_matches(raw, matches, picks="eeg", tmin=10, tmax=40)
