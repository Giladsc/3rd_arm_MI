#%%
"""
The engine behind every notebook's parameters cell.

Each analysis notebook spells its whole params dict out in one cell, so two things
have to be said somewhere the reader can see: which keys the pipeline actually reads
(a misspelled key is otherwise a silent no-op), and which of the ones it reads are
overwritten downstream regardless of what the cell says. ``describe`` and the
``unknown_keys`` / ``changed_against`` helpers here do that job.

This module deliberately imports NOTHING from the rest of the package.
``analysis_common`` imports from ``tfr_batch``, so anything both of them need has to
live below both of them or the import graph cycles - which is why this is its own
module rather than a section of ``analysis_common``.

A SPEC is three plain data structures, supplied by the caller:

    groups  {group name: (key, ...)}   ordered; how a notebook should treat each key
    docs    {key: one-line description}
    titles  {group name: heading}      '{...}' fields are filled from **fmt

Nothing here knows what a valid value is - the rules differ per pipeline (the
classifier stack requires HighPass > LowPass; the TFR stack runs HighPass=None), so
each module keeps its own ``check_*`` and calls in here only for the parts that are
genuinely common.
"""

import difflib


def format_value(value, key=None, summarize=None, width=56):
    """
    One-line rendering of a params value, short enough to tabulate.

    ``summarize`` maps a key to the unit of its sequence value ('ch', 'Hz'), for the
    long ones worth reporting as a count plus their ends rather than truncating
    mid-item. Everything else is ``repr`` truncated at ``width``, which fits a
    4-class ``desired_events`` uncut - that list names the output tree, and a
    truncated readback of it would be worse than none.
    """
    unit = (summarize or {}).get(key)
    if unit is not None and hasattr(value, '__len__') and len(value) > 4:
        items = [_short(item) for item in value]
        text = f"{len(items)} {unit}: {', '.join(items[:3])} ... {items[-1]}"
    else:
        text = repr(sorted(value) if isinstance(value, (set, frozenset)) else value)
    return text if len(text) <= width else text[:width - 3] + '...'


def _short(item):
    """An item inside a summarised sequence, without quotes around plain strings."""
    return item if isinstance(item, str) else repr(item)


def describe(params, groups, docs, defaults=None, titles=None, known=None,
             summarize=None, width=56, marker='*', **fmt):
    """
    Print every key of ``params`` under its group heading, with values and docs.

    Marks with ``marker`` each key whose value differs from ``defaults``, so a cell
    that departs from the library defaults says so at a glance. Keys of ``params``
    that appear in no group are listed at the end as unread - they are the typos.

    Prints nothing about keys a group names but ``params`` lacks: a cell is free to
    leave a key out and take the library default for it.
    """
    defaults = defaults or {}
    titles = titles or {name: name.upper() for name in groups}
    for group, keys in groups.items():
        present = [key for key in keys if key in params]
        if not present:
            continue
        print(titles[group].format(**fmt))
        for key in present:
            differs = key in defaults and not _same(defaults[key], params[key])
            print(f"  {key:<27}{marker if differs else ' '} "
                  f"{format_value(params[key], key, summarize, width):<{width + 2}}"
                  f"{docs.get(key, '')}")
    known = known if known is not None else {k for keys in groups.values() for k in keys}
    extra = sorted(set(params) - set(known))
    if extra:
        print(f"UNREAD - no part of the pipeline looks at these: {extra}")


def _same(left, right):
    """
    Equality that survives numpy arrays.

    ``==`` on two arrays returns an array, whose truth value raises - and the TFR
    spec holds ``freqs`` as one, so a plain ``!=`` here would break describe().
    """
    try:
        return bool(left == right)
    except ValueError:
        left, right = getattr(left, 'tolist', lambda: left)(), \
                      getattr(right, 'tolist', lambda: right)()
        return left == right


def unknown_keys(params, known):
    """
    Keys of ``params`` that nothing reads, each with a spelling suggestion.

    Returns a list of ready-to-print strings, empty when the dict is clean. The
    suggestion is what makes this worth raising on: 'LowPas' silently doing nothing
    for a whole batch is the failure this prevents.
    """
    hints = []
    for key in sorted(set(params) - set(known)):
        close = difflib.get_close_matches(key, list(known), n=1)
        hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ''))
    return hints


def changed_against(params, recorded, keys):
    """
    Which of ``keys`` differ between ``params`` and what was ``recorded`` on disk.

    Only keys present in both are compared: a recorded run predating a key says
    nothing about it, and reporting it as changed would cry wolf on every old tree.
    """
    return [key for key in keys
            if key in recorded and key in params
            and not _same(recorded[key], params[key])]
