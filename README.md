# mini_cuckoo_filter

A tiny, dependency-free Cuckoo filter for approximate set-membership testing with dynamic deletion.

```python
from mini_cuckoo_filter import CuckooFilter, FilterFull

f = CuckooFilter(capacity=10000)
f.add("user:42")
"user:42" in f          # -> True
f.remove("user:42")     # -> True
"user:42" in f          # -> False (or True, if a false positive occurred)
```

## Why this exists

Bloom filters tell you whether an item is probably in a set, but they cannot
delete entries. Cuckoo filters keep that property while supporting removal,
which matters for caches that expire keys, dedup pipelines that un-queue items
on failure, and any system whose set membership changes over time. The cost is
that a Cuckoo filter can refuse new inserts when load is high, and it admits
false positives on both `__contains__` and (by design) on `remove`.

The specific trade-off here: I picked four slots per bucket and 16-bit
fingerprints because that combination gives ~95% occupancy headroom and a low
false-positive rate without dragging in a C extension. Hashing uses `sha256`
from the standard library, which is slower than a custom `murmur3` would be but
makes the filter reproducible across processes and platforms with no
compile step.

## The awkward edge you will hit

`FilterFull`. Insertions can fail when the filter gets crowded, not when it is
literally full. The caller has to decide what to do — rebuild bigger, drop the
item, or evict something else. `mini_cuckoo_filter` does not auto-grow,
because silent resizing would hide a failure mode the caller likely needs to
know about. Catch `FilterFull` around `add` if losing an insert is not
acceptable.

Removing an item that was never inserted returns `False` silently. It cannot
mean anything stronger, because the filter stores fingerprints rather than
items and so cannot distinguish "absent" from "already deleted by someone
else".

## What it exports

* `CuckooFilter(capacity, *, max_kicks=500, seed=0)` — the filter. `add`,
  `remove`, `contains` (alias for `__contains__`), `__contains__`, `__len__`,
  `capacity`, `load_factor`, and iteration over fingerprints.
* `FilterFull` — raised by `add` when eviction fails.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

