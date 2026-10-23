"""Cuckoo filter data structure.

A cuckoo filter is a probabilistic data structure for approximate set
membership testing. Like a Bloom filter, it can answer \"is this item in the
set?\" with no false negatives but possible false positives. Unlike a Bloom
filter, it supports deletion and can dynamically grow/shrink.

This implementation uses four buckets per cell, 16-bit fingerprints,
and a fingerprint-based partial-key cuckoo hashing scheme
(fan, Fan et al. 2014). The key insight is that a fingerprint f and its
candidate index i satisfy i ^ hash(f) == j, so given one index and the
fingerprint you can derive the other index cheaply.

Interpretation choices we picked (stated plainly so the tests are not
ambiguous):
  * Item hashing uses sha256 (stdlib) truncated to needed width. Stable and
    dependency-free; not the fastest but deterministic across processes.
  * Bucket size (b) is fixed at 4 — the paper's sweet spot.
  * Fingerprint width is 16 bits. Larger fingerprints reduce false-positive
    rate at the cost of occupancy headroom.
  * `add` raises FilterFull after max_kicks eviction attempts (max_kicks=500).
    We do not auto-grow; the caller decides whether to rebuild bigger.
  * `__contains__` returns True/False (the API users expect); `contains` is an
    alias for it. Both never raise for missing items.
  * Deleting an item not present is a no-op (returns False) rather than an
    error, since the filter can't distinguish \"absent\" from \"false negative\".
"""

from __future__ import annotations

import hashlib
import struct
from typing import Iterator, Optional


class FilterFull(Exception):
    """Raised when insertion exhausts eviction attempts without success.

    This means the load factor is too high for the current capacity; rebuild
    with a larger filter and reinsert all items.
    """


class CuckooFilter:
    """A cuckoo filter supporting approximate membership tests and deletion.

    Parameters
    ----------
    capacity:
        Expected number of items. The actual number of buckets is the next
        power of two >= ceil(capacity / b) where b is the bucket size (4).
        Rounding to a power of two lets us use a bitmask instead of modulo
        for index computation, which is both faster and avoids the biased
        modulo-mapping the original paper warns about.
    max_kicks:
        Eviction attempts before declaring the filter full.
    seed:
        Optional 32-bit integer mixed into every hash. Lets callers derive
        independent filters over the same key space.
    """

    __slots__ = ("_b", "_fp_bits", "_mask", "_buckets", "_max_kicks", "_seed", "_count")

    def __init__(
        self,
        capacity: int,
        *,
        max_kicks: int = 500,
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if max_kicks < 1:
            raise ValueError("max_kicks must be >= 1")
        if not (0 <= seed <= 0xFFFFFFFF):
            raise ValueError("seed must fit in 32 bits")

        self._b = 4
        self._fp_bits = 16

        # Number of buckets: at least capacity / b, rounded up to a power of
        # two so we can mask instead of mod. One extra bucket would waste
        # space; we deliberately stay strict to the paper's formula.
        num_buckets = max(1, (capacity + self._b - 1) // self._b)
        n = 1
        while n < num_buckets:
            n <<= 1
        self._mask = n - 1
        self._buckets: list[list[int]] = [[] for _ in range(n)]

        self._max_kicks = max_kicks
        self._seed = seed
        self._count = 0

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------
    def _digest(self, item: object) -> bytes:
        """Return sha256(item || seed) as raw bytes.

        We hash a canonical byte encoding of the item so that 1 and '1' do
        not collide. bytes/bytearray/memoryview are hashed directly; str is
        utf-8 encoded; everything else goes through repr() then utf-8, prefixed
        with type info so distinct types with the same repr stay distinct. This
        is deterministic and obvious at the cost of some performance.
        """
        if isinstance(item, str):
            data = item.encode("utf-8")
        elif isinstance(item, (bytes, bytearray, memoryview)):
            data = bytes(item)
        else:
            data = type(item).__name__.encode("utf-8") + b":" + repr(item).encode("utf-8")
        return hashlib.sha256(data + struct.pack(">I", self._seed & 0xFFFFFFFF)).digest()

    def _indices_and_fingerprint(self, item: object) -> tuple[int, int, int]:
        """Compute the two candidate bucket indices and the fingerprint."""
        digest = self._digest(item)
        # 8 bytes each: enough for 2^63 buckets (far beyond practical use).
        i1 = int.from_bytes(digest[0:8], "big") & self._mask
        # fingerprint stored as unsigned 16-bit; 0 is reserved to mean "empty",
        # so if the hash happens to be all zeros we bump it to 1.
        fp = int.from_bytes(digest[8:10], "big") & ((1 << self._fp_bits) - 1)
        if fp == 0:
            fp = 1
        # alt_index mixes fingerprint bits into the primary index to derive
        # a second, independent candidate bucket. xor is its own inverse,
        # which is exactly the property we need: alt(alt(i, f), f) == i.
        i2 = self._alt_index(i1, fp)
        return i1, i2, fp

    def _alt_index(self, index: int, fp: int) -> int:
        """Return the alternate bucket index for a given (index, fingerprint)."""
        # Mix fingerprint across the whole index width so small fingerprint
        # differences spread widely even for a 2-bit bucket count.
        h = fp
        h ^= fp << 7
        h ^= (h >> 5) ^ (fp << 13)
        h ^= h >> 17
        h ^= fp << 23
        h ^= h >> 29
        return (index ^ h) & self._mask

    # ------------------------------------------------------------------
    # Bucket-level helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _insert_into_bucket(bucket: list[int], fp: int) -> bool:
        """Try to append fp to bucket. Return False if bucket is full."""
        if len(bucket) < 4:
            bucket.append(fp)
            return True
        return False

    def _try_remove_from_bucket(self, bucket: list[int], fp: int) -> bool:
        """Remove one occurrence of fp from the bucket. Return whether removed."""
        try:
            bucket.remove(fp)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, item: object) -> None:
        """Insert item; raise FilterFull if the filter is out of space."""
        i1, i2, fp = self._indices_and_fingerprint(item)
        b1 = self._buckets[i1]
        if self._insert_into_bucket(b1, fp):
            self._count += 1
            return
        b2 = self._buckets[i2]
        if self._insert_into_bucket(b2, fp):
            self._count += 1
            return

        # Both candidate buckets full — relocate existing fingerprints.
        # Always kick from i1 so behaviour is deterministic given a fixed
        # seed; randomness would make the filter's failure mode hard to test.
        index = i1
        bucket = b1
        current_fp = fp
        for _ in range(self._max_kicks):
            # Evict the fingerprint at a stable-but-varying slot. We use
            # len(bucket) (which is b=4 at this point) as a cheap rotation
            # so we don't need a PRNG and the eviction path is deterministic.
            evict_pos = len(bucket) - 1
            current_fp, bucket[evict_pos] = bucket[evict_pos], current_fp
            index = self._alt_index(index, current_fp)
            bucket = self._buckets[index]
            if self._insert_into_bucket(bucket, current_fp):
                self._count += 1
                return

        raise FilterFull(
            "Unable to insert item after %d kicks; capacity is exhausted. "
            "Rebuild with a larger capacity." % self._max_kicks
        )

    def __contains__(self, item: object) -> bool:
        """Return True if item MAY be in the set (false positives possible)."""
        i1, i2, fp = self._indices_and_fingerprint(item)
        return fp in self._buckets[i1] or fp in self._buckets[i2]

    def contains(self, item: object) -> bool:
        """Alias for ``item in filter``."""
        return item in self

    def remove(self, item: object) -> bool:
        """Remove one occurrence of item's fingerprint.

        Returns True if a fingerprint was removed, False otherwise. A False
        result is not a guarantee the item was never inserted: because the
        filter stores fingerprints, not items, it cannot distinguish an item
        that was never added from one whose fingerprint was already deleted.
        Deleting an item not present is a safe no-op.
        """
        i1, i2, fp = self._indices_and_fingerprint(item)
        if self._try_remove_from_bucket(self._buckets[i1], fp):
            self._count -= 1
            return True
        if self._try_remove_from_bucket(self._buckets[i2], fp):
            self._count -= 1
            return True
        return False

    def __len__(self) -> int:
        """Number of successful add() calls minus successful remove() calls."""
        return self._count

    @property
    def capacity(self) -> int:
        """Configured item capacity (not the bucket count)."""
        return (len(self._buckets) * self._b) or 1

    @property
    def load_factor(self) -> float:
        """Fraction of bucket slots currently occupied."""
        slots = len(self._buckets) * self._b
        if slots == 0:
            return 0.0
        total = sum(len(b) for b in self._buckets)
        return total / slots

    def __iter__(self) -> Iterator[int]:
        """Yield fingerprints present in the filter, slot by slot.

        The filter cannot reconstruct original items; it only stores
        fingerprints. This iterator exists for debugging and serialisation,
        not for reconstructing the original set.
        """
        for bucket in self._buckets:
            for fp in bucket:
                yield fp
