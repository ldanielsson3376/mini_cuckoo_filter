"""Unit tests for mini_cuckoo_filter.core."""

import math
import unittest

from mini_cuckoo_filter import CuckooFilter, FilterFull


class TestConstruction(unittest.TestCase):
    def test_default_construction(self):
        f = CuckooFilter(1000)
        self.assertEqual(len(f), 0)
        self.assertEqual(f.capacity, 1024)

    def test_capacity_is_power_of_two_buckets(self):
        # capacity 1000 -> 250 buckets -> rounds up to 256 -> capacity 1024.
        f = CuckooFilter(1000)
        self.assertEqual(f.capacity, 1024)

        # capacity 5 -> 2 buckets -> capacity 8.
        f2 = CuckooFilter(5)
        self.assertEqual(f2.capacity, 8)

    def test_capacity_below_one_raises(self):
        with self.assertRaises(ValueError):
            CuckooFilter(0)
        with self.assertRaises(ValueError):
            CuckooFilter(-10)

    def test_seed_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            CuckooFilter(10, seed=-1)
        with self.assertRaises(ValueError):
            CuckooFilter(10, seed=0x100000000)

    def test_max_kicks_below_one_raises(self):
        with self.assertRaises(ValueError):
            CuckooFilter(10, max_kicks=0)


class TestMembership(unittest.TestCase):
    def test_inserted_item_is_present(self):
        f = CuckooFilter(1000)
        f.add("alpha")
        self.assertIn("alpha", f)
        self.assertTrue(f.contains("alpha"))

    def test_missing_item_is_absent(self):
        f = CuckooFilter(1000)
        self.assertNotIn("beta", f)
        self.assertFalse(f.contains("beta"))

    def test_bytes_and_bytearray_items_supported(self):
        f = CuckooFilter(1000)
        f.add(b"hello")
        self.assertIn(b"hello", f)
        f.add(bytearray(b"world"))
        self.assertIn(bytearray(b"world"), f)

    def test_int_and_string_do_not_collide_by_type(self):
        # 1 and "1" must be treated as different items. We check that the
        # filter derives distinct fingerprints/indices for them; a membership
        # false positive is inherent to the data structure and cannot be
        # asserted away.
        f = CuckooFilter(1000)
        i1_int, i2_int, fp_int = f._indices_and_fingerprint(1)
        i1_str, i2_str, fp_str = f._indices_and_fingerprint("1")
        self.assertTrue(
            fp_int != fp_str or {i1_int, i2_int} != {i1_str, i2_str},
            "1 and '1' must hash to distinct fingerprints or indices",
        )
        f.add(1)
        self.assertIn(1, f)

    def test_seed_isolates_filters(self):
        # Same item, different seed -> different fingerprints/indices, so a
        # membership check across two independent filters should not both
        # return True for an item added to only one of them. We can't assert
        # the second is False (it could be a false positive) but we CAN
        # assert that at least one of them reports False.
        f1 = CuckooFilter(500, seed=1)
        f2 = CuckooFilter(500, seed=2)
        f1.add("item")
        self.assertIn("item", f1)
        self.assertFalse(f2.contains("item"))


class TestRemoval(unittest.TestCase):
    def test_remove_present_item(self):
        f = CuckooFilter(1000)
        f.add("x")
        self.assertTrue(f.remove("x"))
        self.assertEqual(len(f), 0)

    def test_remove_missing_item_is_false(self):
        f = CuckooFilter(1000)
        self.assertFalse(f.remove("never-added"))

    def test_remove_twice_second_returns_false(self):
        f = CuckooFilter(1000)
        f.add("x")
        self.assertTrue(f.remove("x"))
        self.assertFalse(f.remove("x"))

    def test_remove_only_removes_one_occurrence(self):
        f = CuckooFilter(1000)
        # Adding the same item twice is permitted; fingerprints can duplicate.
        f.add("dup")
        f.add("dup")
        self.assertEqual(len(f), 2)
        self.assertTrue(f.remove("dup"))
        self.assertEqual(len(f), 1)
        self.assertIn("dup", f)
        self.assertTrue(f.remove("dup"))
        self.assertEqual(len(f), 0)
        self.assertNotIn("dup", f)


class TestCapacity(unittest.TestCase):
    def test_full_filter_raises_filter_full(self):
        # Tiny capacity to force eviction pressure quickly.
        f = CuckooFilter(4, max_kicks=8)
        # Fill well beyond what 4 slots can hold. The filter should refuse at
        # some point with FilterFull rather than silently dropping entries.
        added = 0
        with self.assertRaises(FilterFull):
            for i in range(1000):
                f.add("item-%d" % i)
                added += 1
        self.assertLess(added, 1000)

    def test_items_survive_below_capacity(self):
        f = CuckooFilter(1000)
        items = ["k-%d" % i for i in range(200)]
        for x in items:
            f.add(x)
        # With load ~0.2 there should be no false negatives.
        for x in items:
            self.assertIn(x, f)

    def test_no_silent_false_negatives_under_load(self):
        # Fill to a moderate load and confirm every inserted item is still
        # reported present (no false negatives).
        f = CuckooFilter(1024)
        items = ["item-%04d" % i for i in range(512)]
        for x in items:
            f.add(x)
        for x in items:
            self.assertIn(x, f, "false negative for %s" % x)


class TestStatistics(unittest.TestCase):
    def test_len_tracks_insertions_and_deletions(self):
        f = CuckooFilter(1000)
        for i in range(50):
            f.add(i)
        self.assertEqual(len(f), 50)
        for i in range(20):
            f.remove(i)
        self.assertEqual(len(f), 30)

    def test_load_factor_increases_with_insertions(self):
        f = CuckooFilter(1024)
        self.assertEqual(f.load_factor, 0.0)
        for i in range(256):
            f.add(i)
        # 256 fingerprints across 1024 slots -> load 0.25 exactly.
        self.assertAlmostEqual(f.load_factor, 0.25)

    def test_iteration_yields_fingerprints_only(self):
        f = CuckooFilter(1000)
        f.add("a")
        f.add("b")
        fps = list(f)
        self.assertEqual(len(fps), 2)
        self.assertTrue(all(isinstance(x, int) for x in fps))
        self.assertTrue(all(x != 0 for x in fps))  # 0 means empty slot


if __name__ == "__main__":
    unittest.main()
