/* Hash tables for OSM ids: a set of int64 and an int64 -> int64 map, built on
   klib's khash.h (vendored next to this file). */
#ifndef PYROSM_INT64_HASH_H
#define PYROSM_INT64_HASH_H

#include <stdint.h>
#include "khash.h"

/* 64-bit MurmurHash2 of one key, folded to 32 bits. */
static kh_inline khint_t pyrosm_hash_int64(int64_t key)
{
    const uint64_t m = 0xc6a4a7935bd1e995ULL;
    const int r = 47;
    uint64_t k = (uint64_t)key;
    uint64_t h = 0xc70f6907b8107a18ULL ^ (8 * m);

    k *= m;
    k ^= k >> r;
    k *= m;
    h ^= k;
    h *= m;
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return (khint_t)((h >> 32) ^ h);
}

#define pyrosm_int64_equal(a, b) ((a) == (b))

KHASH_INIT(i64set, int64_t, char, 0, pyrosm_hash_int64, pyrosm_int64_equal)
KHASH_INIT(i64map, int64_t, int64_t, 1, pyrosm_hash_int64, pyrosm_int64_equal)

#endif
