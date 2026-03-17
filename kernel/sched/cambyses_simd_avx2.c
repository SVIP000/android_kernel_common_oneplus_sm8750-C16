// SPDX-License-Identifier: GPL-2.0
/*
 * cambyses_simd_avx2.c — AVX2 argmax for Cambyses
 *
 * Finds the index of the maximum s16 score using VPMAXSW
 * reduction + VPCMPEQW broadcast-compare + VPMOVMSKB + BSF.
 *
 * Must be called within cambyses_simd_begin/end.
 * Compiled with: CFLAGS += -mavx -mavx2
 *
 * Separate TU to prevent auto-vectorization contamination of scalar code.
 */

#include "cambyses.h"
#include <linux/types.h>

#ifdef CONFIG_SCHED_CAMBYSES_SIMD

/*
 * SIMD argmax: find index of highest s16 score using AVX2.
 *
 * Algorithm:
 *   1. Load 32 scores into 2 ymm registers (s0, s1)
 *   2. VPMAXSW to reduce 32 → 16 lane-wise maxima
 *   3. Scalar horizontal max across 16 values (15 comparisons)
 *   4. Broadcast max value to ymm, VPCMPEQW against s0/s1
 *   5. VPMOVMSKB + BSF to find first matching position
 *
 * @scores: array of SCHED_NR_MIGRATE_BREAK s16 scores (no alignment required).
 *          Unused entries must be S16_MIN.
 *
 * Cost: ~16 SIMD ops + ~15 scalar comparisons ≈ 8–12 cycles on OOO.
 * vs scalar argmax: ~62 ops ≈ 20 cycles.  ~60% faster per extraction.
 */
int cambyses_simd_argmax_avx2(const s16 *scores)
{
	v16hi s0, s1, m, bcast, c0, c1;
	int j, mask0, mask1;
	s16 max_val;

	/* Load all 32 scores: 64 bytes = 2 ymm (unaligned via v16hi_u) */
	s0 = *(const v16hi_u *)&scores[0];
	s1 = *(const v16hi_u *)&scores[16];

	/* VPMAXSW: lane-wise max, 32 → 16 values */
	m = __builtin_ia32_pmaxsw256(s0, s1);

	/*
	 * Horizontal max across 16 lanes.  Staying in SIMD for this
	 * would need VPERM2I128 + VPSHUFD + VPSHUFLW (6 shuffle+max
	 * pairs) then a GPR extract — roughly the same cost as 15
	 * scalar comparisons from vector lane extraction.  GCC
	 * compiles m[j] to VPEXTRW which is 1 µop on Intel/AMD.
	 */
	max_val = m[0];
	for (j = 1; j < 16; j++)
		if (m[j] > max_val)
			max_val = m[j];

	/* Broadcast max value to all 16 lanes */
	bcast = (v16hi){max_val, max_val, max_val, max_val,
			max_val, max_val, max_val, max_val,
			max_val, max_val, max_val, max_val,
			max_val, max_val, max_val, max_val};

	/*
	 * VPCMPEQW: each lane → 0xFFFF if equal to max, 0 otherwise.
	 * VPMOVMSKB: extract MSB of each byte → 32-bit mask.
	 * Each s16 match produces 2 consecutive set bits in the mask.
	 * BSF finds the first set bit; dividing by 2 gives the s16 index.
	 */
	c0 = (s0 == bcast);
	c1 = (s1 == bcast);
	mask0 = __builtin_ia32_pmovmskb256((v32qi)c0);
	mask1 = __builtin_ia32_pmovmskb256((v32qi)c1);

	if (mask0)
		return __builtin_ctz(mask0) >> 1;
	return 16 + (__builtin_ctz(mask1) >> 1);
}

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
