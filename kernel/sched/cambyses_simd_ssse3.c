// SPDX-License-Identifier: GPL-2.0
/*
 * cambyses_simd_ssse3.c — SSSE3 argmax for Cambyses
 *
 * Finds index of maximum s16 score using PMAXSW reduction +
 * PCMPEQW broadcast-compare + PMOVMSKB + BSF.
 *
 * Must be called within cambyses_simd_begin/end.
 * Compiled with: CFLAGS += -mssse3
 *
 * Separate TU to prevent auto-vectorization contamination of scalar code.
 */

#include "cambyses.h"
#include <linux/types.h>

#ifdef CONFIG_SCHED_CAMBYSES_SIMD

/*
 * SIMD argmax: find index of highest s16 score using SSSE3 (SSE2 subset).
 *
 * Algorithm:
 *   1. Load 32 scores into 4 xmm registers
 *   2. PMAXSW pair reductions: 4 → 2 → 1 xmm (8 maxima)
 *   3. Scalar horizontal max across 8 values (7 comparisons)
 *   4. Broadcast max value, PCMPEQW against all 4 original xmm
 *   5. PMOVMSKB + BSF to find first matching position
 *
 * @scores: array of SCHED_NR_MIGRATE_BREAK s16 scores (no alignment required).
 *          Unused entries must be S16_MIN.
 *
 * Cost: ~24 SIMD ops + ~7 scalar comparisons ≈ 12–16 cycles on OOO.
 * vs scalar argmax: ~62 ops ≈ 20 cycles.  ~30% faster per extraction.
 */
int cambyses_simd_argmax_ssse3(const s16 *scores)
{
	v8hi s0, s1, s2, s3, m01, m23, m, bcast, c0, c1, c2, c3;
	int j, mask0, mask1, mask2, mask3;
	s16 max_val;

	/* Load all 32 scores: 64 bytes = 4 xmm (unaligned via v8hi_u) */
	s0 = *(const v8hi_u *)&scores[0];
	s1 = *(const v8hi_u *)&scores[8];
	s2 = *(const v8hi_u *)&scores[16];
	s3 = *(const v8hi_u *)&scores[24];

	/* PMAXSW pair reductions: 32 → 16 → 8 lane-wise maxima */
	m01 = __builtin_ia32_pmaxsw128(s0, s1);
	m23 = __builtin_ia32_pmaxsw128(s2, s3);
	m   = __builtin_ia32_pmaxsw128(m01, m23);

	/* Horizontal max across 8 lanes */
	max_val = m[0];
	for (j = 1; j < 8; j++)
		if (m[j] > max_val)
			max_val = m[j];

	/* Broadcast max value to all 8 lanes */
	bcast = (v8hi){max_val, max_val, max_val, max_val,
		       max_val, max_val, max_val, max_val};

	/* PCMPEQW + PMOVMSKB: find first matching position */
	c0 = (s0 == bcast);
	c1 = (s1 == bcast);
	c2 = (s2 == bcast);
	c3 = (s3 == bcast);
	mask0 = __builtin_ia32_pmovmskb128((v16qi)c0);
	mask1 = __builtin_ia32_pmovmskb128((v16qi)c1);
	mask2 = __builtin_ia32_pmovmskb128((v16qi)c2);
	mask3 = __builtin_ia32_pmovmskb128((v16qi)c3);

	if (mask0)
		return __builtin_ctz(mask0) >> 1;
	if (mask1)
		return 8 + (__builtin_ctz(mask1) >> 1);
	if (mask2)
		return 16 + (__builtin_ctz(mask2) >> 1);
	return 24 + (__builtin_ctz(mask3) >> 1);
}

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
