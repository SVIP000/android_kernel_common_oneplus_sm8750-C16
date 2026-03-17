// SPDX-License-Identifier: GPL-2.0
/*
 * cambyses_simd_neon.c — NEON argmax for Cambyses
 *
 * Finds index of maximum s16 score using SMAXP reduction +
 * SMAXV horizontal max + CMEQ + bitmask extraction.
 *
 * Must be called within cambyses_simd_begin/end.
 * Compiled with: CFLAGS_REMOVE += -mgeneral-regs-only
 *                CFLAGS += $(CC_FLAGS_FPU)
 *
 * Separate TU to prevent auto-vectorization contamination of scalar code.
 */

#include "cambyses.h"
#include <linux/types.h>

#ifdef CONFIG_SCHED_CAMBYSES_SIMD

#include <asm/neon-intrinsics.h>

/*
 * SIMD argmax: find index of highest s16 score using NEON.
 *
 * Algorithm:
 *   1. Load 32 scores into 4 Q registers (int16x8_t)
 *   2. SMAXP pair reductions: 4 → 2 → 1 register (8 maxima)
 *   3. SMAXV horizontal max across 8 lanes → scalar max_val
 *   4. DUP broadcast + CMEQ against all 4 original registers
 *   5. Bitmask extraction via AND with positional weights + ADDV
 *
 * @scores: array of SCHED_NR_MIGRATE_BREAK s16 scores (no alignment required).
 *          Unused entries must be S16_MIN (0x8000).
 *
 * AArch64 SMAXV provides hardware horizontal max in a single
 * instruction — no iterative reduction needed (unlike x86).
 *
 * Cost: ~20 ops ≈ 10–14 cycles on OOO (Cortex-X series).
 * vs scalar argmax: ~62 ops ≈ 20 cycles.  ~40% faster per extraction.
 */
int cambyses_simd_argmax_neon(const s16 *scores)
{
	int16x8_t s0, s1, s2, s3;
	int16x8_t m01, m23, m;
	int16_t max_val;
	int16x8_t bcast;
	uint16x8_t c0, c1, c2, c3;
	/*
	 * Positional weight vectors for bitmask extraction.
	 * After CMEQ, matching lanes are 0xFFFF.  AND with weights
	 * isolates a single bit per lane.  ADDV sums all bits into
	 * a compact bitmask.
	 */
	static const uint16x8_t pos_weights = {1, 2, 4, 8, 16, 32, 64, 128};
	uint16_t bits0, bits1, bits2, bits3;
	uint32_t combined;

	/* Load all 32 scores: 64 bytes = 4 Q registers */
	s0 = vld1q_s16(&scores[0]);
	s1 = vld1q_s16(&scores[8]);
	s2 = vld1q_s16(&scores[16]);
	s3 = vld1q_s16(&scores[24]);

	/* SMAXP pair reductions: 32 → 16 → 8 lane-wise maxima */
	m01 = vmaxq_s16(s0, s1);
	m23 = vmaxq_s16(s2, s3);
	m   = vmaxq_s16(m01, m23);

	/* SMAXV: hardware horizontal max across 8 lanes → scalar */
	max_val = vmaxvq_s16(m);

	/* Broadcast max value to all lanes */
	bcast = vdupq_n_s16(max_val);

	/* CMEQ: each lane → 0xFFFF if equal to max, 0 otherwise */
	c0 = vceqq_s16(s0, bcast);
	c1 = vceqq_s16(s1, bcast);
	c2 = vceqq_s16(s2, bcast);
	c3 = vceqq_s16(s3, bcast);

	/*
	 * Bitmask extraction: AND with positional weights, then ADDV
	 * to sum all bits.  Each matching lane contributes its bit
	 * weight (1, 2, 4, ..., 128) to the sum.
	 */
	bits0 = vaddvq_u16(vandq_u16(c0, pos_weights));
	bits1 = vaddvq_u16(vandq_u16(c1, pos_weights));
	bits2 = vaddvq_u16(vandq_u16(c2, pos_weights));
	bits3 = vaddvq_u16(vandq_u16(c3, pos_weights));

	/*
	 * Combine into a single 32-bit mask: 8 bits per register.
	 * __builtin_ctz finds the first set bit = candidate index.
	 */
	combined = (uint32_t)bits0 | ((uint32_t)bits1 << 8)
		 | ((uint32_t)bits2 << 16) | ((uint32_t)bits3 << 24);

	return __builtin_ctz(combined);
}

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
