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
 * Register usage adapts to CAMBYSES_SIMD_SCORES_SIZE at compile time:
 *   32: 4 Q register loads, 3-stage vmaxq reduction
 *   16: 2 Q register loads, 1-stage reduction
 *    8: 1 Q register load, direct SMAXV
 *
 * @scores: array of CAMBYSES_SIMD_SCORES_SIZE s16 values.
 *          Unused entries must be S16_MIN (0x8000).
 */
int cambyses_simd_argmax_neon(const s16 *scores)
{
	static const uint16x8_t pos_weights = {1, 2, 4, 8, 16, 32, 64, 128};

#if CAMBYSES_SIMD_SCORES_SIZE >= 32
	int16x8_t s0, s1, s2, s3;
	int16x8_t m01, m23, m;
	int16_t max_val;
	int16x8_t bcast;
	uint16x8_t c0, c1, c2, c3;
	uint16_t bits0, bits1, bits2, bits3;
	uint32_t combined;

	s0 = vld1q_s16(&scores[0]);
	s1 = vld1q_s16(&scores[8]);
	s2 = vld1q_s16(&scores[16]);
	s3 = vld1q_s16(&scores[24]);

	m01 = vmaxq_s16(s0, s1);
	m23 = vmaxq_s16(s2, s3);
	m   = vmaxq_s16(m01, m23);

	max_val = vmaxvq_s16(m);
	bcast = vdupq_n_s16(max_val);

	c0 = vceqq_s16(s0, bcast);
	c1 = vceqq_s16(s1, bcast);
	c2 = vceqq_s16(s2, bcast);
	c3 = vceqq_s16(s3, bcast);

	bits0 = vaddvq_u16(vandq_u16(c0, pos_weights));
	bits1 = vaddvq_u16(vandq_u16(c1, pos_weights));
	bits2 = vaddvq_u16(vandq_u16(c2, pos_weights));
	bits3 = vaddvq_u16(vandq_u16(c3, pos_weights));

	combined = (uint32_t)bits0 | ((uint32_t)bits1 << 8)
		 | ((uint32_t)bits2 << 16) | ((uint32_t)bits3 << 24);

	return __builtin_ctz(combined);

#elif CAMBYSES_SIMD_SCORES_SIZE >= 16
	int16x8_t s0, s1, m;
	int16_t max_val;
	int16x8_t bcast;
	uint16x8_t c0, c1;
	uint16_t bits0, bits1;
	uint32_t combined;

	s0 = vld1q_s16(&scores[0]);
	s1 = vld1q_s16(&scores[8]);

	m = vmaxq_s16(s0, s1);

	max_val = vmaxvq_s16(m);
	bcast = vdupq_n_s16(max_val);

	c0 = vceqq_s16(s0, bcast);
	c1 = vceqq_s16(s1, bcast);

	bits0 = vaddvq_u16(vandq_u16(c0, pos_weights));
	bits1 = vaddvq_u16(vandq_u16(c1, pos_weights));

	combined = (uint32_t)bits0 | ((uint32_t)bits1 << 8);

	return __builtin_ctz(combined);

#else /* CAMBYSES_SIMD_SCORES_SIZE == 8 */
	int16x8_t s0;
	int16_t max_val;
	int16x8_t bcast;
	uint16x8_t c0;
	uint16_t bits0;

	s0 = vld1q_s16(&scores[0]);

	max_val = vmaxvq_s16(s0);
	bcast = vdupq_n_s16(max_val);

	c0 = vceqq_s16(s0, bcast);
	bits0 = vaddvq_u16(vandq_u16(c0, pos_weights));

	return __builtin_ctz(bits0);
#endif
}

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
