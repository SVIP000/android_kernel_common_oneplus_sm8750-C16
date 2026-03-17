// SPDX-License-Identifier: GPL-2.0
/*
 * cambyses_simd_ssse3.c — SSSE3 argmax for Cambyses
 *
 * Finds index of maximum s16 score using either:
 *   - PHMINPOSUW (SSE4.1, runtime-gated via static branch) — fast path
 *   - PMAXSW reduction + PCMPEQW + PMOVMSKB + BSF — fallback
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
 * PHMINPOSUW via inline asm — the instruction is SSE4.1 but this TU
 * is compiled with -mssse3, so __builtin_ia32_phminposuw128 is not
 * available.  The static branch ensures this is only reached on
 * SSE4.1-capable hardware.
 *
 * result[0] = minimum u16 value
 * result[1] = lane index (0–7) of the minimum
 */
static __always_inline v8hi __phminposuw128(v8hi a)
{
	v8hi result;

	asm("phminposuw %1, %0" : "=x" (result) : "xm" (a));
	return result;
}

/*
 * Portable wrappers for SSE2 instructions — GCC builtins
 * (__builtin_ia32_pmaxsw128 etc.) are not available in Clang.
 * Inline asm works on both compilers.
 */
static __always_inline v8hi __pmaxsw128(v8hi a, v8hi b)
{
	asm("pmaxsw %1, %0" : "+x" (a) : "xm" (b));
	return a;
}

static __always_inline int __pmovmskb128(v16qi a)
{
	int result;

	asm("pmovmskb %1, %0" : "=r" (result) : "x" (a));
	return result;
}

/*
 * XOR mask: s16 max → u16 min conversion for PHMINPOSUW.
 *   high s16 score → low u16 → selected by PHMINPOSUW
 *   S16_MIN (0x8000) → 0xFFFF → never selected
 */
static const v8hi phminpos_xor = {
	0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF,
	0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF
};

/*
 * SIMD argmax: find index of highest s16 score.
 *
 * Two paths selected at runtime via static branch:
 *
 * SSE4.1 (PHMINPOSUW): XOR + PHMINPOSUW per XMM, then scalar
 *   comparison across groups.  ~3 ops for 8 entries.
 *
 * SSSE3 fallback: PMAXSW reduction → scalar horizontal max →
 *   broadcast PCMPEQW → PMOVMSKB + BSF.  ~12 ops for 8 entries.
 *
 * @scores: array of CAMBYSES_SIMD_SCORES_SIZE s16 values.
 *          Unused entries must be S16_MIN.
 */
int cambyses_simd_argmax_ssse3(const s16 *scores)
{
#if CAMBYSES_SIMD_SCORES_SIZE >= 32
	v8hi s0, s1, s2, s3;

	s0 = *(const v8hi_u *)&scores[0];
	s1 = *(const v8hi_u *)&scores[8];
	s2 = *(const v8hi_u *)&scores[16];
	s3 = *(const v8hi_u *)&scores[24];

	if (static_branch_likely(&cambyses_has_sse41)) {
		v8hi r0, r1, r2, r3;
		u16 v0, v1, v2, v3;

		r0 = __phminposuw128(s0 ^ phminpos_xor);
		r1 = __phminposuw128(s1 ^ phminpos_xor);
		r2 = __phminposuw128(s2 ^ phminpos_xor);
		r3 = __phminposuw128(s3 ^ phminpos_xor);

		v0 = (u16)r0[0]; v1 = (u16)r1[0];
		v2 = (u16)r2[0]; v3 = (u16)r3[0];

		if (v0 <= v1 && v0 <= v2 && v0 <= v3)
			return (u16)r0[1];
		if (v1 <= v2 && v1 <= v3)
			return 8 + (u16)r1[1];
		if (v2 <= v3)
			return 16 + (u16)r2[1];
		return 24 + (u16)r3[1];
	} else {
		v8hi m01, m23, m, bcast, c0, c1, c2, c3;
		int j, mask0, mask1, mask2, mask3;
		s16 max_val;

		m01 = __pmaxsw128(s0, s1);
		m23 = __pmaxsw128(s2, s3);
		m   = __pmaxsw128(m01, m23);

		max_val = m[0];
		for (j = 1; j < 8; j++)
			if (m[j] > max_val)
				max_val = m[j];

		bcast = (v8hi){max_val, max_val, max_val, max_val,
			       max_val, max_val, max_val, max_val};

		c0 = (s0 == bcast);
		c1 = (s1 == bcast);
		c2 = (s2 == bcast);
		c3 = (s3 == bcast);
		mask0 = __pmovmskb128((v16qi)c0);
		mask1 = __pmovmskb128((v16qi)c1);
		mask2 = __pmovmskb128((v16qi)c2);
		mask3 = __pmovmskb128((v16qi)c3);

		if (mask0)
			return __builtin_ctz(mask0) >> 1;
		if (mask1)
			return 8 + (__builtin_ctz(mask1) >> 1);
		if (mask2)
			return 16 + (__builtin_ctz(mask2) >> 1);
		return 24 + (__builtin_ctz(mask3) >> 1);
	}

#elif CAMBYSES_SIMD_SCORES_SIZE >= 16
	v8hi s0, s1;

	s0 = *(const v8hi_u *)&scores[0];
	s1 = *(const v8hi_u *)&scores[8];

	if (static_branch_likely(&cambyses_has_sse41)) {
		v8hi r0, r1;
		u16 v0, v1;

		r0 = __phminposuw128(s0 ^ phminpos_xor);
		r1 = __phminposuw128(s1 ^ phminpos_xor);

		v0 = (u16)r0[0]; v1 = (u16)r1[0];

		if (v0 <= v1)
			return (u16)r0[1];
		return 8 + (u16)r1[1];
	} else {
		v8hi m, bcast, c0, c1;
		int j, mask0, mask1;
		s16 max_val;

		m = __pmaxsw128(s0, s1);

		max_val = m[0];
		for (j = 1; j < 8; j++)
			if (m[j] > max_val)
				max_val = m[j];

		bcast = (v8hi){max_val, max_val, max_val, max_val,
			       max_val, max_val, max_val, max_val};

		c0 = (s0 == bcast);
		c1 = (s1 == bcast);
		mask0 = __pmovmskb128((v16qi)c0);
		mask1 = __pmovmskb128((v16qi)c1);

		if (mask0)
			return __builtin_ctz(mask0) >> 1;
		return 8 + (__builtin_ctz(mask1) >> 1);
	}

#else /* CAMBYSES_SIMD_SCORES_SIZE == 8 */
	v8hi s0;

	s0 = *(const v8hi_u *)&scores[0];

	if (static_branch_likely(&cambyses_has_sse41)) {
		v8hi r = __phminposuw128(s0 ^ phminpos_xor);

		return (u16)r[1];
	} else {
		v8hi bcast, c0;
		int j, mask0;
		s16 max_val;

		max_val = s0[0];
		for (j = 1; j < 8; j++)
			if (s0[j] > max_val)
				max_val = s0[j];

		bcast = (v8hi){max_val, max_val, max_val, max_val,
			       max_val, max_val, max_val, max_val};

		c0 = (s0 == bcast);
		mask0 = __pmovmskb128((v16qi)c0);

		return __builtin_ctz(mask0) >> 1;
	}
#endif
}

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
