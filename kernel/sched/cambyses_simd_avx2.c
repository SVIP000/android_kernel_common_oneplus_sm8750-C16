// SPDX-License-Identifier: GPL-2.0
/*
 * cambyses_simd_avx2.c — AVX2 argmax for Cambyses
 *
 * Finds the index of the maximum s16 score using PHMINPOSUW
 * (SSE4.1 horizontal min+position, always available under -mavx2).
 *
 * Scores are XOR'd with 0x7FFF to convert s16-max → u16-min, then
 * PHMINPOSUW returns both the minimum value and its lane index.
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
 * XOR mask: s16 max → u16 min conversion for PHMINPOSUW.
 *   score 1746  (0x06D2) → 0x792D (low u16 → selected)
 *   score -777  (0xFCF7) → 0x8308
 *   S16_MIN     (0x8000) → 0xFFFF (max u16 → never selected)
 */
static const v8hi phminpos_xor = {
	0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF,
	0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF
};

/*
 * argmax of a single XMM (8 × s16) via PHMINPOSUW.
 * Returns lane index (0–7) of the maximum s16 value.
 * On tie, returns the lowest index (PHMINPOSUW guarantee).
 */
static __always_inline int argmax_xmm(v8hi s)
{
	v8hi r = (v8hi)__builtin_ia32_phminposuw128(s ^ phminpos_xor);

	return (unsigned short)r[1];
}

/* 8 entries — 1 PHMINPOSUW (direct result) */
int cambyses_simd_argmax_avx2_8(const s16 *scores)
{
	return argmax_xmm(*(const v8hi_u *)&scores[0]);
}

/* 16 entries — 2 PHMINPOSUW + 1 scalar comparison */
int cambyses_simd_argmax_avx2_16(const s16 *scores)
{
	v8hi r0, r1;
	u16 v0, v1;

	r0 = (v8hi)__builtin_ia32_phminposuw128(
		*(const v8hi_u *)&scores[0] ^ phminpos_xor);
	r1 = (v8hi)__builtin_ia32_phminposuw128(
		*(const v8hi_u *)&scores[8] ^ phminpos_xor);

	v0 = (u16)r0[0]; v1 = (u16)r1[0];

	if (v0 <= v1)
		return (u16)r0[1];
	return 8 + (u16)r1[1];
}

/* 32 entries — 4 PHMINPOSUW + 3 scalar comparisons */
int cambyses_simd_argmax_avx2_32(const s16 *scores)
{
	v8hi r0, r1, r2, r3;
	u16 v0, v1, v2, v3;

	r0 = (v8hi)__builtin_ia32_phminposuw128(
		*(const v8hi_u *)&scores[0]  ^ phminpos_xor);
	r1 = (v8hi)__builtin_ia32_phminposuw128(
		*(const v8hi_u *)&scores[8]  ^ phminpos_xor);
	r2 = (v8hi)__builtin_ia32_phminposuw128(
		*(const v8hi_u *)&scores[16] ^ phminpos_xor);
	r3 = (v8hi)__builtin_ia32_phminposuw128(
		*(const v8hi_u *)&scores[24] ^ phminpos_xor);

	v0 = (u16)r0[0]; v1 = (u16)r1[0];
	v2 = (u16)r2[0]; v3 = (u16)r3[0];

	/* Lowest inverted value = highest original score; prefer lower index */
	if (v0 <= v1 && v0 <= v2 && v0 <= v3)
		return (u16)r0[1];
	if (v1 <= v2 && v1 <= v3)
		return 8 + (u16)r1[1];
	if (v2 <= v3)
		return 16 + (u16)r2[1];
	return 24 + (u16)r3[1];
}

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
