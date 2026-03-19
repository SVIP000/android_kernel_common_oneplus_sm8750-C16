/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Cambyses — Context-Aware Migration Balancer Yielding Scored Entity Selection
 *
 * Scored migration selection for CFS load balancer Pull path.
 * See IMPLEMENTATION.md for design details.
 */
#ifndef _KERNEL_SCHED_CAMBYSES_H
#define _KERNEL_SCHED_CAMBYSES_H

#include <linux/types.h>
#include <linux/jump_label.h>

#ifdef CONFIG_SCHED_CAMBYSES

/*
 * Candidate entry for scored migration selection.
 * Stored on stack during detach_tasks_cambyses().
 * Scores are kept in a separate s16 array for SIMD argmax access.
 *
 * Size: 8 bytes (pointer only)
 * Stack usage: 32 candidates = 256 bytes, 8 candidates = 64 bytes
 */
struct cambyses_candidate {
	struct task_struct	*p;
};

/* Static key for zero-cost runtime disable (NOP patching) */
extern struct static_key_true sched_cambyses;

/* sysctl tunable weights (0–3, 2bit) */
extern u8 sysctl_cambyses_w0;	/* cache coldness weight (default: 2) */
extern u8 sysctl_cambyses_w1;	/* CPU lightness weight (default: 3) */
extern u8 sysctl_cambyses_w2;	/* vol switch ratio weight (default: 1) */
extern u8 sysctl_cambyses_w3;	/* wakee penalty weight (default: 1) */

/*
 * Score Shadow — cached feature values updated at natural update points.
 * Called from core.c (__schedule) and fair.c (record_wakee) to maintain
 * se.cambyses_f2/f3 so Phase 1 scoring avoids 2 DRAM cache-line misses.
 */
void cambyses_update_f2(struct task_struct *p);
void cambyses_update_f3(struct task_struct *p);

/*
 * SIMD argmax — separate TUs to prevent auto-vectorization contamination.
 * Each file is compiled with its own ISA flags.
 *
 * Three size variants per ISA (8, 16, 32 entries) are exported
 * unconditionally.  SIMD TUs cannot include sched.h (auto-vectorization
 * contamination), so they have no access to SCHED_NR_MIGRATE_BREAK.
 * Instead, cambyses.c (which includes sched.h) selects the correct
 * variant at compile time via dispatch macros below.
 *
 * x86 uses PHMINPOSUW (SSE4.1) for O(1) horizontal argmax per XMM:
 *   XOR 0x7FFF converts s16-max to u16-min, PHMINPOSUW returns both
 *   the minimum value and its lane index in one instruction.
 *   AVX2 always has SSE4.1; SSSE3 path uses it when detected at boot.
 *
 * Register usage per variant:
 *   _32: 4 xmm loads → 4 PHMINPOSUW + 3 comparisons
 *   _16: 2 xmm loads → 2 PHMINPOSUW + 1 comparison
 *    _8: 1 xmm load  → 1 PHMINPOSUW (direct result)
 */
#ifdef CONFIG_SCHED_CAMBYSES_SIMD

/*
 * Scores array size for SIMD argmax — rounded up from
 * SCHED_NR_MIGRATE_BREAK to the next SIMD-friendly boundary.
 * This ensures every XMM/Q load stays within bounds.
 *
 * SCHED_NR_MIGRATE_BREAK is defined in sched.h, which SIMD TUs
 * cannot include.  Guard with #ifdef so SIMD TUs compile cleanly;
 * they see only the raw _8/_16/_32 declarations below.
 */
#ifdef SCHED_NR_MIGRATE_BREAK
#if SCHED_NR_MIGRATE_BREAK <= 8
#define CAMBYSES_SIMD_SCORES_SIZE	8
#elif SCHED_NR_MIGRATE_BREAK <= 16
#define CAMBYSES_SIMD_SCORES_SIZE	16
#elif SCHED_NR_MIGRATE_BREAK <= 32
#define CAMBYSES_SIMD_SCORES_SIZE	32
#else
#error "SCHED_NR_MIGRATE_BREAK > 32 not supported by Cambyses SIMD argmax"
#endif

/*
 * Minimum candidates to enter SIMD path.
 * Must be <= SCHED_NR_MIGRATE_BREAK.
 */
#define CAMBYSES_SIMD_THRESHOLD		min_t(int, 8, SCHED_NR_MIGRATE_BREAK)
#endif /* SCHED_NR_MIGRATE_BREAK */

/* Static keys for ISA dispatch — enabled at boot based on CPUID/HWCAP */
#ifdef CONFIG_X86_64
extern struct static_key_false cambyses_has_avx2;
extern struct static_key_false cambyses_has_ssse3;
extern struct static_key_false cambyses_has_sse41;
#endif
#ifdef CONFIG_ARM64
extern struct static_key_false cambyses_has_neon;
#endif

/* Size-specific SIMD argmax — all variants always compiled */
#ifdef CONFIG_X86_64
int cambyses_simd_argmax_avx2_8(const s16 *scores);
int cambyses_simd_argmax_avx2_16(const s16 *scores);
int cambyses_simd_argmax_avx2_32(const s16 *scores);
int cambyses_simd_argmax_ssse3_8(const s16 *scores);
int cambyses_simd_argmax_ssse3_16(const s16 *scores);
int cambyses_simd_argmax_ssse3_32(const s16 *scores);
#endif

#ifdef CONFIG_ARM64
int cambyses_simd_argmax_neon_8(const s16 *scores);
int cambyses_simd_argmax_neon_16(const s16 *scores);
int cambyses_simd_argmax_neon_32(const s16 *scores);
#endif

/*
 * Compile-time dispatch — maps generic name to the size-correct variant
 * via token pasting (e.g. _avx2 → _avx2_8 when SCORES_SIZE == 8).
 * Only available when SCHED_NR_MIGRATE_BREAK is known (i.e. in cambyses.c
 * via fair.c / sched.h).  SIMD TUs never see these macros.
 */
#ifdef CAMBYSES_SIMD_SCORES_SIZE
#define _CAMBYSES_PASTE(fn, sz)		fn##sz
#define _CAMBYSES_DISPATCH(fn, sz)	_CAMBYSES_PASTE(fn, sz)
#define cambyses_simd_argmax_avx2(s)	_CAMBYSES_DISPATCH(cambyses_simd_argmax_avx2_, CAMBYSES_SIMD_SCORES_SIZE)(s)
#define cambyses_simd_argmax_ssse3(s)	_CAMBYSES_DISPATCH(cambyses_simd_argmax_ssse3_, CAMBYSES_SIMD_SCORES_SIZE)(s)
#define cambyses_simd_argmax_neon(s)	_CAMBYSES_DISPATCH(cambyses_simd_argmax_neon_, CAMBYSES_SIMD_SCORES_SIZE)(s)
#endif /* CAMBYSES_SIMD_SCORES_SIZE */

/*
 * x86 SIMD vector type definitions — only available in TUs compiled
 * with the corresponding ISA flags.
 *
 * Following Nap's pattern: GCC vector extensions + __builtin_ia32_*.
 * <immintrin.h> is a userspace header and cannot be used in kernel.
 */
#ifdef __SSSE3__
typedef short v8hi  __attribute__((__vector_size__(16)));  /* 8 × s16 */
typedef char  v16qi __attribute__((__vector_size__(16)));   /* 16 × s8/u8 */
/* Unaligned load type — forces MOVDQU regardless of source alignment */
typedef short v8hi_u  __attribute__((__vector_size__(16), __aligned__(2)));
#endif

#ifdef __AVX2__
typedef short v16hi __attribute__((__vector_size__(32)));   /* 16 × s16 */
typedef char  v32qi __attribute__((__vector_size__(32)));    /* 32 × s8/u8 */
/* Unaligned load type — forces VMOVDQU regardless of source alignment */
typedef short v16hi_u __attribute__((__vector_size__(32), __aligned__(2)));
#endif

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */

#endif /* CONFIG_SCHED_CAMBYSES */
#endif /* _KERNEL_SCHED_CAMBYSES_H */
