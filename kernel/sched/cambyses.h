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
extern u8 sysctl_cambyses_w0;	/* cache coldness weight (default: 1) */
extern u8 sysctl_cambyses_w1;	/* load contribution weight (default: 2) */
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
 * Finds the index of the maximum s16 score across all
 * SCHED_NR_MIGRATE_BREAK entries.  Unused entries must be S16_MIN.
 * No alignment requirement — loads are unaligned-safe.
 */
#ifdef CONFIG_SCHED_CAMBYSES_SIMD

#define CAMBYSES_SIMD_THRESHOLD		8

/* Static keys for ISA dispatch — enabled at boot based on CPUID/HWCAP */
#ifdef CONFIG_X86_64
extern struct static_key_false cambyses_has_avx2;
extern struct static_key_false cambyses_has_ssse3;
#endif
#ifdef CONFIG_ARM64
extern struct static_key_false cambyses_has_neon;
#endif

#ifdef CONFIG_X86_64
/*
 * AVX2 argmax: loads 32 × s16 into 2 ymm, VPMAXSW reduction,
 * VPCMPEQW + VPMOVMSKB + BSF for position.  ~16 ops per extraction.
 */
int cambyses_simd_argmax_avx2(const s16 *scores);

/*
 * SSSE3 argmax: loads 32 × s16 into 4 xmm, PMAXSW reduction,
 * PCMPEQW + PMOVMSKB + BSF for position.  ~24 ops per extraction.
 */
int cambyses_simd_argmax_ssse3(const s16 *scores);
#endif

#ifdef CONFIG_ARM64
/*
 * NEON argmax: loads 32 × s16 into 4 Q registers, SMAXP reduction,
 * SMAXV + CMEQ + bitmask extraction.  ~20 ops per extraction.
 */
int cambyses_simd_argmax_neon(const s16 *scores);
#endif

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
