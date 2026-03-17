// SPDX-License-Identifier: GPL-2.0
/*
 * Cambyses — Context-Aware Migration Balancer Yielding Scored Entity Selection
 *
 * Scored migration selection for CFS load balancer.
 * Replaces FIFO selection with a multi-feature scoring function that evaluates
 * cache coldness, load contribution, voluntary switch ratio, and wakee stability.
 */

/*
 * This file is #included from fair.c (not compiled separately)
 * to access static functions: can_migrate_task(), detach_task(),
 * task_h_load(), task_util_est(), task_fits_cpu(), etc.
 */

/**************************************************************
 * Version Information:
 */

#define CAMBYSES_PROGNAME "Cambyses Migration Selector"
#define CAMBYSES_AUTHOR   "Masahito Suzuki"

#define CAMBYSES_VERSION  "0.2.0"

/* Runtime toggle — NOP-patched when disabled */
DEFINE_STATIC_KEY_TRUE(sched_cambyses);

/* Default weights: w1=2 (load contribution dominant), w0=w2=w3=1 */
u8 sysctl_cambyses_w0 = 1;
u8 sysctl_cambyses_w1 = 2;
u8 sysctl_cambyses_w2 = 1;
u8 sysctl_cambyses_w3 = 1;

#ifdef CONFIG_SCHED_CAMBYSES_SIMD
#ifdef CONFIG_X86_64
DEFINE_STATIC_KEY_FALSE(cambyses_has_avx2);
DEFINE_STATIC_KEY_FALSE(cambyses_has_ssse3);
#endif
#ifdef CONFIG_ARM64
DEFINE_STATIC_KEY_FALSE(cambyses_has_neon);
#endif
#endif

/*
 * log2p1_u64_u8fp2 — fixed-point log2(v+1) with 2-bit mantissa
 *
 * Returns (exponent << 2) | mantissa, giving 4× finer granularity than
 * fls64().  Based on BORE scheduler's log2p1_u64_u32fp().
 *
 * Result fits in u8 for all practical scheduler values (max ~163 for
 * nanosecond-scale deltas).
 */
static inline u8 log2p1_u64_u8fp2(u64 v)
{
	int clz, exponent;
	u8 mantissa;

	if (unlikely(!v))
		return 0;
	clz = __builtin_clzll(v);
	exponent = 64 - clz;
	mantissa = (u8)((v << clz) << 1 >> 62);
	return (u8)(exponent << 2 | mantissa);
}

/*
 * vol_switch_ratio_u6 — voluntary context switch ratio, 0–64
 *
 * Approximates (nvcsw / (nvcsw + nivcsw)) × 64 without division, using
 * log2 subtraction: log2(a/b) = log2(a) - log2(b).
 *
 * Both nvcsw and total are converted to pseudo-log2 via log2p1_u64_u8fp2
 * (CLZ + 2 shifts each), then subtracted.  The +1 offsets in log2p1 cancel.
 * Result is offset by +64 and clamped to [0, 64].
 *
 * Higher value = more I/O-bound = transient cache footprint = cheaper to migrate.
 */
static inline u8 vol_switch_ratio_u6(struct task_struct *p)
{
	unsigned long total = p->nvcsw + p->nivcsw;

	if (unlikely(!total))
		return 0;
	return (u8)clamp_t(int,
		(int)log2p1_u64_u8fp2(p->nvcsw)
		- (int)log2p1_u64_u8fp2(total) + 64,
		0, 64);
}

/*
 * cambyses_update_f2 — recompute and cache the voluntary switch ratio.
 *
 * Called from __schedule() after incrementing nvcsw/nivcsw.
 * The cached value lives at se.cambyses_f2, on the same cache line
 * as group_node — so Phase 1 scoring reads it for free.
 */
void cambyses_update_f2(struct task_struct *p)
{
	p->se.cambyses_f2 = vol_switch_ratio_u6(p);
}

/*
 * cambyses_update_f3 — recompute and cache the wakee penalty.
 *
 * Called from record_wakee() after modifying wakee_flips.
 * Includes both the increment and decay paths.
 */
void cambyses_update_f3(struct task_struct *p)
{
	p->se.cambyses_f3 = (u8)log2p1_u64_u8fp2(p->wakee_flips + 1);
}

/*
 * prefetch_migration_task — prefetch task_struct cache lines that
 * can_migrate_task() + score_task_cambyses() will access.
 *
 * On in-order CPUs (Cortex-A55, Atom Bonnell, RISC-V), the fields
 * accessed by scoring span multiple separate cache lines.  Without
 * OOO execution to overlap the loads, each miss serializes (~400cy).
 * Issuing prefetches one iteration ahead hides most of that latency.
 *
 * F2 (vol_switch_ratio) and F3 (wakee_penalty) are cached in
 * se.cambyses_f2/f3, co-located with group_node on the same cache
 * line — no prefetch needed for those.  This reduces DRAM misses
 * from 4 to 2 per task on limited-MLP cores (Silvermont, Gracemont).
 *
 * On modern OOO CPUs (Zen 4, Golden Cove), this is effectively free:
 * prefetch instructions for already-in-flight loads are NOPs.
 */
static inline void prefetch_migration_task(struct task_struct *p)
{
	/* F0: se.exec_start — same cache line as group_node, but prefetch
	 * ensures the line is in-flight before we dereference group_node. */
	prefetch(&p->se.exec_start);
	/* F1: se.avg.load_avg (used by task_h_load) — separate cache line */
	prefetch(&p->se.avg.load_avg);
	/* can_migrate_task: cpus_ptr + migration_disabled — separate cache line */
	prefetch(&p->cpus_ptr);
	/* F2/F3: cached in se.cambyses_f2/f3 (same cache line as group_node)
	 * — no prefetch needed, eliminating 2 DRAM misses per task. */
}

/*
 * score_task_cambyses — compute migration suitability score for a task
 *
 * Features (u8, range varies per feature):
 *   F0: cache coldness       — log2p1(time since last exec) (higher = colder = better)
 *   F1: load contribution    — log2p1(task hierarchical load) (higher = more effective)
 *   F2: vol switch ratio     — nvcsw/(nvcsw+nivcsw) × 64 (higher = I/O-bound = cheaper)
 *   F3: wakee penalty        — log2p1(wakee_flips + 1) (higher = riskier)
 *
 * Score range: max ~2295, min ~-765 → fits in s16.
 */
static s16 score_task_cambyses(struct task_struct *p, struct lb_env *env)
{
	int f0 = log2p1_u64_u8fp2(rq_clock_task(env->src_rq) - p->se.exec_start);
	int f1 = log2p1_u64_u8fp2(max_t(unsigned long, task_h_load(p), 1));
	/* F2/F3: read from Score Shadow cache (same cache line as group_node) */
	int f2 = p->se.cambyses_f2;
	int f3 = p->se.cambyses_f3;

	return (s16)((int)sysctl_cambyses_w0 * f0
		   + (int)sysctl_cambyses_w1 * f1
		   + (int)sysctl_cambyses_w2 * f2
		   - (int)sysctl_cambyses_w3 * f3);
}

/*
 * check_imbalance_cambyses — coarse filter for Phase 1 candidate inclusion
 *
 * Mirrors the existing detach_tasks() switch logic but does NOT consume
 * imbalance (that happens in Phase 3 after sorting by score).
 */
static bool check_imbalance_cambyses(struct task_struct *p,
				     struct lb_env *env)
{
	switch (env->migration_type) {
	case migrate_load: {
		unsigned long load = max_t(unsigned long, task_h_load(p), 1);

		if (sched_feat(LB_MIN) &&
		    load < 16 && !env->sd->nr_balance_failed)
			return false;
		if (shr_bound(load, env->sd->nr_balance_failed) > env->imbalance)
			return false;
		return true;
	}
	case migrate_util: {
		unsigned long util = task_util_est(p);

		if (shr_bound(util, env->sd->nr_balance_failed) > env->imbalance)
			return false;
		return true;
	}
	case migrate_task:
		return true;
	case migrate_misfit:
		return !task_fits_cpu(p, env->src_cpu);
	}
	return false;
}

/*
 * consume_imbalance_cambyses — deduct task cost from imbalance budget
 *
 * Called in Phase 3 for each detached task, in score-descending order.
 * Matches Vanilla behavior: allows imbalance to go negative on the last task.
 */
static void consume_imbalance_cambyses(struct task_struct *p,
				       struct lb_env *env)
{
	switch (env->migration_type) {
	case migrate_load:
		env->imbalance -= max_t(unsigned long, task_h_load(p), 1);
		break;
	case migrate_util:
		env->imbalance -= task_util_est(p);
		break;
	case migrate_task:
		env->imbalance--;
		break;
	case migrate_misfit:
		env->imbalance = 0;
		break;
	}
}

#ifdef CONFIG_SCHED_CAMBYSES_SIMD

/*
 * cambyses_simd_begin/end — FPU/FPSIMD context management for
 * IRQ-disabled scheduler context.
 *
 * x86: uses kernel_fpu_begin/end (handles save, lazy restore, and
 *      FPU register cache invalidation).
 *
 * ARM64: kernel_neon_begin() BUG_ON's with irqs_disabled(), so we
 *        bypass it by driving the FPSIMD save directly.  If
 *        TIF_FOREIGN_FPSTATE is already set (common case after context
 *        switch), the state is already saved — cost ≈ 0.
 */

#ifdef CONFIG_X86_64
#include <asm/fpu/api.h>

static __always_inline void cambyses_simd_begin(void)
{
	kernel_fpu_begin();
}

static __always_inline void cambyses_simd_end(void)
{
	kernel_fpu_end();
}
#endif /* CONFIG_X86_64 */

#ifdef CONFIG_ARM64
#include <asm/fpsimd.h>

static __always_inline void cambyses_simd_begin(void)
{
	if (!(current->flags & PF_KTHREAD) &&
	    !test_thread_flag(TIF_FOREIGN_FPSTATE)) {
		/*
		 * Invalidate the per-CPU FPSIMD register cache.  Without
		 * this, fpsimd_thread_switch() would see
		 *   fpsimd_last_state.st == &current->thread.uw.fpsimd_state
		 *   && fpsimd_cpu == this_cpu
		 * and CLEAR TIF_FOREIGN_FPSTATE — causing
		 * fpsimd_restore_current_state() to skip the restore on
		 * return to userspace, with NEON-clobbered registers.
		 *
		 * kernel_neon_begin() does this via fpsimd_flush_cpu_state().
		 * We write NULL directly to avoid the full flush (which
		 * also handles SME streaming mode, unnecessary here).
		 */
		fpsimd_save_and_flush_cpu_state();
	}
}

static __always_inline void cambyses_simd_end(void)
{
	/* Nothing — lazy restore via fpsimd_restore_current_state() */
}
#endif /* CONFIG_ARM64 */

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */


/*
 * argmax_scores — find index of highest score (branchless).
 *
 * Compiles to CMP + CMOV per iteration — no branches, no mispredictions.
 * For 32 candidates: ~62 micro-ops, ~20 cycles on modern OOO.
 *
 * Replaces full bitonic sort: instead of O(n log²n) comparators to
 * establish total order, extract the single best candidate in O(n).
 * Repeated extraction (K times) costs O(K*n) — for typical K=1..4
 * this is 5-20x faster than a full sort.
 */
static __always_inline int argmax_scores(const s16 *scores, int nr_cands)
{
	int best = 0, j;

	for (j = 1; j < nr_cands; j++)
		if (scores[j] > scores[best])
			best = j;
	return best;
}

/*
 * detach_tasks_cambyses — scored migration selection for Pull path
 *
 * Called from detach_tasks() when sched_cambyses is active.
 * Operates under rq_lock_irqsave (inherited from caller).
 *
 * Phase 1: Sample candidates from cfs_tasks, score each one
 * Phase 2: Repeated argmax extraction (SIMD or scalar), consuming
 *          imbalance budget
 */
static int detach_tasks_cambyses(struct lb_env *env)
{
	struct list_head *tasks = &env->src_rq->cfs_tasks;
	struct cambyses_candidate cands[SCHED_NR_MIGRATE_BREAK];
	s16 scores[SCHED_NR_MIGRATE_BREAK];
	LIST_HEAD(cand_tasks);
	int nr_cands = 0;
	int detached = 0;
	struct task_struct *p;

	/*
	 * Phase 1: Sampling — collect eligible candidates and score them.
	 *
	 * Mirrors the loop structure of vanilla detach_tasks():
	 * same loop_max, loop_break, idle checks.
	 *
	 * Accepted candidates are moved to cand_tasks (off cfs_tasks)
	 * to prevent the list rotation from re-collecting the same task
	 * as the loop wraps around.  Skipped tasks stay on cfs_tasks.
	 */
	while (!list_empty(tasks)) {
		if (env->idle && env->src_rq->nr_running <= 1)
			break;

		env->loop++;
		if (env->loop > env->loop_max)
			break;
		if (env->loop > env->loop_break) {
			env->loop_break += SCHED_NR_MIGRATE_BREAK;
			env->flags |= LBF_NEED_BREAK;
			break;
		}

		p = list_last_entry(tasks, struct task_struct, se.group_node);

		/*
		 * Prefetch next candidate (one-ahead).  We scan from
		 * the tail, so the next candidate is prev in the list.
		 * On in-order CPUs this hides ~60% of DRAM stall time
		 * by overlapping the prefetch with current task scoring.
		 * On OOO CPUs the hardware already parallelizes the
		 * loads, so these prefetches are essentially free NOPs.
		 */
		if (p->se.group_node.prev != tasks)
			prefetch_migration_task(
				list_prev_entry(p, se.group_node));

		if (!can_migrate_task(p, env)) {
			list_move(&p->se.group_node, tasks);
			continue;
		}

		if (!check_imbalance_cambyses(p, env))
			goto skip;

		/* Score and collect */
		cands[nr_cands].p = p;
		scores[nr_cands] = score_task_cambyses(p, env);
		nr_cands++;

		/*
		 * Remove from cfs_tasks so it cannot be re-scanned.
		 * detach_task() in Phase 3 will list_del this node;
		 * non-detached candidates are spliced back below.
		 */
		list_move(&p->se.group_node, &cand_tasks);

		if (nr_cands >= SCHED_NR_MIGRATE_BREAK)
			break;
		continue;
skip:
		list_move(&p->se.group_node, tasks);
	}

	if (!nr_cands) {
		/* cand_tasks is empty here — nothing to splice back */
		return 0;
	}

	/*
	 * Phase 2: Repeated argmax extraction, consuming imbalance budget.
	 *
	 * Repeatedly extract the single best candidate in O(n) and mark
	 * it consumed.  For typical K=1..4, this is 5-20x faster than
	 * a full sort (O(n log²n)).
	 *
	 * SIMD path (AVX2/SSSE3/NEON): loads the entire scores[] array
	 * into vector registers, uses PMAXSW reduction + broadcast compare
	 * + bitmask scan.  ~60% faster per extraction than scalar.
	 * FPU context cost ≈ 0 (TIF_NEED_FPU_LOAD already set after
	 * context switch in the common scheduler path).
	 *
	 * Scalar fallback: branchless CMP+CMOV loop, ~20 cycles / extraction.
	 *
	 * S16_MIN (-32768) is used as a tombstone: real scores range
	 * from -765 to +2295, so it can never be a valid score.
	 */
#ifdef CONFIG_SCHED_CAMBYSES_SIMD
	if (nr_cands >= CAMBYSES_SIMD_THRESHOLD) {
		int selected[SCHED_NR_MIGRATE_BREAK];
		int nr_selected = 0;
		int j;

		/* Pad unused entries so SIMD loads see S16_MIN */
		for (j = nr_cands; j < SCHED_NR_MIGRATE_BREAK; j++)
			scores[j] = S16_MIN;

		/*
		 * Phase 2a: SIMD selection — pick winners inside FPU context.
		 * Repeated argmax extracts candidates in score order,
		 * consuming imbalance budget.  detach_task() is deferred
		 * to Phase 2b (outside kernel_fpu) to avoid holding
		 * kernel_fpu_begin across dequeue_entity callbacks.
		 */

		cambyses_simd_begin();
		while (env->imbalance > 0) {
			int best;

#ifdef CONFIG_X86_64
			if (static_branch_likely(&cambyses_has_avx2))
				best = cambyses_simd_argmax_avx2(scores);
			else if (static_branch_likely(&cambyses_has_ssse3))
				best = cambyses_simd_argmax_ssse3(scores);
			else
				best = argmax_scores(scores, nr_cands);
#endif
#ifdef CONFIG_ARM64
			if (static_branch_likely(&cambyses_has_neon))
				best = cambyses_simd_argmax_neon(scores);
			else
				best = argmax_scores(scores, nr_cands);
#endif

			if (scores[best] == S16_MIN)
				break;

			scores[best] = S16_MIN;
			selected[nr_selected++] = best;

			consume_imbalance_cambyses(cands[best].p, env);

#ifdef CONFIG_PREEMPTION
			if (env->idle == CPU_NEWLY_IDLE)
				break;
#endif
		}
		cambyses_simd_end();

		/* Phase 2b: detach selected candidates (outside FPU context) */
		for (j = 0; j < nr_selected; j++) {
			p = cands[selected[j]].p;
			detach_task(p, env);
			list_add(&p->se.group_node, &env->tasks);
			detached++;
		}
	} else
#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
	{
		while (env->imbalance > 0) {
			int best = argmax_scores(scores, nr_cands);

			if (scores[best] == S16_MIN)
				break;

			p = cands[best].p;
			scores[best] = S16_MIN;

			consume_imbalance_cambyses(p, env);
			detach_task(p, env);
			list_add(&p->se.group_node, &env->tasks);
			detached++;

#ifdef CONFIG_PREEMPTION
			if (env->idle == CPU_NEWLY_IDLE)
				break;
#endif
		}
	}

	/*
	 * Return non-detached candidates to cfs_tasks.
	 * These were removed in Phase 1 but not selected in Phase 3
	 * (imbalance exhausted or preemption break).
	 */
	list_splice(&cand_tasks, tasks);

	schedstat_add(env->sd->lb_gained[env->idle], detached);
	return detached;
}

/*
 * detach_one_task_cambyses — scored selection for Push path (active balancing)
 *
 * Replaces the FIFO "first migratable task" policy in detach_one_task().
 * Scans all migratable tasks on src_rq and selects the one with the
 * highest migration score.
 *
 * Push moves exactly 1 task, so no sort is needed — simple max scan.
 * The stop_machine context is already heavy, so scoring overhead is negligible.
 */
static struct task_struct *detach_one_task_cambyses(struct lb_env *env)
{
	struct task_struct *p, *best = NULL;
	s16 best_score = S16_MIN;

	lockdep_assert_rq_held(env->src_rq);

	list_for_each_entry_reverse(p,
			&env->src_rq->cfs_tasks, se.group_node) {
		s16 score;

		/* Prefetch next candidate (one-ahead in reverse) */
		if (p->se.group_node.prev != &env->src_rq->cfs_tasks)
			prefetch_migration_task(
				list_prev_entry(p, se.group_node));

		if (!can_migrate_task(p, env))
			continue;

		score = score_task_cambyses(p, env);
		if (score > best_score) {
			best_score = score;
			best = p;
		}
	}

	if (!best)
		return NULL;

	detach_task(best, env);
	schedstat_inc(env->sd->lb_gained[env->idle]);
	return best;
}

#ifdef CONFIG_SCHED_CAMBYSES_SIMD
#include <asm/cpufeature.h>

static int __init cambyses_init(void)
{
	const char *simd_name;

#ifdef CONFIG_X86_64
	if (boot_cpu_has(X86_FEATURE_AVX2)) {
		static_branch_enable(&cambyses_has_avx2);
		simd_name = "AVX2";
	} else if (boot_cpu_has(X86_FEATURE_SSSE3)) {
		static_branch_enable(&cambyses_has_ssse3);
		simd_name = "SSSE3";
	} else {
		simd_name = "none";
	}
#endif

#ifdef CONFIG_ARM64
	if (system_supports_fpsimd()) {
		static_branch_enable(&cambyses_has_neon);
		simd_name = "NEON";
	} else {
		simd_name = "none";
	}
#endif

	pr_info("%s v%s by %s [SIMD argmax: %s]\n",
		CAMBYSES_PROGNAME, CAMBYSES_VERSION,
		CAMBYSES_AUTHOR, simd_name);

	return 0;
}
#else /* !CONFIG_SCHED_CAMBYSES_SIMD */
static int __init cambyses_init(void)
{
	pr_info("%s v%s by %s\n",
		CAMBYSES_PROGNAME, CAMBYSES_VERSION,
		CAMBYSES_AUTHOR);

	return 0;
}
#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
late_initcall(cambyses_init);

/* ======== sysctl interface ======== */

#ifdef CONFIG_SYSCTL
static int sched_cambyses_handler(struct ctl_table *table,
					  int write, void *buffer,
					  size_t *lenp, loff_t *ppos)
{
	static u8 sched_cambyses_val;
	struct ctl_table tmp = {
		.data	= &sched_cambyses_val,
		.maxlen	= sizeof(u8),
		.mode	= table->mode,
		.extra1	= SYSCTL_ZERO,
		.extra2	= SYSCTL_ONE,
	};
	int ret;

	if (!write)
		sched_cambyses_val = static_key_enabled(&sched_cambyses);

	ret = proc_dou8vec_minmax(&tmp, write, buffer, lenp, ppos);
	if (ret || !write)
		return ret;

	if (sched_cambyses_val)
		static_branch_enable(&sched_cambyses);
	else
		static_branch_disable(&sched_cambyses);

	return 0;
}

static struct ctl_table sched_cambyses_sysctls[] = {
	{
		.procname	= "sched_cambyses",
		.data		= NULL, /* handled by custom handler */
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= sched_cambyses_handler,
	},
	{
		.procname	= "sched_cambyses_w0",
		.data		= &sysctl_cambyses_w0,
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= proc_dou8vec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_THREE,
	},
	{
		.procname	= "sched_cambyses_w1",
		.data		= &sysctl_cambyses_w1,
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= proc_dou8vec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_THREE,
	},
	{
		.procname	= "sched_cambyses_w2",
		.data		= &sysctl_cambyses_w2,
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= proc_dou8vec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_THREE, /* vol switch ratio weight */
	},
	{
		.procname	= "sched_cambyses_w3",
		.data		= &sysctl_cambyses_w3,
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= proc_dou8vec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_THREE,
	},
};

static int __init sched_cambyses_sysctl_init(void)
{
	register_sysctl_init("kernel", sched_cambyses_sysctls);
	return 0;
}
late_initcall(sched_cambyses_sysctl_init);
#endif /* CONFIG_SYSCTL */
