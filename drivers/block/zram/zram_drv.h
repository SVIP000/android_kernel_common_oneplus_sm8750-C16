/*
 * Compressed RAM block device
 *
 * Copyright (C) 2008, 2009, 2010  Nitin Gupta
 *               2012, 2013 Minchan Kim
 *
 * This code is released using a dual license strategy: BSD/GPL
 * You can choose the licence that better fits your requirements.
 *
 * Released under the terms of 3-clause BSD License
 * Released under the terms of GNU General Public License Version 2.0
 *
 */

#ifndef _ZRAM_DRV_H_
#define _ZRAM_DRV_H_

#include <linux/rwsem.h>
#include <linux/zsmalloc.h>
#include <linux/crypto.h>
#include <linux/list_lru.h>
#include <linux/percpu_counter.h>
#include <linux/xarray.h>

#include "zcomp.h"

#define SECTORS_PER_PAGE_SHIFT	(PAGE_SHIFT - SECTOR_SHIFT)
#define SECTORS_PER_PAGE	(1 << SECTORS_PER_PAGE_SHIFT)
#define ZRAM_LOGICAL_BLOCK_SHIFT 12
#define ZRAM_LOGICAL_BLOCK_SIZE	(1 << ZRAM_LOGICAL_BLOCK_SHIFT)
#define ZRAM_SECTOR_PER_LOGICAL_BLOCK	\
	(1 << (ZRAM_LOGICAL_BLOCK_SHIFT - SECTOR_SHIFT))


/*
 * ZRAM is mainly used for memory efficiency so we want to keep memory
 * footprint small and thus squeeze size and zram pageflags into a flags
 * member. The lower ZRAM_FLAG_SHIFT bits is for object size (excluding
 * header), which cannot be larger than PAGE_SIZE (requiring PAGE_SHIFT
 * bits), the higher bits are for zram_pageflags.
 *
 * We use BUILD_BUG_ON() to make sure that zram pageflags don't overflow.
 */
#define ZRAM_FLAG_SHIFT (PAGE_SHIFT + 1)

/* Flags for zram pages (table[page_no].flags) */
enum zram_pageflags {
	/* zram slot is locked */
	ZRAM_LOCK = ZRAM_FLAG_SHIFT,
	ZRAM_SAME,	/* Page consists the same element */
	ZRAM_WB,	/* page is stored on backing_device */
	ZRAM_PP_SLOT,	/* Selected for post-processing */
	ZRAM_HUGE,	/* Incompressible page */
	ZRAM_IDLE,	/* not accessed page since last idle marking */
	ZRAM_INCOMPRESSIBLE, /* none of the algorithms could compress it */
	ZRAM_PAGE_ANON,	/* page came from anonymous memory */
	ZRAM_PAGE_FILE,	/* page came from file-backed memory */
	ZRAM_PAGE_DIRTY,	/* file-backed page was dirty/writeback */
	ZRAM_WB_SECOND_CHANCE, /* one-time escape before writeback */

	ZRAM_REFERENCED, /* Page was referenced since last shrinker scan */
	ZRAM_ACTIVE, /* Page is in active list (percpu_pagevec or active_list) */
	ZRAM_TEMP_0, /* temperature bit0: read-frequency tier */
	ZRAM_TEMP_1, /* temperature bit1: read-frequency tier */

	__NR_ZRAM_PAGEFLAGS,
};

/*-- Data structures */

/* Allocated for each disk page */
struct zram_table_entry {
	unsigned long handle;
	unsigned long flags;
#ifdef CONFIG_ZRAM_TRACK_ENTRY_ACTIME
	ktime_t ac_time;
#endif
#ifdef	CONFIG_ZRAM_WRITEBACK
	struct list_head lru;
	u16 wb_ext_len;
	u16 wb_ext_off;
#endif
};

#ifdef CONFIG_ZRAM_WRITEBACK
#define BATCH_SIZE 32
#define WINDOW_RADIUS 4
#define ZRAM_WINDOW_CLAIMED_SIZE (WINDOW_RADIUS * 2 + 1)
#define MIN_AGGREGATE 2
#define ZRAM_PAGEVEC_SIZE 128
struct zram_pagevec {
	spinlock_t lock;  /* 使用标准自旋锁以支持跨 CPU drain */
	unsigned long indices[ZRAM_PAGEVEC_SIZE];
	int nr;
};

struct zram_shrink_work {
    struct zram *zram;
    struct zram_pp_ctl *ctl;              /* 写回控制器 */
    int nr_candidates;                    /* 当前收集数量 */
	unsigned long candidates[BATCH_SIZE]; /* 候选页面索引数组 */
	unsigned long window_claimed[ZRAM_WINDOW_CLAIMED_SIZE];
};
#endif

#define ZRAM_WB_READ_POLICY_STRICT	0
#define ZRAM_WB_READ_POLICY_RELAXED	1
#define ZRAM_WB_READ_POLICY_ADAPTIVE	2

#define ZRAM_WB_FRAG_MODE_OFF	0
#define ZRAM_WB_FRAG_MODE_AUTO	1
#define ZRAM_WB_FRAG_MODE_ON	2

#define ZRAM_READ_BATCH_MAX	16

struct zram_stats {
	struct percpu_counter compr_data_size;	/* compressed size of pages stored */
	atomic64_t failed_reads;	/* can happen when memory is too low */
	atomic64_t failed_writes;	/* can happen when memory is too low */
	struct percpu_counter notify_free;	/* no. of swap slot free notifications */
	struct percpu_counter same_pages;		/* no. of same element filled pages */
	struct percpu_counter huge_pages;		/* no. of huge pages */
	struct percpu_counter huge_pages_since;	/* no. of huge pages since zram set up */
	struct percpu_counter pages_stored;	/* no. of pages currently stored */
	atomic_long_t max_used_pages;	/* no. of maximum pages stored */
	atomic64_t writestall;		/* no. of write slow paths */
	atomic64_t miss_free;		/* no. of missed free */
#ifdef	CONFIG_ZRAM_WRITEBACK
	struct percpu_counter bd_count;		/* no. of pages in backing device */
	struct percpu_counter bd_reads;		/* no. of reads from backing device */
	struct percpu_counter bd_writes;		/* no. of writes from backing device */
	atomic64_t wb_pages_skipped;	/* no. of pages skipped by writeback filters */
	atomic64_t wb_read_batch_pages;	/* total pages served via wb read batches */
	atomic64_t wb_read_batch_bios;	/* total bios submitted for wb read batches */
	atomic64_t wb_read_batch_fallbacks;	/* wb read batches that degraded to 1 page */
#endif
};

struct zram {
	struct zram_table_entry *table;
	struct zs_pool *mem_pool;
	struct zcomp *comp;
	struct gendisk *disk;
	struct rw_semaphore init_lock;
	unsigned long limit_pages;

	struct zram_stats stats;
	u64 disksize;
	const char *comp_alg;
	bool claim;
	struct bio_set zram_bio_set;
	mempool_t *io_page_pool;
#ifdef CONFIG_ZRAM_WRITEBACK
	struct file *backing_dev;
	spinlock_t wb_limit_lock;
	bool wb_limit_enable;
	u64 bd_wb_limit;
	struct block_device *bdev;
	unsigned long *bitmap;
	unsigned long nr_pages;
	spinlock_t bitmap_lock;
	unsigned long bitmap_last_free_hint;
	struct shrinker *zram_shrinker;
	struct list_lru zram_list_lru;
	mempool_t *wb_page_pool;
	unsigned long shrinker_active_start;
	atomic_t shrinker_in_active_period;
	struct zram_pagevec __percpu *active_pagevecs;
	struct list_head active_list;
	spinlock_t active_list_lock;
	atomic_long_t active_pages;
	struct xarray wb_extent_xa;
	u8 wb_read_policy;
	u8 wb_read_gap_pages;
	u8 wb_frag_mode;
	u8 wb_frag_reserved;
	u32 wb_read_batch_ewma;
	u32 wb_read_fallback_ewma;
#endif
#ifdef CONFIG_ZRAM_MEMORY_TRACKING
	struct dentry *debugfs_dir;
#endif
	atomic_t pp_in_progress;
#ifdef CONFIG_ZRAM_AUTO_SIZE
	unsigned int historical_mem_pressure;
	unsigned int historical_zram_pressure;
	spinlock_t pressure_lock; 
#endif
};

void zram_slot_lock(struct zram *zram, u32 index);
void zram_slot_unlock(struct zram *zram, u32 index);
void zram_set_handle(struct zram *zram, u32 index, unsigned long handle);
bool zram_test_flag(struct zram *zram, u32 index, enum zram_pageflags flag);
void zram_set_flag(struct zram *zram, u32 index, enum zram_pageflags flag);
void zram_clear_flag(struct zram *zram, u32 index, enum zram_pageflags flag);
void zram_free_page(struct zram *zram, size_t index);

#ifdef CONFIG_ZRAM_WRITEBACK
struct zram_pp_slot {
	unsigned long		index;
	struct list_head	entry;
};

/*
 * A post-processing bucket is, essentially, a size class, this defines
 * the range (in bytes) of pp-slots sizes in particular bucket.
 */
#define PP_BUCKET_SIZE_RANGE	64
#define NUM_PP_BUCKETS		((PAGE_SIZE / PP_BUCKET_SIZE_RANGE) + 1)

struct zram_pp_ctl {
	struct list_head	pp_buckets[NUM_PP_BUCKETS];
	struct completion	all_done;
	atomic_t		num_pp_slots;
	unsigned long		deadline_jiffies;  /* 时间限制截止时间 */
};

void free_pp_slot(struct zram *zram, struct zram_pp_slot *pps);
void zram_wb_extent_init(struct zram *zram);
void zram_wb_extent_destroy(struct zram *zram);
void zram_wb_extent_record_run(struct zram *zram,
			      unsigned long index_start,
			      unsigned long blk_start,
			      unsigned int nr_pages,
			      gfp_t gfp);
bool zram_wb_extent_lookup(struct zram *zram,
			 unsigned long index,
			 unsigned long *blk,
			 unsigned int *max_pages);
#else
static inline void zram_wb_extent_init(struct zram *zram) {}
static inline void zram_wb_extent_destroy(struct zram *zram) {}
static inline void zram_wb_extent_record_run(struct zram *zram,
			      unsigned long index_start,
			      unsigned long blk_start,
			      unsigned int nr_pages,
			      gfp_t gfp) {}
static inline bool zram_wb_extent_lookup(struct zram *zram,
			 unsigned long index,
			 unsigned long *blk,
			 unsigned int *max_pages)
{
	return false;
}
#endif

#endif
