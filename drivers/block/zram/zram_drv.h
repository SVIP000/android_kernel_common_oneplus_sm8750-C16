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

	ZRAM_REFERENCED, /* Page was referenced since last shrinker scan */
	ZRAM_ACTIVE, /* Page is in active list (percpu_pagevec or active_list) */

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
#endif
};

#ifdef CONFIG_ZRAM_WRITEBACK
#define BATCH_SIZE 64
#define WINDOW_RADIUS 8
#define MIN_AGGREGATE 4
#define ZRAM_PAGEVEC_SIZE 128
struct zram_pagevec {
	spinlock_t lock;  /* 使用标准自旋锁以支持跨 CPU drain */
	unsigned long indices[ZRAM_PAGEVEC_SIZE];
	int nr;
};

struct zram_shrink_work {
    struct zram *zram;
    unsigned long candidates[BATCH_SIZE]; /* 候选页面索引数组 */
    int nr_candidates;                    /* 当前收集数量 */
    struct zram_pp_ctl *ctl;              /* 写回控制器 */
};
#endif

/* Hot page cache: avoids repeated decompression for frequently accessed pages */
#define ZRAM_HC_SHIFT		7
#define ZRAM_HC_SIZE		(1 << ZRAM_HC_SHIFT)
#define ZRAM_HC_MASK		(ZRAM_HC_SIZE - 1)
#define ZRAM_READ_BATCH_MAX	8

struct zram_hc_entry {
	spinlock_t lock;
	u32 index;
	struct page *page;
	bool referenced;
};

struct zram_hot_cache {
	struct zram_hc_entry entries[ZRAM_HC_SIZE];
	atomic_long_t hits;
	atomic_long_t misses;
};

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
	struct percpu_counter written_back_pages;
	atomic64_t reject_reclaim_fail;
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
	struct zram_hot_cache hot_cache;
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
#endif

#endif