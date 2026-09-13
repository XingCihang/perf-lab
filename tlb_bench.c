/* Lab 1:实测大页对TLB的影响
   用指针追逐(pointer chasing)对比普通4K页与2MB大页的随机访问延迟

   为什么使用指针追逐而不是普通循环:
   顺序或者是有规律的访问会被硬件预取器提前加载，页表会被提交walk，测出来的不是真实的TLB miss
   代价，指针追逐里下一个地址来自当前读到的内容，CPU无法预测，预取失效，暴露真实延迟

 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <sys/mman.h>
#include <stddef.h>      /* ptrdiff_t */

#define HUGE_2MB (2UL * 1024 * 1024)
#define STRIDE   4096                   // 每次条一个4KB页，最大化TLB压力

static double
now_sec(void){

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void*
alloc_mem(size_t size,int huge){
    
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if(huge)
        flags |= MAP_HUGETLB;   // 这个macro只在Linux有定义
                                // 2.6.32+，在 <sys/mman.h>，需要 _GNU_SOURCE
    
    void *p = mmap(NULL,size,PROT_READ|PROT_WRITE,flags,-1,0);
    if(p == MAP_FAILED){
        fprintf(stderr," mmap(%zu MB,%s) 失败: %s \n",
            size >> 20,huge ? "大页":"普通页",strerror(errno));
        if(huge)
            fprintf(stderr," 提示:检查大页是否够用 -- grep HugePages_Free /proc/meminfo\n"
                "需要提供%zu 个2MB大页\n",size / HUGE_2MB);
        
        return NULL;
    }

#ifdef MADV_NOHUGEPAGE
    /* ★ 对照组显式拒绝透明大页(THP)。
     * THP=always 的机器上，这块匿名映射会被内核自动提升成 2MB 大页，
     * 于是"4KB 组"其实也是大页 → 两组都是大页 → 加速比≈1。
     * 本机 THP=madvise，这行是防御性的：换台机器跑也能保证对照组纯净。 */
    if (!huge)
        madvise(p, size, MADV_NOHUGEPAGE);
#endif

    memset(p,0,size);
    return p;
}

/*
 * TODO 1: 构造随机置换的指针环
 *
 * 目标: 把 mem 切成 n = size/STRIDE 个槽位，每个槽位开头存放"下一个槽位的地址"，
 *       并且这些槽位的连接顺序是随机的，最终首尾相接形成一个环。
 *
 *   buf[perm[0]] -> buf[perm[1]] -> ... -> buf[perm[n-1]] -> buf[perm[0]]
 *
 * 步骤提示:
 *   1. 生成 0..n-1 的数组，用 Fisher-Yates 洗牌打乱得到随机置换 perm[]
 *   2. 对每个 i，令 *(void **)(base + perm[i]*STRIDE) = base + perm[(i+1)%n]*STRIDE
 *   3. 返回链的起点（比如 base + perm[0]*STRIDE）
 *
 * 注意: perm 数组本身用 malloc，别占用被测内存
 */
static void*
build_chain(void *mem, size_t size){

    // 切槽:将内存切成n段，每段只用8字节（存一个指针）
    size_t n = size / STRIDE;

    size_t* perm = malloc(n * sizeof(size_t));  // 分配一块n*size_t大小的内存
    if(!perm){
        perror("malloc perm");
        exit(1);
    }

    for(size_t i = 0;i < n;i++) //进行每个槽位赋值
        perm[i] = i;
    
    srand(12345);

    for(size_t i = n - 1;i > 0;i--){

        // 在i的位置之前随机挑一个位置
        size_t j = (size_t)((double)rand() / ((double)RAND_MAX + 1.0 ) * (double)(i + 1));
        
        // 交换位置
        size_t t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }

    char* base = (char*)mem;
    // i = n - 1 也就是最后一个元素的时候指向开头，成环，可以无限跑
    for(size_t i = 0;i < n;i++){        // i:顺序访问的编号
        // perm[i] 第i步访问的槽位编号,如果p[0] = 4,那么要实际访问4号槽位
        // base + perm[i] * STRIDE 计算得到这个槽位的实际地址
        void** slot = (void**)(base + perm[i] * STRIDE); 
        // 在slot这个位置写入的是下一步的槽位地址
        *slot = base + perm[(i + 1) % n] * STRIDE;
    }

    void* start = base + perm[0] * STRIDE;
    free(perm);
    return start;
}

/*
 * TODO 2: 沿指针链走 iters 步，返回平均每次访问的纳秒数
 *
 * 步骤提示:
 *   1. warm up: 先完整走一遍（或走 n 步），把数据带进 cache，
 *      这样测到的差异主要来自 TLB 而不是 cache miss
 *   2. 记录起始时间 -> 循环 iters 次 p = *(void **)p -> 记录结束时间
 *   3. 防止编译器把循环优化掉: 把最终的 p 以某种方式"用掉"
 *      (比如累加到一个 volatile 变量，或者返回前打印其低位)
 *   4. 返回 (结束-开始) * 1e9 / iters
 */
static double 
run_chase(void *start, size_t iters){

    void* p = start;

    for(size_t i = 0;i < iters / 10;i++)
        p = *(void**)p;
    
    double t0 = now_sec();
    for(size_t i = 0;i < iters;i++)
        p = *(void **)p;
    double t1 = now_sec();

    static volatile uintptr_t sink;
    sink = (uintptr_t)p;

    return (t1 - t0) * 1e9 / (double)iters;
}

/* ★ 自检：证明指针环真的随机、且覆盖全部槽位。
 * 抓两类会让整个实验失效、但不会报错的错误：
 *   ① 置换退化（洗牌写错 → 恒定步长 → 预取器生效 → 加速比≈1）
 *   ② 链断成多个小环（工作集缩水 → TLB 压力归零） */
static int
verify_chain(void *start, size_t n)
{
    /* 走 n 步必须回到起点 —— 单环且覆盖全部槽位 */
    void *q = start;
    for (size_t k = 0; k < n; k++)
        q = *(void **)q;
    if (q != start) {
        fprintf(stderr, "  \u2717 自检失败: 走 %zu 步没回到起点，链不是单环\n", n);
        return 0;
    }

    /* 前 16 步的步长不能恒定 —— 恒定说明置换退化成了轮转 */
    q = start;
    ptrdiff_t d0 = (char *)*(void **)q - (char *)q;
    int constant = 1;
    for (int k = 0; k < 16 && constant; k++) {
        void *nx = *(void **)q;
        if ((char *)nx - (char *)q != d0) constant = 0;
        q = nx;
    }
    if (constant) {
        fprintf(stderr, "  \u2717 自检失败: 前16步步长恒定(%td)，置换失效，预取器会生效\n", d0);
        return 0;
    }

    printf("  自检: 单环覆盖 %zu 槽位 OK, 步长随机 OK\n", n);
    return 1;
}

static void
bench_one(const char* label,size_t size,int huge,double* out_ns){
    
    printf("\n[%s]\n",label);

    void* mem = alloc_mem(size,huge);
    if(!mem){
        *out_ns = -1.0;
        return;
    }

    size_t n = size / STRIDE;
    printf(" 内存%zu MB,槽位 %zu 个,页数 %zu\n",
        size >> 20,n,huge ? size / HUGE_2MB : size / 4096);
    
    void* start = build_chain(mem,size);
    if (!verify_chain(start, n)) {          /* ★ 自检不过就别测了，数字没意义 */
        munmap(mem, size);
        *out_ns = -1.0;
        return;
    }
    double ns = run_chase(start,20UL * 1000 * 1000);

    printf(" 平均时延: %.2f ns/access\n",ns);
    *out_ns = ns;

    munmap(mem,size);
}

int main(int argc,char** argv){

    size_t mb = (argc > 1) ? strtoul(argv[1],NULL,10) : 256;
    size_t size = mb * 1024 * 1024;

    size = (size + HUGE_2MB - 1) / HUGE_2MB * HUGE_2MB;
	printf("=== TLB 影响测试 ===\n");
	printf("工作集: %zu MB   步长: %lu B\n", size >> 20, STRIDE);
	printf("对比: %zu 个 4KB 页  vs  %zu 个 2MB 页\n",
	       size / 4096, size / HUGE_2MB);
	printf("(典型 L2 TLB 约 1536 项，据此判断哪种会大量 miss)\n");

    /* 第二个参数: 0=只测4KB, 1=只测大页, 缺省=两个都测。给 perf 分开计数用 */
    int mode = (argc > 2) ? atoi(argv[2]) : -1;

    double ns_normal = 0,ns_huge = 0;
    if (mode != 1) bench_one("普通4KB页",size,0,&ns_normal);
    if (mode != 0) bench_one("2MB 大页",size,1,&ns_huge);

	if (ns_normal > 0 && ns_huge > 0) {
		printf("\n=== 结果 ===\n");
		printf("  4KB 页: %6.2f ns/access\n", ns_normal);
		printf("  2MB 页: %6.2f ns/access\n", ns_huge);
		printf("  加速比: %.2fx\n", ns_normal / ns_huge);
		printf("\n把这个数字记进 环境记录.md\n");
	}    



    return 0;
}