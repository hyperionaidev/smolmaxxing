// Can this box hand out transparent huge pages, and do they speed up weight streaming?
// usage: thpcheck <MB per buffer> [files to evict from the page cache...]
// Evicts the given files (posix_fadvise DONTNEED), then faults a 2 MiB aligned MADV_HUGEPAGE buffer and a plain 4 KiB one
// and reads each with 3 threads, sequentially, as a repacked matmul does. Reports huge pages obtained and GB/s.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>
#include <arm_neon.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static long field(const char*f,const char*k){FILE*p=fopen(f,"r");char l[256];long v=-1;size_t n=strlen(k);while(p&&fgets(l,256,p))if(!strncmp(l,k,n)){v=atol(l+n);break;}if(p)fclose(p);return v;}
static long vm(const char*k){FILE*p=fopen("/proc/vmstat","r");char a[128];long v;long r=-1;while(p&&fscanf(p,"%127s %ld",a,&v)==2)if(!strcmp(a,k)){r=v;break;}if(p)fclose(p);return r;}
struct job{char*p;size_t n;int rep;uint64_t sum;};
static void*rd(void*a){struct job*j=a;uint32x4_t s=vdupq_n_u32(0);for(int r=0;r<j->rep;r++)for(size_t i=0;i+64<=j->n;i+=64){uint8x16x4_t v=vld1q_u8_x4((uint8_t*)j->p+i);s=vaddq_u32(s,vdotq_u32(vdupq_n_u32(0),v.val[0],v.val[3]));}j->sum=vaddvq_u32(s);return 0;}
static double bench(char*p,size_t n){pthread_t t[3];struct job j[3];double b=now();size_t c=n/3&~(size_t)63;for(int i=0;i<3;i++){j[i]=(struct job){p+i*c,c,4,0};pthread_create(&t[i],0,rd,&j[i]);}for(int i=0;i<3;i++)pthread_join(t[i],0);return 4.0*c*3/(now()-b)/1e9;}
static char*alloc(size_t n,int huge){char*p;size_t a=2u<<20;void*q=mmap(0,n+a,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);p=(char*)(((size_t)q+a-1)&~(a-1));if(huge)madvise(p,n,MADV_HUGEPAGE);else madvise(p,n,MADV_NOHUGEPAGE);double b=now();long f0=vm("thp_fault_alloc"),f1=vm("thp_fault_fallback");for(size_t i=0;i<n;i+=4096)p[i]=(char)i;printf("%s: faulted %zu MB in %.2f s, thp_fault_alloc +%ld, fallback +%ld, process AnonHuge %ld kB\n",huge?"THP":"4K ",n>>20,now()-b,vm("thp_fault_alloc")-f0,vm("thp_fault_fallback")-f1,field("/proc/self/smaps_rollup","AnonHugePages:"));return p;}
int main(int c,char**v){size_t n=(size_t)atol(v[1])<<20;for(int i=2;i<c;i++){int f=open(v[i],O_RDONLY);if(f>=0){posix_fadvise(f,0,0,POSIX_FADV_DONTNEED);close(f);}}
printf("MemFree %ld kB, MemAvailable %ld kB, Cached %ld kB\n",field("/proc/meminfo","MemFree:"),field("/proc/meminfo","MemAvailable:"),field("/proc/meminfo","Cached:"));
char*a=alloc(n,1),*b=alloc(n,0);for(int r=0;r<3;r++){printf("round %d: THP-request buffer %.2f GB/s, 4K buffer %.2f GB/s\n",r,bench(a,n),bench(b,n));}return 0;}
