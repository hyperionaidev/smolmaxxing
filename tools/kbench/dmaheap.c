// dmaheap: can a shell process use the box's DMA heaps (CMA-backed) as extra, cached RAM?
// usage: dmaheap <heap-name> <chunk-MB> <max-MB> [hold-seconds]
// Allocates chunk by chunk until max or failure, touches every page, measures read/write bandwidth over the last chunk,
// prints /proc/meminfo deltas, holds, then frees. Stops early if MemAvailable < 200 MB.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <arm_neon.h>
struct dma_heap_allocation_data { uint64_t len; uint32_t fd; uint32_t fd_flags; uint64_t heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void mem(const char*tag,long*avail){FILE*f=fopen("/proc/meminfo","r");char l[128];long fr=0,av=0,cm=0,ca=0,an=0,sw=0;
 while(fgets(l,sizeof l,f)){sscanf(l,"MemFree: %ld",&fr);sscanf(l,"MemAvailable: %ld",&av);sscanf(l,"CmaFree: %ld",&cm);sscanf(l,"Cached: %ld",&ca);sscanf(l,"AnonPages: %ld",&an);sscanf(l,"SwapFree: %ld",&sw);}
 fclose(f);printf("%-14s free %5ld avail %5ld cmafree %5ld cached %5ld anon %5ld swapfree %5ld MB\n",tag,fr/1024,av/1024,cm/1024,ca/1024,an/1024,sw/1024);fflush(stdout);if(avail)*avail=av/1024;}
static void bw(uint8_t*p,size_t n){
 uint64x2_t acc=vdupq_n_u64(0);double best=0;
 for(int r=0;r<3;r++){double t=now();for(size_t i=0;i+64<=n;i+=64){acc=veorq_u64(acc,vreinterpretq_u64_u8(vld1q_u8(p+i)));acc=veorq_u64(acc,vreinterpretq_u64_u8(vld1q_u8(p+i+16)));acc=veorq_u64(acc,vreinterpretq_u64_u8(vld1q_u8(p+i+32)));acc=veorq_u64(acc,vreinterpretq_u64_u8(vld1q_u8(p+i+48)));}
 t=now()-t;if(n/t>best)best=n/t;}
 double t=now();memset(p,0x5a,n);t=now()-t;
 printf("  read %.2f GB/s (best of 3, x=%llx), memset %.2f GB/s over %zu MB\n",best/1e9,(unsigned long long)vgetq_lane_u64(acc,0),n/t/1e9,n>>20);fflush(stdout);}
int main(int c,char**v){
 if(c<4){puts("usage: dmaheap heap chunkMB maxMB [hold]");return 2;}
 char path[128];snprintf(path,sizeof path,"/dev/dma_heap/%s",v[1]);
 int h=open(path,O_RDONLY|O_CLOEXEC);if(h<0){perror(path);return 1;}
 size_t chunk=(size_t)atoi(v[2])<<20,total=0,maxb=(size_t)atoi(v[3])<<20;int hold=c>4?atoi(v[4]):3;
 long av;mem("before",&av);
 uint8_t*last=NULL;
 while(total<maxb){
  struct dma_heap_allocation_data d;memset(&d,0,sizeof d);d.len=chunk;d.fd_flags=O_RDWR|O_CLOEXEC;
  double t=now();
  if(ioctl(h,DMA_HEAP_IOCTL_ALLOC,&d)<0){perror("alloc");break;}
  double ta=now()-t;
  uint8_t*p=mmap(NULL,chunk,PROT_READ|PROT_WRITE,MAP_SHARED,d.fd,0);
  if(p==MAP_FAILED){perror("mmap");break;}
  memset(p,1,chunk);total+=chunk;last=p;
  char tag[32];snprintf(tag,sizeof tag,"%zu MB (%.0fms)",total>>20,ta*1e3);mem(tag,&av);
  if(av<200){puts("stop: MemAvailable < 200");break;}
 }
 if(last)bw(last,chunk);
 printf("holding %d s, total %zu MB\n",hold,total>>20);fflush(stdout);sleep(hold);mem("after hold",&av);
 return 0;}
