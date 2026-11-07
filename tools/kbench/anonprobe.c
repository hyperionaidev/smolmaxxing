// anonprobe: grow anonymous memory in steps and log where it comes from (MemFree vs CmaFree vs cache vs swap).
// usage: anonprobe <step-MB> <max-MB> <stop-avail-MB> [hold-s]   MEM_REPEAT=1 keeps re-touching the region while holding.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static long mem(const char*tag){FILE*f=fopen("/proc/meminfo","r");char l[128];long fr=0,av=0,cm=0,ca=0,an=0,sw=0,sl=0,ac=0,ic=0,sc=0;
 while(fgets(l,sizeof l,f)){sscanf(l,"MemFree: %ld",&fr);sscanf(l,"MemAvailable: %ld",&av);sscanf(l,"CmaFree: %ld",&cm);sscanf(l,"Cached: %ld",&ca);sscanf(l,"AnonPages: %ld",&an);sscanf(l,"SwapFree: %ld",&sw);sscanf(l,"Slab: %ld",&sl);sscanf(l,"Active(file): %ld",&ac);sscanf(l,"Inactive(file): %ld",&ic);sscanf(l,"SwapCached: %ld",&sc);}
 fclose(f);printf("%-12s free %5ld cmafree %5ld (noncma %5ld) avail %5ld cached %5ld actf %5ld inactf %5ld anon %5ld swapfree %5ld swapcached %4ld slab %4ld\n",tag,fr/1024,cm/1024,(fr-cm)/1024,av/1024,ca/1024,ac/1024,ic/1024,an/1024,sw/1024,sc/1024,sl/1024);fflush(stdout);return av/1024;}
int main(int c,char**v){size_t step=(size_t)atoi(v[1])<<20,maxb=(size_t)atoi(v[2])<<20,tot=0;long stop=atol(v[3]);int hold=c>4?atoi(v[4]):5;
 uint8_t*p=mmap(NULL,maxb,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(p==MAP_FAILED){perror("mmap");return 1;}
 mem("start");
 while(tot<maxb){double t=now();for(size_t i=0;i<step;i+=4096){p[tot+i]=(uint8_t)(i*2654435761u>>13)+1;/* non-zero, poorly compressible-ish */}
  tot+=step;t=now()-t;char tag[32];snprintf(tag,sizeof tag,"%zu %.2fs",tot>>20,t);long a=mem(tag);if(a<stop){puts("stop");break;}}
 for(int s=0;s<hold;s++){sleep(1);double t=now();size_t sum=0;for(size_t i=0;i<tot;i+=4096)sum+=p[i];t=now()-t;char tag[32];snprintf(tag,sizeof tag,"hold%d %.2fs",s,t);mem(tag);if(sum==1)puts("");}
 return 0;}
