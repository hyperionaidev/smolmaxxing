// Sequential uncached read speed from flash, 1 and 2 threads, for several block sizes.
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
struct job{const char*p;off_t a,b;size_t bs;};
static void*rd(void*v){struct job*j=v;int fd=open(j->p,O_RDONLY);fcntl(fd,F_NOCACHE,1);void*b;posix_memalign(&b,16384,j->bs);
 for(off_t o=j->a;o<j->b;o+=j->bs){if(pread(fd,b,j->bs,o)<=0)break;}free(b);close(fd);return 0;}
int main(int c,char**v){struct stat st;stat(v[1],&st);off_t mb256=(off_t)256<<20;
 size_t bss[]={64<<10,256<<10,1<<20,4<<20};off_t base=0;
 for(int t=1;t<=2;t++)for(int i=0;i<4;i++){pthread_t th[2];struct job j[2];double t0=now();
  if(base+mb256>st.st_size)base=0;
  for(int k=0;k<t;k++){j[k]=(struct job){v[1],base+mb256*k/t,base+mb256*(k+1)/t,bss[i]};pthread_create(&th[k],0,rd,&j[k]);}
  for(int k=0;k<t;k++)pthread_join(th[k],0);
  printf("threads %d block %4zu KB: %.0f MB/s\n",t,bss[i]>>10,256/(now()-t0));base+=mb256/4;}
 return 0;}
