// Random uncached reads from flash, as the expert pager does them: rrbench file from_byte block_KB threads reads_per_thread
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
struct job{const char*p;off_t lo,hi;size_t bs;int n;unsigned seed;};
static void*rd(void*v){struct job*j=v;int fd=open(j->p,O_RDONLY);fcntl(fd,F_NOCACHE,1);void*b;posix_memalign(&b,16384,j->bs);
 for(int i=0;i<j->n;i++){off_t o=j->lo+(off_t)((double)rand_r(&j->seed)/RAND_MAX*(double)(j->hi-j->lo-(off_t)j->bs));o&=~(off_t)31;
  size_t left=j->bs;char*d=b;while(left){ssize_t r=pread(fd,d,left,o);if(r<=0)break;d+=r;left-=r;o+=r;}}
 free(b);close(fd);return 0;}
int main(int c,char**v){if(c<6){fprintf(stderr,"usage: %s file from_byte block_KB threads reads\n",v[0]);return 1;}
 struct stat st;stat(v[1],&st);off_t lo=atoll(v[2]);size_t bs=(size_t)atoi(v[3])<<10;int t=atoi(v[4]),n=atoi(v[5]);
 pthread_t th[16];struct job j[16];double t0=now();
 for(int k=0;k<t;k++){j[k]=(struct job){v[1],lo,st.st_size,bs,n,(unsigned)(k*7919+1)};pthread_create(&th[k],0,rd,&j[k]);}
 for(int k=0;k<t;k++)pthread_join(th[k],0);double dt=now()-t0;
 printf("block %4zu KB, %d threads: %.0f MB/s, %.2f ms per read\n",bs>>10,t,(double)bs*t*n/dt/1e6,dt*1e3*t/(t*n));return 0;}
