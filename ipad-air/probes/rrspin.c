// Do flash reads keep their speed while both cores compute? rrspin file from_byte block_KB readers spinners reader_prio(0|1) seconds
// Readers do uncached random reads; spinners run a NEON FMA loop (like matmul threads). Prints read MB/s and spinner Gflop/s.
#include <arm_neon.h>
#include <fcntl.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static atomic_int stop; static atomic_long bytes, flops; static const char*path; static off_t lo,hi; static size_t bs; static int prio;
static void*rd(void*v){unsigned seed=(unsigned)(long)v*7919+1; if(prio)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
 int fd=open(path,O_RDONLY);fcntl(fd,F_NOCACHE,1);void*b;posix_memalign(&b,16384,bs);
 while(!stop){off_t o=lo+(off_t)((double)rand_r(&seed)/RAND_MAX*(double)(hi-lo-(off_t)bs));o&=~(off_t)31;
  size_t left=bs;char*d=b;while(left){ssize_t r=pread(fd,d,left,o);if(r<=0)break;d+=r;left-=r;o+=r;} bytes+=bs;}
 return 0;}
static void*spin(void*v){float32x4_t a=vdupq_n_f32(1.0001f),b=vdupq_n_f32(0.9999f),c0=vdupq_n_f32(0),c1=c0,c2=c0,c3=c0;long n=0;
 while(!stop){for(int i=0;i<10000;i++){c0=vfmaq_f32(c0,a,b);c1=vfmaq_f32(c1,a,b);c2=vfmaq_f32(c2,a,b);c3=vfmaq_f32(c3,a,b);}n+=10000*32;}
 flops+=n; volatile float s=vaddvq_f32(vaddq_f32(vaddq_f32(c0,c1),vaddq_f32(c2,c3)));(void)s;return 0;}
int main(int c,char**v){if(c<8){fprintf(stderr,"usage\n");return 1;}struct stat st;path=v[1];stat(path,&st);lo=atoll(v[2]);hi=st.st_size;
 bs=(size_t)atoi(v[3])<<10;int nr=atoi(v[4]),ns=atoi(v[5]);prio=atoi(v[6]);double secs=atof(v[7]);pthread_t th[32];int n=0;
 for(int i=0;i<ns;i++)pthread_create(&th[n++],0,spin,0);for(int i=0;i<nr;i++)pthread_create(&th[n++],0,rd,(void*)(long)i);
 double t0=now();usleep((useconds_t)(secs*1e6));stop=1;for(int i=0;i<n;i++)pthread_join(th[i],0);double dt=now()-t0;
 printf("block %zu KB readers %d spinners %d prio %d: read %.0f MB/s, spin %.2f Gflop/s\n",bs>>10,nr,ns,prio,bytes/dt/1e6,flops/dt/1e9);return 0;}
