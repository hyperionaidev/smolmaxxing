// Read bandwidth from a file mapping vs anonymous memory: membw file MB passes
#include <arm_neon.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
struct job{const uint8_t*p;size_t n;uint32_t r;};
static void*sum(void*v){struct job*j=v;uint32x4_t a=vdupq_n_u32(0),b=a,c=a,d=a;
 for(size_t i=0;i+64<=j->n;i+=64){a=vpadalq_u16(a,vpaddlq_u8(vld1q_u8(j->p+i)));b=vpadalq_u16(b,vpaddlq_u8(vld1q_u8(j->p+i+16)));
  c=vpadalq_u16(c,vpaddlq_u8(vld1q_u8(j->p+i+32)));d=vpadalq_u16(d,vpaddlq_u8(vld1q_u8(j->p+i+48)));}
 j->r=vaddvq_u32(vaddq_u32(vaddq_u32(a,b),vaddq_u32(c,d)));return 0;}
static double run(const uint8_t*p,size_t n,int passes){double best=1e9;for(int k=0;k<passes;k++){pthread_t t[2];struct job j[2];double t0=now();
 for(int i=0;i<2;i++){j[i]=(struct job){p+n/2*i,n/2,0};pthread_create(&t[i],0,sum,&j[i]);}for(int i=0;i<2;i++)pthread_join(t[i],0);
 double dt=now()-t0;if(dt<best)best=dt;}return n/best/1e9;}
int main(int c,char**v){size_t n=(size_t)atoi(v[2])<<20;int passes=atoi(v[3]);int fd=open(v[1],O_RDONLY);
 uint8_t*m=mmap(0,n,PROT_READ,MAP_SHARED,fd,0);if(m==MAP_FAILED){perror("mmap");return 1;}
 printf("mapped file:     %.2f GB/s\n",run(m,n,passes));
 uint8_t*a;posix_memalign((void**)&a,16384,n);memcpy(a,m,n);
 printf("anonymous copy:  %.2f GB/s\n",run(a,n,passes));
 printf("mapped again:    %.2f GB/s\n",run(m,n,passes));return 0;}
