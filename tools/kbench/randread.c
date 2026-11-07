// randread: O_DIRECT random reads of a file, size S x T threads: what expert-sized reads cost on the TV's eMMC.
// usage: randread <file> <KB> <threads> <MB total>
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
static int fd; static size_t S, fsz, per; static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void *run(void *a){ unsigned seed=(unsigned)(uintptr_t)a*2654435761u+12345; char *b=mmap(0,S,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 for(size_t d=0;d<per;d+=S){ uint64_t r=((uint64_t)rand_r(&seed)<<31)|rand_r(&seed); off_t o=(off_t)(r%(fsz-S))&~(off_t)4095; if(pread(fd,b,S,o)<=0){perror("pread");break;} } return 0;}
int main(int c,char**v){fd=open(v[1],O_RDONLY|O_DIRECT);if(fd<0){perror("open");return 1;}struct stat st;fstat(fd,&st);fsz=st.st_size;S=(size_t)atoi(v[2])<<10;int T=atoi(v[3]);per=((size_t)atoi(v[4])<<20)/T;
 pthread_t p[16];double t=now();for(long i=0;i<T;i++)pthread_create(&p[i],0,run,(void*)(i+1));for(int i=0;i<T;i++)pthread_join(p[i],0);t=now()-t;
 printf("%5d KB x %d thr: %6.1f MB/s, %6.1f us per read\n",atoi(v[2]),T,atoi(v[4])/t,t*1e6/((double)atoi(v[4])*1024/atoi(v[2])));return 0;}
