// Memory read-bandwidth probe for the TV box: how fast can 1..4 A55 cores stream DRAM?
// This is the ceiling for bandwidth-bound LLM token generation (every weight read once per token).
// usage: membw [MB=256] [reps=6]
#include <arm_neon.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint8_t *buf; static size_t n; static int nth;
static volatile uint32_t sink[16];
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void *rd(void *a){ long id=(long)a; size_t ch=n/nth, s=id*ch, e=s+ch;
  uint32x4_t acc0=vdupq_n_u32(0),acc1=acc0,acc2=acc0,acc3=acc0;
  for(size_t i=s;i<e;i+=64){ const uint32_t *p=(const uint32_t*)(buf+i);
    acc0=veorq_u32(acc0,vld1q_u32(p)); acc1=veorq_u32(acc1,vld1q_u32(p+4));
    acc2=veorq_u32(acc2,vld1q_u32(p+8)); acc3=veorq_u32(acc3,vld1q_u32(p+12)); }
  acc0=veorq_u32(veorq_u32(acc0,acc1),veorq_u32(acc2,acc3)); sink[id]=vgetq_lane_u32(acc0,0); return 0; }
int main(int c,char**v){ size_t mb=c>1?atoi(v[1]):256; int reps=c>2?atoi(v[2]):6; n=mb<<20;
  buf=aligned_alloc(64,n); memset(buf,1,n);
  for(nth=1;nth<=4;nth++){ double best=0;
    for(int r=0;r<reps;r++){ pthread_t t[4]; double t0=now();
      for(long i=0;i<nth;i++) pthread_create(&t[i],0,rd,(void*)i);
      for(int i=0;i<nth;i++) pthread_join(t[i],0);
      double gbs=n/(now()-t0)/1e9; if(gbs>best) best=gbs; }
    printf("threads=%d  read %.2f GB/s (best of %d, %zu MB)\n",nth,best,reps,mb); }
  return 0; }
