// kbench_t3: which base-3 trit decode is fastest on the in-order Cortex-A55? Same 5-trit bytes and dot products in
// every variant, checked equal. usage: kbench_t3 [KB resident=32] [MB dram=256]
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int8_t act[256];
typedef int64_t (*fn)(const uint8_t*,size_t);
#define LOADA int8x16_t A[5]; for(int k=0;k<5;k++)A[k]=vld1q_s8(act+32*k);
#define FIN int64_t s=vaddvq_s32(c0)+(int64_t)vaddvq_s32(c1); return s;
#define MUL3(v) vmulq_u8(v,vdupq_n_u8(3))
#define T0(v) vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8((v),vshrq_n_u8((v),1)),6))
// generic loop body: unrolled for 4 lanes as in the kernel
#define LOOP(BODY) LOADA int32x4_t c0=vdupq_n_s32(0),c1=c0; for(size_t i=0;i+128<=n;i+=128){ \
  {uint8x16_t u0=vld1q_u8(p+i),u1=vld1q_u8(p+i+16);BODY(0)} {uint8x16_t u0=vld1q_u8(p+i+32),u1=vld1q_u8(p+i+48);BODY(1)} \
  {uint8x16_t u0=vld1q_u8(p+i+64),u1=vld1q_u8(p+i+80);BODY(2)} {uint8x16_t u0=vld1q_u8(p+i+96),u1=vld1q_u8(p+i+112);BODY(3)} } FIN
#define DOT2(ta,tb,k,L) c0=vdotq_laneq_s32(c0,ta,A[k],L);c1=vdotq_laneq_s32(c1,tb,A[k],L);
// V1 = V0 unrolled (the real kernel's shape)
#define B1(L) {uint8x16_t a=u0,b=u1;DOT2(T0(a),T0(b),0,L) a=vmulq_u8(u0,vdupq_n_u8(3));b=vmulq_u8(u1,vdupq_n_u8(3));DOT2(T0(a),T0(b),1,L) a=vmulq_u8(u0,vdupq_n_u8(9));b=vmulq_u8(u1,vdupq_n_u8(9));DOT2(T0(a),T0(b),2,L) a=vmulq_u8(u0,vdupq_n_u8(27));b=vmulq_u8(u1,vdupq_n_u8(27));DOT2(T0(a),T0(b),3,L) a=vmulq_u8(u0,vdupq_n_u8(81));b=vmulq_u8(u1,vdupq_n_u8(81));DOT2(T0(a),T0(b),4,L)}
static int64_t v1(const uint8_t*p,size_t n){LOOP(B1)}
// V2: chained v = v*3 by add/shift (no mul), same uhadd decode
#define X3(v) vaddq_u8(vshlq_n_u8(v,1),v)
#define B2(L) {uint8x16_t a=u0,b=u1;DOT2(T0(a),T0(b),0,L) a=X3(a);b=X3(b);DOT2(T0(a),T0(b),1,L) a=X3(a);b=X3(b);DOT2(T0(a),T0(b),2,L) a=X3(a);b=X3(b);DOT2(T0(a),T0(b),3,L) a=X3(a);b=X3(b);DOT2(T0(a),T0(b),4,L)}
static int64_t v2(const uint8_t*p,size_t n){LOOP(B2)}
// V3: compare decode: t = (v>=86)+(v>=171), masks are -1, so the dot is negated at the end
#define T3(v) vreinterpretq_s8_u8(vaddq_u8(vcgeq_u8((v),vdupq_n_u8(86)),vcgeq_u8((v),vdupq_n_u8(171))))
#define B3(L) {uint8x16_t a=u0,b=u1;DOT2(T3(a),T3(b),0,L) a=vmulq_u8(u0,vdupq_n_u8(3));b=vmulq_u8(u1,vdupq_n_u8(3));DOT2(T3(a),T3(b),1,L) a=vmulq_u8(u0,vdupq_n_u8(9));b=vmulq_u8(u1,vdupq_n_u8(9));DOT2(T3(a),T3(b),2,L) a=vmulq_u8(u0,vdupq_n_u8(27));b=vmulq_u8(u1,vdupq_n_u8(27));DOT2(T3(a),T3(b),3,L) a=vmulq_u8(u0,vdupq_n_u8(81));b=vmulq_u8(u1,vdupq_n_u8(81));DOT2(T3(a),T3(b),4,L)}
int64_t v3i(const uint8_t*p,size_t n){LOOP(B3)}
static int64_t v3(const uint8_t*p,size_t n){return -v3i(p,n);}
// V4: compare decode + chained add/shift multiply
#define B4(L) {uint8x16_t a=u0,b=u1;DOT2(T3(a),T3(b),0,L) a=X3(a);b=X3(b);DOT2(T3(a),T3(b),1,L) a=X3(a);b=X3(b);DOT2(T3(a),T3(b),2,L) a=X3(a);b=X3(b);DOT2(T3(a),T3(b),3,L) a=X3(a);b=X3(b);DOT2(T3(a),T3(b),4,L)}
int64_t v4i(const uint8_t*p,size_t n){LOOP(B4)}
static int64_t v4(const uint8_t*p,size_t n){return -v4i(p,n);}
// V5: separate decode from dots: all 10 decodes in vector code first, then the 10 dots (no result-to-use interlocks between them)
#define B5(L) {uint8x16_t a1=vmulq_u8(u0,vdupq_n_u8(3)),b1=vmulq_u8(u1,vdupq_n_u8(3)),a2=vmulq_u8(u0,vdupq_n_u8(9)),b2=vmulq_u8(u1,vdupq_n_u8(9)),a3=vmulq_u8(u0,vdupq_n_u8(27)),b3=vmulq_u8(u1,vdupq_n_u8(27)),a4=vmulq_u8(u0,vdupq_n_u8(81)),b4=vmulq_u8(u1,vdupq_n_u8(81)); \
 int8x16_t t0=T3(u0),s0=T3(u1),t1=T3(a1),s1=T3(b1),t2=T3(a2),s2=T3(b2),t3=T3(a3),s3=T3(b3),t4=T3(a4),s4=T3(b4); DOT2(t0,s0,0,L) DOT2(t1,s1,1,L) DOT2(t2,s2,2,L) DOT2(t3,s3,3,L) DOT2(t4,s4,4,L)}
int64_t v5i(const uint8_t*p,size_t n){LOOP(B5)}
static int64_t v5(const uint8_t*p,size_t n){return -v5i(p,n);}
struct job{fn f;const uint8_t*p;size_t n;int rep;int64_t r;};
static void*run(void*a){struct job*j=a;for(int i=0;i<j->rep;i++)j->r+=j->f(j->p,j->n);return 0;}
static double go(fn f,const uint8_t*p,size_t n,int th,int rep,int64_t*chk){pthread_t t[4];struct job j[4];size_t c=(n/th)&~(size_t)127;double b=now();for(int i=0;i<th;i++){j[i]=(struct job){f,p+i*c,c,rep,0};pthread_create(&t[i],0,run,&j[i]);}int64_t s=0;for(int i=0;i<th;i++){pthread_join(t[i],0);s+=j[i].r;}*chk=s;double e=now()-b;return (double)c*th*rep*5/e/1e9;}
int main(int ac,char**av){size_t kb=ac>1?atol(av[1]):32;size_t mb=ac>2?atol(av[2]):256;size_t nr=kb<<10,nd=mb<<20;uint8_t*r,*d;posix_memalign((void**)&r,64,nr*4);posix_memalign((void**)&d,64,nd);
 for(int i=0;i<256;i++)act[i]=(int8_t)(rand()%255-127);for(size_t i=0;i<nr*4;i++)r[i]=rand()%243;for(size_t i=0;i<nd;i++)d[i]=(uint8_t)((i*2654435761u>>13)%243);
 struct{const char*n;fn f;}V[]={{"v1 mul+uhadd (current)",v1},{"v2 chain+uhadd",v2},{"v3 mul+cmp",v3},{"v4 chain+cmp",v4},{"v5 mul+cmp, decode first",v5}};
 for(int k=0;k<3;k++)for(unsigned i=0;i<sizeof(V)/sizeof(V[0]);i++){int64_t a,b,c;double s1=go(V[i].f,r,nr,1,20000,&a),s3=go(V[i].f,r,nr*3,3,20000,&b),sd=go(V[i].f,d,nd,3,3,&c);
  printf("%-26s resident 1T %6.2f 3T %6.2f  DRAM 3T %6.2f G weights/s  chk %lld\n",V[i].n,s1,s3,sd,(long long)(b/20000));fflush(stdout);}
 return 0;}
