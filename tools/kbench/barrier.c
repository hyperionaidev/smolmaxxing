// barrier cost on the TV: ggml's spin barrier (seq_cst counter + passed counter), n threads, N rounds.
// usage: barrier <threads> <rounds> [work_ns]   (work_ns: busy work per round on thread 0 only, like a 1-row op)
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static atomic_int nb, np; static int NT, R; static long W;
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void barrier(void){
  int p = atomic_load_explicit(&np, memory_order_relaxed);
  if (atomic_fetch_add_explicit(&nb,1,memory_order_seq_cst) == NT-1){ atomic_store_explicit(&nb,0,memory_order_relaxed); atomic_fetch_add_explicit(&np,1,memory_order_seq_cst); return; }
  while (atomic_load_explicit(&np,memory_order_relaxed) == p) __asm__ volatile("yield");
  atomic_thread_fence(memory_order_seq_cst);
}
static void *run(void *a){ long id=(long)a; volatile long x=0; for(int r=0;r<R;r++){ if(id==0&&W){ double t=now(); while((now()-t)*1e9<W) x++; } barrier(); } return 0; }
int main(int c,char**v){ NT=atoi(v[1]); R=atoi(v[2]); W=c>3?atol(v[3]):0; pthread_t t[8];
  double t0=now(); for(long i=1;i<NT;i++) pthread_create(&t[i],0,run,(void*)i); run(0); for(int i=1;i<NT;i++) pthread_join(t[i],0);
  double dt=now()-t0; printf("threads %d rounds %d work %ld ns: %.2f us per round (%.2f us over the work)\n",NT,R,W,dt/R*1e6,dt/R*1e6-W/1e3); }
