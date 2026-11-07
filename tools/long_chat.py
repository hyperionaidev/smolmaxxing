import json,urllib.request,time
import os
U=os.environ.get("URL","http://127.0.0.1:8090")+"/v1/chat/completions"
body={"messages":[{"role":"user","content":"Write a long, detailed essay about the history of computing, from the abacus to modern AI. Cover every decade."}],"max_tokens":1500,"temperature":float(os.environ.get("TEMP","0.7")),"top_k":int(os.environ.get("TOPK","40")),"top_p":float(os.environ.get("TOPP","0.95")),"seed":int(os.environ.get("SEED","42")),"repeat_penalty":float(os.environ.get("RP","1.0")),"repeat_last_n":128,"stream":True,"stream_options":{"include_usage":True}}
r=urllib.request.urlopen(urllib.request.Request(U,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=1200)
t0=time.time(); n=0; last=t0; lastn=0; txt=''
for line in r:
    if not line.startswith(b"data: ") or b"[DONE]" in line: continue
    d=json.loads(line[6:])
    for c in d.get("choices",[]):
        x=c.get("delta",{}).get("content")
        if x: n+=1; txt+=x
    if time.time()-last>=30:
        if os.environ.get("VERBOSE"): print(f"t+{time.time()-t0:5.0f}s tokens {n:5d}  last-30s {(n-lastn)/(time.time()-last):.2f} tok/s",flush=True)
        last=time.time(); lastn=n
    if 'timings' in d: print('server timings',{k:round(v,2) for k,v in d['timings'].items() if k in('predicted_per_second','prompt_per_second','predicted_n')})
w=txt.split(); seen=set(); rep=run=best=0
for j in range(len(w)-5):
    g=tuple(w[j:j+6])
    if g in seen: rep+=1; run+=1; best=max(best,run)
    else: run=0
    seen.add(g)
print('words',len(w),'repeated 6-grams %.1f %%, longest repeated run %d words'%(100*rep/max(1,len(w)-5),best))
if os.environ.get('VERBOSE'): print('tail:',txt[-260:])
if os.environ.get('TXT'): open(os.environ['TXT'],'a').write('=== '+os.environ.get('LABEL','')+' seed '+os.environ.get('SEED','42')+'\n'+txt+'\n\n')
