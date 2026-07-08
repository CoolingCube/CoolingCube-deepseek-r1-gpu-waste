"""
ServeGen 2026 -- Structural GPU Waste Analysis
Alibaba Cloud Production Workloads | NSDI 2026
DeepSeek-R1, Qwen2.5-VL, General 14B/72B/310B

Run in Colab:
  exec(open('/content/drive/MyDrive/servegen_waste_2026.py').read())
"""
import subprocess, sys, os
import numpy as np
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

REPO_DIR = '/content/ServeGen'
if not os.path.exists(REPO_DIR):
    print("Cloning ServeGen...")
    subprocess.check_call(['git','clone','--quiet',
        'https://github.com/alibaba/ServeGen.git', REPO_DIR])
    subprocess.check_call([sys.executable,'-m','pip','install','-q','-e',REPO_DIR])
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)

from servegen import Category, ClientPool
from servegen.construct import generate_workload

TS   = list(range(0, 86400, 600))   # 144 timestamps, 600s apart
WIN  = 60;  PEAK = 0.95;  BUF = 1.30;  COST = 2.50
TPUT = {'m-large':300,'m-mid':800,'m-small':2000,'deepseek-r1':200,'mm-image':600}
MODELS = [
    (Category.LANGUAGE,   'm-large',     'General 310B'),
    (Category.LANGUAGE,   'm-mid',       'General 72B'),
    (Category.LANGUAGE,   'm-small',     'General 14B'),
    (Category.REASON,     'deepseek-r1', 'DeepSeek-R1 671B'),
    (Category.MULTIMODAL, 'mm-image',    'Qwen2.5-VL Image'),
]

def rate_fn(base=25.0):
    rf = {}
    for t in TS:
        h = (t/3600) % 24
        rf[t] = base * max(0.1, 0.3 + 0.7*max(0, np.sin(np.pi*(h-4)/14)))
    return rf

def fast_extract(reqs):
    """Extract fields directly from dataclasses -- avoids slow pandas conversion."""
    ts  = np.array([r.timestamp for r in reqs], dtype=np.float64)
    inp = np.array([r.data.get('input_tokens', 0) for r in reqs], dtype=np.int32)
    out = np.array([r.data.get('output_tokens', 0) for r in reqs], dtype=np.int32)
    return ts, inp, out, inp + out

def analyze(ts, total_tokens, model):
    tput = TPUT.get(model, 1000)
    t0=ts.min(); t1=ts.max()
    bins=np.arange(t0, t1+WIN, WIN)
    idx=np.digitize(ts, bins) - 1
    tpw=np.bincount(idx, weights=total_tokens, minlength=len(bins)-1).astype(float)
    gd=tpw/(tput*WIN)
    avg=gd.mean(); peak=np.quantile(gd, PEAK); idle=(gd<0.001).mean()
    burst=peak/max(avg,1e-9); util=avg/max(peak,1e-9)*100
    ded=max(1,np.ceil(peak)); pool=max(1,np.ceil(avg*BUF))
    waste=ded-pool; pct=waste/ded*100; usd=waste*COST*24*30
    return dict(avg=avg,peak=peak,idle=idle,burst=burst,util=util,
                ded=ded,pool=pool,waste=waste,pct=pct,usd=usd,gd=gd)

print("="*60)
print("ServeGen 2026 -- Structural GPU Waste")
print("Alibaba Cloud | NSDI 2026 | 3.54B real production requests")
print("="*60)
print("\nGenerating workloads (streaming extraction -- fast)...")

results = {}
stats   = {}

for cat, model, label in MODELS:
    try:
        pool = ClientPool(cat, model)
        reqs = generate_workload(pool, rate_fn(), duration=86400)
        ts, inp, out, total = fast_extract(reqs)
        r = analyze(ts, total, model)
        results[model] = r
        stats[model] = dict(label=label, n=len(ts),
                            avg_in=inp.mean(), avg_out=out.mean())
        print(f"  {model}: {len(ts):,} reqs | "
              f"{inp.mean():.0f}in→{out.mean():.0f}out tok | "
              f"burst={r['burst']:.1f}x waste={r['pct']:.0f}%")
    except Exception as e:
        print(f"  {model}: SKIP ({e})")

if not results:
    raise SystemExit("No models loaded")

# Combined pooling
comb_waste=comb_pct=comb_usd=tot_ded=comb_pool=0
if len(results) >= 2:
    max_len = max(len(r['gd']) for r in results.values())
    comb = sum(
        np.pad(r['gd'], (0, max_len-len(r['gd'])))
        for r in results.values()
    )
    tot_ded = sum(r['ded'] for r in results.values())
    comb_pool = max(1, np.ceil(comb.mean()*BUF))
    comb_waste = tot_ded - comb_pool
    comb_pct = comb_waste/tot_ded*100
    comb_usd = comb_waste*COST*24*30

print(f"\n{'='*60}")
print("RESULTS")
print("="*60)
for model, r in results.items():
    s = stats[model]
    print(f"\n{s['label']}:")
    print(f"  {s['n']:,} reqs | avg {s['avg_in']:.0f}→{s['avg_out']:.0f} tokens")
    print(f"  Burstiness:   {r['burst']:.1f}x")
    print(f"  Utilization:  {r['util']:.1f}%")
    print(f"  Idle windows: {r['idle']*100:.1f}%")
    print(f"  Waste:        {r['waste']:.0f} GPU ({r['pct']:.1f}%) = ${r['usd']:,.0f}/month")
if comb_waste > 0:
    print(f"\nCombined pooling: {tot_ded:.0f} → {comb_pool:.0f} GPUs | "
          f"{comb_pct:.0f}% waste | ${comb_usd:,.0f}/month")

# Key finding
if 'deepseek-r1' in stats and 'm-small' in stats:
    dr=stats['deepseek-r1']; ms=stats['m-small']
    ratio=dr['avg_out']/max(ms['avg_out'],1)
    print(f"\nKEY: DeepSeek-R1 outputs {dr['avg_out']:.0f} tok vs {ms['avg_out']:.0f} tok (General 14B)")
    print(f"     Reasoning uses {ratio:.0f}x more GPU time per request")

# Chart
colors={'m-large':'#1f77b4','m-mid':'#ff7f0e','m-small':'#2ca02c',
        'deepseek-r1':'#d62728','mm-image':'#9467bd'}
fig,axes=plt.subplots(2,2,figsize=(14,10))
fig.suptitle('ServeGen 2026 -- Structural GPU Waste\n'
    'Alibaba Cloud Production | NSDI 2026 | '
    'DeepSeek-R1 + Qwen2.5-VL + General LLMs',
    fontsize=12,fontweight='bold')

best=max(results.items(),key=lambda x:x[1]['burst'])
ax=axes[0,0]
t=np.arange(len(best[1]['gd']))*WIN/3600
ax.fill_between(t,0,best[1]['gd'],alpha=0.4,color=colors.get(best[0],'steelblue'))
ax.axhline(best[1]['avg'],color='green',ls='--',lw=1.5,
           label=f"Avg {best[1]['avg']:.2f} GPUs")
ax.axhline(best[1]['peak'],color='red',ls='--',lw=1.5,
           label=f"P95 {best[1]['peak']:.2f} GPUs")
ax.set_title(f"{stats[best[0]]['label']} — {best[1]['burst']:.1f}x burst, {best[1]['util']:.0f}% util")
ax.set_xlabel('Hour'); ax.set_ylabel('GPU demand'); ax.legend(fontsize=8)

ax=axes[0,1]
labs=[stats[m]['label'].replace(' ','\n') for m in results]
pcts=[r['pct'] for r in results.values()]
bars=ax.bar(range(len(labs)),pcts,
            color=[colors.get(m,'gray') for m in results],edgecolor='black',lw=0.5)
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,fontsize=7)
ax.set_ylim(0,80); ax.set_title('Waste by Model'); ax.set_ylabel('%')
for bar,p in zip(bars,pcts):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
            f'{p:.0f}%',ha='center',fontsize=9,fontweight='bold')

ax=axes[1,0]
for m,c in [('deepseek-r1','#d62728'),('m-small','#2ca02c')]:
    if m in stats:
        out_data=np.clip(
            np.array([r.data.get('output_tokens',0)
                      for r in generate_workload(
                          ClientPool(Category.REASON if m=='deepseek-r1' else Category.LANGUAGE, m),
                          {t:3.0 for t in TS}, duration=3600)]),
            0, 8000)
        ax.hist(out_data,bins=60,alpha=0.5,color=c,label=stats[m]['label'],density=True)
ax.set_title('Output Tokens: Reasoning vs Language')
ax.legend(fontsize=8); ax.set_xlabel('Output tokens'); ax.set_ylabel('Density')

ax=axes[1,1]
labs2=[stats[m]['label'] for m in results]
usds=[r['usd'] for r in results.values()]
if comb_waste>0: labs2.append('Combined\n(pooled)'); usds.append(comb_usd)
cols2=[colors.get(m,'gray') for m in results]+(['darkred'] if comb_waste>0 else [])
bars2=ax.bar(range(len(labs2)),usds,color=cols2,edgecolor='black',lw=0.5)
ax.set_xticks(range(len(labs2)))
ax.set_xticklabels(labs2,fontsize=7,rotation=15,ha='right')
ax.set_title('Monthly Waste Cost (USD)'); ax.set_ylabel('USD/month')
for bar,u in zip(bars2,usds):
    if u>0:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+50,
                f'${u:,.0f}',ha='center',fontsize=8,fontweight='bold')

plt.tight_layout()
plt.savefig('/content/servegen_waste_2026.png',dpi=150,bbox_inches='tight')
try:
    plt.savefig('/content/drive/MyDrive/servegen_waste_2026.png',dpi=150,bbox_inches='tight')
    print("\nSaved to Drive")
except Exception: pass
plt.show()

print(f"\nData: github.com/alibaba/ServeGen (NSDI 2026, Apache-2.0)")
print(f"Code: github.com/CoolingCube/structural-gpu-waste")
