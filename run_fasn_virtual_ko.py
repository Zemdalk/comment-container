from __future__ import annotations
import gzip, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.io import mmread
from scipy.stats import mannwhitneyu
from scTenifold import virtual_knockout

SEED=20260907
rng=np.random.default_rng(SEED)
RAW=Path('raw_fasn_knk'); OUT=Path('results_fasn_knk'); RAW.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
SAMPLES=[
 ('GSM7173751','NP_SP21_015','H1','GSM7173nnn'),
 ('GSM7173752','NP_SP21_018','H2','GSM7173nnn'),
 ('GSM7173753','NP_SP22_001','H3','GSM7173nnn'),
]
MODULES={
 'fatty_acid_synthesis':['FASN','ACACA','ACACB','ACLY','SREBF1','SCD','ELOVL6','FADS1','FADS2','ME1','G6PD','PGD','HMGCR','HMGCS1'],
 'lysosome':['LAMP1','LAMP2','CTSB','CTSD','CTSL','ATP6AP1','ATP6AP2','ATP6V0A1','ATP6V0D1','ATP6V1A','ATP6V1B2','ATP6V1C1','TFEB','TFE3','MCOLN1'],
 'autophagy':['ATG5','ATG7','BECN1','MAP1LC3B','SQSTM1','ULK1','ULK2','ATG12','ATG16L1','WIPI1','WIPI2'],
 'ECM_homeostasis':['ACAN','COL2A1','COL1A1','COL3A1','COL6A1','COL6A2','COMP','DCN','LUM','SOX9','MMP3','MMP13','ADAMTS4','ADAMTS5','TIMP1','TIMP2'],
 'senescence':['CDKN1A','CDKN2A','TP53','GLB1','SERPINE1','IL6','CXCL8','GDF15'],
 'ferroptosis':['ACSL4','GPX4','SLC7A11','TFRC','FTH1','FTL','NCOA4','ALOX15','LPCAT3','SAT1','HMOX1'],
 'oxidative_stress':['NFE2L2','KEAP1','HMOX1','NQO1','SOD1','SOD2','CAT','GPX1','GCLC','GCLM']
}

def dl(url,dst):
    if dst.exists() and dst.stat().st_size>0:return
    print('Downloading',url,flush=True)
    with requests.get(url,stream=True,timeout=(30,300)) as r:
        r.raise_for_status()
        with open(dst,'wb') as f:
            for c in r.iter_content(1024*1024):
                if c:f.write(c)

def load_one(rec):
    gsm,name,donor,bucket=rec; d=RAW/gsm; d.mkdir(exist_ok=True)
    files={}
    for suffix in ['barcodes.tsv.gz','features.tsv.gz','matrix.mtx.gz']:
        p=d/suffix; dl(f'https://ftp.ncbi.nlm.nih.gov/geo/samples/{bucket}/{gsm}/suppl/{gsm}_{name}_{suffix}',p); files[suffix]=p
    bar=pd.read_csv(files['barcodes.tsv.gz'],sep='\t',header=None,compression='gzip')[0].astype(str).to_numpy()
    feat=pd.read_csv(files['features.tsv.gz'],sep='\t',header=None,compression='gzip')
    genes=feat.iloc[:,1].astype(str).to_numpy(); M=mmread(str(files['matrix.mtx.gz'])).tocsr().astype(np.float32)
    keepg=np.asarray((M>0).sum(axis=1)).ravel()>=3; M=M[keepg]; genes=genes[keepg]
    # collapse duplicate symbols by first occurrence to keep deterministic input
    _,idx=np.unique(genes,return_index=True); idx=np.sort(idx); M=M[idx]; genes=genes[idx]
    nfeat=np.asarray((M>0).sum(axis=0)).ravel(); ncount=np.asarray(M.sum(axis=0)).ravel(); mt=np.char.startswith(genes.astype(str),'MT-')
    mtc=np.asarray(M[mt].sum(axis=0)).ravel() if mt.any() else np.zeros(M.shape[1]); pct=100*mtc/np.maximum(ncount,1)
    keep=(nfeat>=200)&(nfeat<=10000)&(ncount>=500)&(ncount<=100000)&(pct<15)
    M=M[:,keep]; bar=bar[keep]
    print(gsm,donor,'QC cells',M.shape[1],flush=True)
    return genes,M,bar

loaded=[load_one(x) for x in SAMPLES]
common=set(loaded[0][0]);
for g,_,_ in loaded[1:]: common &= set(g)
common=np.array(sorted(common),dtype=str); cidx={g:i for i,g in enumerate(common)}
blocks=[]; cellnames=[]; donor_ranges=[]
start=0
for rec,(genes,M,bar) in zip(SAMPLES,loaded):
    mp={g:i for i,g in enumerate(genes)}; rows=[mp[g] for g in common]; X=M[rows]
    target=min(500,X.shape[1]); sel=rng.choice(X.shape[1],target,replace=False); X=X[:,sel]
    blocks.append(X); cellnames.extend([f'{rec[2]}_{i}' for i in range(target)]); donor_ranges.append((rec[2],start,start+target)); start+=target
B=sparse.hstack(blocks,format='csr')
# predeclared broad network universe from healthy WT, then force only the perturbation target FASN in.
det=np.asarray((B>0).mean(axis=1)).ravel(); nuisance=np.array([bool(re.match(r'^(MT-|RPL|RPS|HBA|HBB)',g)) for g in common]); cand=np.flatnonzero((det>=0.05)&(~nuisance))
lib=np.asarray(B.sum(axis=0)).ravel(); L=B[cand]@sparse.diags(1e6/np.maximum(lib,1)); L=L.tocsr(); L.data=np.log1p(L.data)
mu=np.asarray(L.mean(axis=1)).ravel(); mu2=np.asarray(L.multiply(L).mean(axis=1)).ravel(); vr=np.maximum(mu2-mu*mu,0)
order=np.argsort(-vr,kind='stable')[:1500]; sel=cand[order]
if 'FASN' not in common: raise RuntimeError('FASN absent from common healthy-gene set')
fi=cidx['FASN']; forced=False
if fi not in sel:
    sel=np.concatenate([sel,[fi]]); forced=True
sel_genes=common[sel]; X=B[sel].toarray().astype(np.float32)
print('network genes',len(sel_genes),'FASN forced',forced,'FASN detection',float(det[fi]),flush=True)
pd.DataFrame(X,index=sel_genes,columns=cellnames).to_csv(OUT/'knk_input_counts.csv.gz',compression='gzip')
pd.DataFrame({'gene':sel_genes,'forced_target':[g=='FASN' and forced for g in sel_genes]}).to_csv(OUT/'network_universe.csv',index=False)

data=pd.DataFrame(X,index=sel_genes,columns=cellnames)
result=virtual_knockout(
    data,
    ko_genes=['FASN'],
    qc_kws={'min_lib_size':1,'min_percent':0.001,'min_exp_avg':0,'min_exp_sum':0},
    network_kws={'n_nets':5,'n_samp_cells':500},
    ko_method='default',
    backend='serial',
    n_jobs=1,
)
result=result.sort_values(['adjusted p-value','p-value','Distance'],ascending=[True,True,False]).reset_index(drop=True)
result['rank']=np.arange(1,len(result)+1)
result.to_csv(OUT/'FASN_virtual_KO_differential_regulation.csv',index=False)
print('TOP PERTURBED GENES',flush=True); print(result.head(30).to_string(index=False),flush=True)
# module-level enrichment: compare KO distance ranks of present module genes vs all others.
gene_col='Gene'; dist_col='Distance'; rank_map=dict(zip(result[gene_col].astype(str),result['rank'])); dist_map=dict(zip(result[gene_col].astype(str),result[dist_col].astype(float)))
module_rows=[]
for name,geneset in MODULES.items():
    present=[g for g in geneset if g in rank_map]
    other=[g for g in result[gene_col].astype(str) if g not in set(present)]
    if present:
        ranks=np.array([rank_map[g] for g in present],float); d=np.array([dist_map[g] for g in present],float); od=np.array([dist_map[g] for g in other],float)
        p=float(mannwhitneyu(d,od,alternative='greater').pvalue) if len(other)>0 else np.nan
        module_rows.append({'module':name,'n_present':len(present),'genes_present':';'.join(present),'median_rank':float(np.median(ranks)),'best_rank':int(np.min(ranks)),'mean_distance':float(np.mean(d)),'MWU_distance_enrichment_p':p})
    else: module_rows.append({'module':name,'n_present':0,'genes_present':'','median_rank':np.nan,'best_rank':np.nan,'mean_distance':np.nan,'MWU_distance_enrichment_p':np.nan})
mod=pd.DataFrame(module_rows).sort_values('MWU_distance_enrichment_p',na_position='last'); mod.to_csv(OUT/'module_perturbation_summary.csv',index=False)
# targeted genes for story inspection
TARGETS=['FASN','SREBF1','ACACA','SCD','LAMP1','LAMP2','CTSB','CTSD','ATP6AP2','ATP6V1A','ATP6V1B2','TFEB','TFE3','ACAN','COL2A1','SOX9','MMP3','MMP13','ADAMTS4','ADAMTS5','GPX4','ACSL4','CDKN1A','CDKN2A']
target=result[result[gene_col].isin(TARGETS)].copy(); target.to_csv(OUT/'target_gene_inspection.csv',index=False)
summary={'seed':SEED,'WT_source':'Healthy H1-H3 donor-balanced 500 cells each','n_cells':int(X.shape[1]),'n_network_genes':int(X.shape[0]),'FASN_forced_into_network':forced,'FASN_detection_fraction':float(det[fi]),'n_significant_FDR_0.05':int((result['adjusted p-value']<0.05).sum()),'top20_genes':result.head(20)[gene_col].astype(str).tolist(),'modules':mod.to_dict(orient='records'),'target_genes_found':target[[gene_col,'rank','Distance','Z','p-value','adjusted p-value']].to_dict(orient='records')}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('===== FASN VIRTUAL KO SUMMARY =====',flush=True); print(json.dumps(summary,indent=2),flush=True)
