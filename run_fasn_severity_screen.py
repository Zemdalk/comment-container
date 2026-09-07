from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse, stats
from scipy.io import mmread
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

SEED = 20260907
rng = np.random.default_rng(SEED)
ROOT = Path('.')
RAW = ROOT / 'raw_10x'
OUT = ROOT / 'results_severity'
RAW.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SAMPLES = [
    ('GSM7173751','NP_SP21_015','Healthy','H1','Thompson II',2.0,'GSM7173nnn'),
    ('GSM7173752','NP_SP21_018','Healthy','H2','Thompson II',2.0,'GSM7173nnn'),
    ('GSM7173753','NP_SP22_001','Healthy','H3','Thompson II',2.0,'GSM7173nnn'),
    ('GSM7235341','NP_SP21_007','Disease','D1','Thompson II-III',2.5,'GSM7235nnn'),
    ('GSM7235342','NP_SP21_011','Disease','D1','Thompson III',3.0,'GSM7235nnn'),
    ('GSM7235343','NP_SP21_013','Disease','D2','Thompson III',3.0,'GSM7235nnn'),
    ('GSM7235344','NP_SP21_014','Disease','D2','Thompson II-III',2.5,'GSM7235nnn'),
    ('GSM7235345','NP_SP21_016','Disease','D1','Thompson III-IV',3.5,'GSM7235nnn'),
    ('GSM7235346','NP_SP21_017','Disease','D2','Thompson III-IV',3.5,'GSM7235nnn'),
    ('GSM7235347','NP_SP22_002','Disease','D3','Thompson III-IV',3.5,'GSM7235nnn'),
    ('GSM7235348','NP_SP22_003','Disease','D4','Thompson III-IV',3.5,'GSM7235nnn'),
]

# Predeclared canonical fatty-acid synthesis / lipogenesis program.
# It is used only for module-level analysis; the genome-wide severity screen is independent of this list.
LIPOGENESIS = [
    'ACLY','ACACA','ACACB','FASN','SCD','SCD5','FADS1','FADS2','ELOVL1','ELOVL2','ELOVL3','ELOVL4','ELOVL5','ELOVL6','ELOVL7',
    'SREBF1','SREBF2','MLXIPL','INSIG1','INSIG2','SCAP','GPAM','GPAT3','GPAT4','AGPAT1','AGPAT2','AGPAT3','AGPAT4','DGAT1','DGAT2',
    'ACSL1','ACSL3','ACSL4','ACSL5','ACSL6','ME1','G6PD','PGD','IDH1','HMGCS1','HMGCR','MVK','MVD','FDPS','SQLE'
]


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, float)
    prev = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        val = p[order[i]] * n / rank
        prev = min(prev, val)
        q[order[i]] = min(prev, 1.0)
    return q


def url_for(sample, sample_name, bucket, suffix):
    return f'https://ftp.ncbi.nlm.nih.gov/geo/samples/{bucket}/{sample}/suppl/{sample}_{sample_name}_{suffix}'


def download(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return
    print('Downloading', url, flush=True)
    with requests.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + '.part')
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dst)


def read_counts(rec):
    sample, sample_name, condition, donor, grade, sev, bucket = rec
    d = RAW / sample
    d.mkdir(exist_ok=True)
    bf, ff, mf = d/'barcodes.tsv.gz', d/'features.tsv.gz', d/'matrix.mtx.gz'
    download(url_for(sample, sample_name, bucket, 'barcodes.tsv.gz'), bf)
    download(url_for(sample, sample_name, bucket, 'features.tsv.gz'), ff)
    download(url_for(sample, sample_name, bucket, 'matrix.mtx.gz'), mf)
    feat = pd.read_csv(ff, sep='\t', header=None, compression='gzip')
    genes = feat.iloc[:,1].astype(str).to_numpy()
    M = mmread(str(mf)).tocsr().astype(np.float64)  # genes x cells
    # Seurat-style min.cells=3/min.features=200, then same generic QC as pilot.
    keep_gene = np.asarray((M > 0).sum(axis=1)).ravel() >= 3
    M, genes = M[keep_gene], genes[keep_gene]
    nfeat = np.asarray((M > 0).sum(axis=0)).ravel()
    keep_cell = nfeat >= 200
    M = M[:,keep_cell]
    nfeat = nfeat[keep_cell]
    ncount = np.asarray(M.sum(axis=0)).ravel()
    mt = np.array([g.startswith('MT-') for g in genes])
    mtcount = np.asarray(M[mt].sum(axis=0)).ravel() if mt.any() else np.zeros(M.shape[1])
    pct_mt = 100 * mtcount / np.maximum(ncount,1)
    keep = (nfeat >= 200) & (nfeat <= 10000) & (ncount >= 500) & (ncount <= 100000) & (pct_mt < 15)
    M = M[:,keep]
    # Collapse duplicate symbols by summing counts.
    df = pd.DataFrame({'gene':genes})
    codes, uniq = pd.factorize(df['gene'], sort=False)
    if len(uniq) != len(genes):
        rows = sparse.coo_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes))).tocsr()
        M = rows @ M
        genes = np.asarray(uniq, dtype=str)
    sums = np.asarray(M.sum(axis=1)).ravel()
    detected_cells = np.asarray((M>0).sum(axis=1)).ravel()
    print(sample, grade, 'cells=', M.shape[1], 'genes=', M.shape[0], flush=True)
    return genes, sums, detected_cells, M.shape[1]

records = []
all_genes = set()
per_sample = {}
for rec in SAMPLES:
    genes, sums, det, ncells = read_counts(rec)
    per_sample[rec[0]] = (genes, sums, det, ncells)
    all_genes.update(genes.tolist())

all_genes = np.array(sorted(all_genes), dtype=str)
gidx = {g:i for i,g in enumerate(all_genes)}
counts = np.zeros((len(SAMPLES), len(all_genes)), dtype=float)
detect = np.zeros_like(counts)
meta = []
for si, rec in enumerate(SAMPLES):
    sample, sample_name, condition, donor, grade, sev, bucket = rec
    genes, sums, det, ncells = per_sample[sample]
    idx = np.fromiter((gidx[g] for g in genes), dtype=int, count=len(genes))
    counts[si,idx] = sums
    detect[si,idx] = det / max(ncells,1)
    meta.append({'sample':sample,'sample_name':sample_name,'condition':condition,'donor':donor,'grade':grade,'severity':sev,'n_cells':ncells})
meta = pd.DataFrame(meta)
meta.to_csv(OUT/'sample_metadata.csv', index=False)

# Pseudobulk logCPM; filter genes detected in >=5% cells in at least two samples and remove nuisance families.
lib = counts.sum(axis=1)
cpm = counts / np.maximum(lib[:,None],1) * 1e6
logcpm = np.log1p(cpm)
keep = (detect >= 0.05).sum(axis=0) >= 2
nuis = np.array([bool(re.match(r'^(MT-|RPL|RPS|HBA|HBB)', g)) for g in all_genes])
keep &= ~nuis
genes = all_genes[keep]
X = logcpm[:,keep]
sev = meta['severity'].to_numpy(float)

# Genome-wide monotonic disease-severity screen: Spearman across 11 independent samples.
rho = np.zeros(len(genes)); pval = np.ones(len(genes))
for j in range(len(genes)):
    r,p = stats.spearmanr(sev, X[:,j])
    rho[j] = 0 if not np.isfinite(r) else r
    pval[j] = 1 if not np.isfinite(p) else p
qval = bh_fdr(pval)
absrho = np.abs(rho)
order = np.argsort(-absrho, kind='stable')
rank = np.empty(len(genes), int); rank[order] = np.arange(1,len(genes)+1)
percentile = 100*(1-(rank-1)/max(len(genes)-1,1))
trend = pd.DataFrame({'gene':genes,'spearman_rho':rho,'p':pval,'BH_FDR':qval,'rank_abs_rho':rank,'percentile':percentile,'mean_logCPM':X.mean(axis=0)})
trend = trend.sort_values('rank_abs_rho')
trend.to_csv(OUT/'genomewide_severity_trend.csv', index=False)

# Also test non-monotonicity: quadratic vs intercept-only ANOVA-style F test.
quad_rows=[]
Z = np.column_stack([np.ones(len(sev)), sev, sev**2])
for j,g in enumerate(genes):
    y=X[:,j]
    beta=np.linalg.lstsq(Z,y,rcond=None)[0]
    resid=y-Z@beta
    sse1=float(resid@resid)
    sse0=float(((y-y.mean())@(y-y.mean())))
    df1=2; df2=len(y)-3
    if sse1 <= 0 or df2 <= 0:
        F=0; p=1
    else:
        F=max(((sse0-sse1)/df1)/(sse1/df2),0)
        p=float(stats.f.sf(F,df1,df2))
    quad_rows.append((g,F,p,beta[1],beta[2]))
quad=pd.DataFrame(quad_rows,columns=['gene','quadratic_F','quadratic_p','linear_coef','quadratic_coef'])
quad['quadratic_BH_FDR']=bh_fdr(quad['quadratic_p'].to_numpy())
quad=quad.sort_values('quadratic_p')
quad['quadratic_rank']=np.arange(1,len(quad)+1)
quad['quadratic_percentile']=100*(1-(quad['quadratic_rank']-1)/max(len(quad)-1,1))
quad.to_csv(OUT/'genomewide_quadratic_severity.csv', index=False)

# Lipogenesis module score per sample and module-member ranking by association with severity and module score.
module_present=[g for g in LIPOGENESIS if g in genes]
mi=[np.where(genes==g)[0][0] for g in module_present]
Xm=X[:,mi]
# z-score each gene across samples then average, so highly expressed genes do not dominate.
Xmz=(Xm-Xm.mean(axis=0))/np.where(Xm.std(axis=0,ddof=1)>0,Xm.std(axis=0,ddof=1),1)
module_score=Xmz.mean(axis=1)
meta['lipogenesis_module_score']=module_score
meta.to_csv(OUT/'sample_metadata_with_module_score.csv',index=False)
mod_rho,mod_p=stats.spearmanr(sev,module_score)
module_rows=[]
for k,g in enumerate(module_present):
    rsev,psev=stats.spearmanr(sev,Xm[:,k])
    rmod,pmod=stats.spearmanr(module_score,Xm[:,k])
    module_rows.append((g,rsev,psev,rmod,pmod,Xm[:,k].mean()))
module=pd.DataFrame(module_rows,columns=['gene','severity_rho','severity_p','module_rho','module_p','mean_logCPM'])
module['severity_abs_rho']=module['severity_rho'].abs()
module=module.sort_values(['severity_abs_rho','module_rho'],ascending=[False,False]).reset_index(drop=True)
module['module_rank']=np.arange(1,len(module)+1)
module['module_percentile']=100*(1-(module['module_rank']-1)/max(len(module)-1,1))
module.to_csv(OUT/'lipogenesis_member_ranking.csv',index=False)

# Exploratory sample-level ML only. Small N: report feature ranks, not predictive performance claims.
# Use top variable genes selected without labels; fixed at min(1000, available).
var=X.var(axis=0,ddof=1)
top=np.argsort(-var)[:min(1000,len(genes))]
ml_genes=genes[top]; Xml=X[:,top]
y=(meta['condition']=='Disease').astype(int).to_numpy()
Xs=StandardScaler().fit_transform(Xml)
logit=LogisticRegression(penalty='l1',solver='liblinear',C=0.2,random_state=SEED,max_iter=5000)
logit.fit(Xs,y)
coef=np.abs(logit.coef_[0])
rf=RandomForestClassifier(n_estimators=1000,max_features='sqrt',min_samples_leaf=2,random_state=SEED,class_weight='balanced')
rf.fit(Xml,y)
imp=rf.feature_importances_
# ranks within the predeclared top-variable ML universe
r1=pd.Series(-coef).rank(method='min').astype(int).to_numpy()
r2=pd.Series(-imp).rank(method='min').astype(int).to_numpy()
cons=(r1+r2)/2
ml=pd.DataFrame({'gene':ml_genes,'l1_abs_coef':coef,'l1_rank':r1,'rf_importance':imp,'rf_rank':r2,'consensus_mean_rank':cons})
ml=ml.sort_values('consensus_mean_rank').reset_index(drop=True)
ml['consensus_rank']=np.arange(1,len(ml)+1)
ml['consensus_percentile']=100*(1-(ml['consensus_rank']-1)/max(len(ml)-1,1))
ml.to_csv(OUT/'exploratory_ml_ranking.csv',index=False)


def row_or_none(df, gene):
    z=df[df.gene==gene]
    return None if z.empty else z.iloc[0].to_dict()

summary={
    'seed':SEED,
    'n_samples':len(meta),
    'n_donors':int(meta.donor.nunique()),
    'n_genes_genomewide':int(len(genes)),
    'severity_levels':sorted(meta.severity.unique().tolist()),
    'lipogenesis_genes_predeclared':len(LIPOGENESIS),
    'lipogenesis_genes_present':len(module_present),
    'lipogenesis_module_vs_severity':{'spearman_rho':float(mod_rho),'p':float(mod_p)},
    'FASN_genomewide_monotonic':row_or_none(trend,'FASN'),
    'FASN_genomewide_quadratic':row_or_none(quad,'FASN'),
    'FASN_lipogenesis_module':row_or_none(module,'FASN'),
    'FASN_exploratory_ML':row_or_none(ml,'FASN'),
    'notes':[
        'Primary inference uses sample-level pseudobulk across 11 samples; cells are not treated as independent replicates.',
        'Severity is encoded a priori as Thompson II=2, II-III=2.5, III=3, III-IV=3.5.',
        'ML is exploratory because only 11 samples / 7 donors are available; rankings are descriptive, not validation of predictive performance.'
    ]
}
with open(OUT/'fasn_summary.json','w') as f: json.dump(summary,f,indent=2,default=lambda x: float(x) if isinstance(x,(np.floating,np.integer)) else x)
print('===== FASN SUMMARY =====')
print(json.dumps(summary,indent=2,default=str),flush=True)
