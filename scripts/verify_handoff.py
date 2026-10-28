#!/usr/bin/env python3
"""Read-only audit for Grisha's gold-prospectivity research freeze (2026-10-27).

Usage:
    python scripts/verify_handoff.py --root /path/to/extracted-1.7z --report audit.json

Never mutates source files. Does not run heavy model retraining. Optional rasterio
validation uses the exact packaged GeoTIFF grids.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

EPS=1e-6
EXPECTED={
    'raw_o10':0.19860323003055436,
    'detrended_o10':0.3622872108249673,
    'ceiling_raw':0.6988214753382802,
    'ceiling_detrended':0.6219991270187691,
    'random_cv_coordinates':0.9968391087094173,
    'spatial_cv_coordinates':0.5398692272378223,
    'random_cv_135':0.9974655152912424,
    'spatial_cv_135':0.4145797009072901,
}

def read_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run(root:Path, strict:bool=False):
    root=root.resolve()
    reports={'source':str(root),'checks':{},'observed':{},'warnings':[],'status':'PARTIAL'}
    checks=reports['checks'];obs=reports['observed']
    required=[root/'04_metrics/main_metrics.csv',root/'04_metrics/ceiling_metrics.csv',root/'04_metrics/controls.csv',root/'04_metrics/o10_block_ci.csv',root/'02_protocol/fold_leakage_check.json',root/'05_maps/crit.tif',root/'05_maps/oof_remote.tif',root/'05_maps/overlap_class.tif',root/'03_runs/stdout_pytest_full.log']
    missing=[str(f.relative_to(root)) for f in required if not f.is_file()]
    checks['required_inputs_present']={'pass':not missing,'missing':missing}
    if missing:
        reports['status']='INCOMPLETE'
        return reports
    main=read_csv(root/'04_metrics/main_metrics.csv');ceil=read_csv(root/'04_metrics/ceiling_metrics.csv');controls=read_csv(root/'04_metrics/controls.csv');ci=read_csv(root/'04_metrics/o10_block_ci.csv')
    def get(rows,prefix,col):
        match=[r for r in rows if r[list(r.keys())[0]].startswith(prefix)]
        if len(match)!=1:raise AssertionError((prefix,len(match)))
        return float(match[0][col])
    computed={
        'raw_o10':get(main,'спектры только на обнажённых','совпадение_верх_10'),
        'detrended_o10':get(main,'спектры только на обнажённых','совпадение_без_тренда'),
        'ceiling_raw':get(ceil,'идеальная карта через конвейер','совпадение_верх_10'),
        'ceiling_detrended':get(ceil,'идеальная карта через конвейер','совпадение_без_тренда'),
        'random_cv_coordinates':next(float(x['roc_auc']) for x in controls if x['схема'].startswith('случайная') and x['набор']=='координаты'),
        'spatial_cv_coordinates':next(float(x['roc_auc']) for x in controls if x['схема'].startswith('полосы') and x['набор']=='координаты'),
        'random_cv_135':next(float(x['roc_auc']) for x in controls if x['схема'].startswith('случайная') and '135' in x['набор']),
        'spatial_cv_135':next(float(x['roc_auc']) for x in controls if x['схема'].startswith('полосы') and '135' in x['набор']),
    }
    obs.update(computed)
    diffs={k:abs(computed[k]-v) for k,v in EXPECTED.items()}
    checks['metrics_match_frozen_reference']={'pass':max(diffs.values())<=EPS,'absolute_differences':diffs,'tolerance':EPS}
    expected_ci={'raw':[0.06518425404127275,0.3481482171852734], 'detrended':[0.1755902382640097,0.5363779722373598]}
    observed_ci={
        'raw':[get(ci,'сырое','ниж95'),get(ci,'сырое','верх95')],
        'detrended':[get(ci,'без тренда','ниж95'),get(ci,'без тренда','верх95')],
    }
    obs['bootstrap_CIs']=observed_ci
    checks['block_CI_matches']={'pass':all(all(abs(u-v)<EPS for u,v in zip(observed_ci[k],expected_ci[k])) for k in expected_ci)}
    folds=json.loads((root/'02_protocol/fold_leakage_check.json').read_text(encoding='utf-8-sig'))
    checks['fold_leakage_check_pass']={'pass':len(folds)==8 and all(x['overlap_train_test']==0 and x['min_dist_ok'] for x in folds),'n_folds':len(folds),'source':'reported fold audit; raw geometry not recalculated here'}
    try:
        import numpy as np
        import rasterio
        with rasterio.open(root/'05_maps/crit.tif') as c, rasterio.open(root/'05_maps/oof_remote.tif') as p, rasterio.open(root/'05_maps/overlap_class.tif') as q:
            map_valid=c.crs==p.crs==q.crs and c.transform==p.transform==q.transform and c.shape==p.shape==q.shape
            a,b,cl=c.read(1),p.read(1),q.read(1)
        valid=np.isfinite(a)&np.isfinite(b)
        vals,ct=np.unique(cl.astype(int),return_counts=True)
        cats=dict(zip(map(str,vals.tolist()),map(int,ct.tolist())))
        # In packaged overlap_class.tif: 0 invalid,1 neither,2 only target,3 only model,4 both.
        n_target=cats.get('2',0)+cats.get('4',0)
        n_model=cats.get('3',0)+cats.get('4',0)
        n_both=cats.get('4',0)
        overlap=n_both/n_target if n_target else None
        checks['raster_grid_consistency']={'pass':bool(map_valid and valid.sum()==22905),'valid_cells':int(valid.sum()),'crs_equal':bool(map_valid),'shape':a.shape}
        checks['independent_raw_O10_from_maps']={'pass':n_target==2291 and n_model==2291 and n_both==455 and abs(overlap-computed['raw_o10'])<EPS,'target_top_cells':n_target,'model_top_cells':n_model,'intersection':n_both,'O10':overlap,'class_counts':cats}
        obs['raster_raw_o10']=overlap
    except ImportError:
        checks['raster_grid_consistency']={'pass':None,'reason':'rasterio/numpy missing'}
        checks['independent_raw_O10_from_maps']={'pass':None,'reason':'rasterio/numpy missing'}
        reports['warnings'].append('Raster check not executed; install rasterio/numpy in an isolated environment')
    # Format is CSV-like: sha256,path header, rows sha *./relative/path,
    manifest=root/'07_repro_package/MANIFEST_SHA256.csv'
    if manifest.exists():
        lines=manifest.read_text(encoding='utf-8-sig').splitlines()
        ok=bad=notfound=unparsed=0;failed=[]
        for row in lines[1:]:
            mt=re.match(r'^([0-9a-f]{64}) \*(\./.*?),?$',row)
            if not mt:
                unparsed+=1;continue
            expected,path=mt.groups();file=root/'07_repro_package'/path[2:]
            if not file.exists():notfound+=1;failed.append(f'MISSING {path}');continue
            if sha(file)!=expected:bad+=1;failed.append(f'HASH {path}');continue
            ok+=1
        checks['repro_manifest_sha256']={'pass':not (bad or notfound or unparsed) and ok>0,'verified_files':ok,'bad_hash':bad,'missing_files':notfound,'invalid_entries':unparsed,'details':failed[:15]}
    else:checks['repro_manifest_sha256']={'pass':False,'reason':'MANIFEST_SHA256.csv is missing'}
    old=(root/'03_runs/stdout_pytest_full.log').read_text(encoding='utf-8-sig',errors='replace')
    last=re.findall(r'(\d+) failed, (\d+) passed',old)
    passed=re.findall(r'(\d+) passed',old)
    if last:reports['observed']['packaged_pytest_log']={'failed':int(last[-1][0]),'passed':int(last[-1][1])}
    elif passed:reports['observed']['packaged_pytest_log']={'failed':0,'passed':int(passed[-1])}
    else:reports['observed']['packaged_pytest_log']={'failed':None,'passed':None}
    reported_240='240 passed' in (root/'00_state/BLOCKERS.md').read_text(encoding='utf-8-sig',errors='replace') if (root/'00_state/BLOCKERS.md').exists() else False
    final_pass=reports['observed']['packaged_pytest_log'] == {'failed':0,'passed':240}
    checks['pytest_claim_consistent']={'pass':final_pass or not reported_240,'claimed_240':reported_240,'packaged_log':reports['observed']['packaged_pytest_log'],'fix_required':reported_240 and not final_pass}
    if reported_240 and not final_pass:reports['warnings'].append('BLOCKERS.md claims 240 passed; packaged full pytest log reports failure: rerun and attach a new verifiable log.')
    test=root/'07_repro_package/project/tests/test_s2_composite.py'
    if test.exists():
        txt=test.read_text(encoding='utf-8-sig')
        old_iter='it = iter(fakes)' in txt and 'next(it)' in txt
        checks['s2_test_repaired_in_copy']={'pass':not old_iter,'contains_one_shot_iterator':old_iter}
    # count hardcoded Windows paths in auxiliary scripts
    paths=[]
    for f in [root/'06_article/build_article.py',root/'05_maps/make_maps.py']:
        if f.is_file():
            content=f.read_text(encoding='utf-8-sig',errors='replace')
            drives=sorted(set(re.findall(r'(?i)[A-Z]:\\\\?[^\s\x00\"\']+',content)))
            if drives:paths.append({'file':str(f.relative_to(root)),'examples':drives[:4]})
    checks['aux_scripts_portable']={'pass':not paths,'absolute_windows_paths':paths}
    reports['warnings'].append('No independent mineral occurrence labels for this study area; O10 is not a mining discovery metric.')
    reports['warnings'].append('No full retraining performed; report validates packaged artifacts, not complete independent model execution.')
    fails=[k for k,v in checks.items() if v.get('pass') is False]
    reports['status']='REWORK' if fails else 'CHECKS_PASSED_WITH_LIMITATIONS'
    reports['failed_checks']=fails
    return reports

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--strict',action='store_true',help='Exit 1 if any check fails')
    args=parser.parse_args();result=run(args.root,args.strict)
    s=json.dumps(result,ensure_ascii=False,indent=2,default=str)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(s,encoding='utf-8')
    print(s)
    if args.strict and result['status']!='CHECKS_PASSED_WITH_LIMITATIONS':raise SystemExit(1)
if __name__=='__main__':main()
