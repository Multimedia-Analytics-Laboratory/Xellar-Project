"""Recompute manuscript prediction metrics using only the Python standard library."""
from pathlib import Path
import csv, gzip, json, math, statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('agonist', 'antagonist', 'blocker', 'inhibitor')

def rows(name):
    with gzip.open(ROOT/'data/predictions'/name, 'rt', newline='') as handle:
        yield from csv.DictReader(handle)

def f1(items):
    scores=[]
    for label in CLASSES:
        tp=sum(a==label and b==label for a,b in items)
        fp=sum(a!=label and b==label for a,b in items)
        fn=sum(a==label and b!=label for a,b in items)
        scores.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.)
    return sum(scores)/len(scores)

def main():
    groups=defaultdict(list)
    pair_sets=defaultdict(set)
    seen=set()
    for filename in ('cpg0004_routing.csv.gz','cpg0004_baselines.csv.gz','cpg0004_baselines_double_disjoint.csv.gz'):
        for row in rows(filename):
            model=row.get('strategy') or row['model']
            key=(row['split'],model,int(row['repeat']))
            pair=(row['drug_name'],row['target_name'])
            assert (*key,*pair) not in seen, ('duplicate prediction',key,pair)
            seen.add((*key,*pair))
            probabilities=[float(row['score__'+label]) for label in CLASSES]
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            assert abs(sum(probabilities)-1)<1e-5
            groups[key].append((row['true_label'],row['predicted_label']))
            pair_sets[key].add((*pair,row['true_label']))
    reference=next(iter(pair_sets.values()))
    assert len(reference)==898
    assert len({p[0] for p in reference})==284
    assert all(p==reference for p in pair_sets.values())
    repeated=defaultdict(list)
    for (split,model,repeat),items in groups.items():
        repeated[(split,model)].append(f1(items))
    out=[]
    for (split,model),values in sorted(repeated.items()):
        assert len(values)==3,(split,model,len(values))
        out.append({'split':split,'model':model,'macro_f1_mean':statistics.mean(values),'macro_f1_sd':statistics.stdev(values),'repeats':len(values)})
    expected={'random_pair':.9662312820436171,'drug_disjoint':.7761913915684975,'target_disjoint':.9351253705634472,'double_disjoint':.7179949267275049}
    for row in out:
        if row['model']=='pair_conditioned_gate':
            assert abs(row['macro_f1_mean']-expected[row['split']])<1e-12,row
    output=ROOT/'reproduced_results'
    output.mkdir(exist_ok=True)
    with (output/'prediction_macro_f1.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    print(json.dumps({'verified_pairs':898,'verified_drugs':284,'rows_checked':len(seen),'result':str(output/'prediction_macro_f1.csv'),'ReVMAP':{r['split']:r['macro_f1_mean'] for r in out if r['model']=='pair_conditioned_gate'}},indent=2))

if __name__=='__main__':
    main()
