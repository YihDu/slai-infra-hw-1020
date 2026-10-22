"""Equivalent fixed-panel bounds formulation, evaluated separately before integration."""
import json,time
from pathlib import Path
import z3

def check(rows,cols,panel):
 group,offset=z3.Ints('panel_index offset')
 full,tail=divmod(rows,panel)
 clauses=[]
 for stride,valid in [(panel,z3.And(group>=0,group<full,offset>=0,offset<panel*cols))]+([(tail,z3.And(group==full,offset>=0,offset<tail*cols))] if tail else []):
  row=group*panel+offset%stride
  col=z3.If(group%2==1,cols-1-offset/stride,offset/stride)
  clauses.append(z3.And(valid,z3.Or(row<0,row>=rows,col<0,col>=cols)))
 solver=z3.Solver();solver.set(timeout=3000);solver.add(z3.Or(*clauses))
 t=time.monotonic();r=solver.check()
 return dict(rows=rows,cols=cols,panel=panel,status=str(r),seconds=time.monotonic()-t,smt2=solver.to_smt2())
if __name__=='__main__':
 reports=[check(r,c,p) for r in [1,2,3,9,32,64,128,256] for c in [1,3,64,128] for p in [4,8,16,32,64]]
 assert all(r['status']=='unsat' for r in reports)
 Path(__file__).with_suffix('.json').write_text(json.dumps(reports,indent=2)+'\n')
 print(json.dumps(dict(cases=len(reports),max_seconds=max(r['seconds'] for r in reports),status='all_unsat')))
