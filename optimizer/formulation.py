"""Validated finite-domain MIP. No measured performance enters tuple construction."""
import itertools
import math
import highspy
import z3

CATALOG = dict(block_M=[32,64,128],block_N=[64,128],block_K=[32,64],num_stages=[1,2,3],threads=[128],swizzle=[0,4,8,16])


def validate(plan):
    if set(plan['domains']) != set(CATALOG): raise ValueError('Incorrect decision domains')
    for key, allowed in CATALOG.items():
        values=plan['domains'][key]
        if not isinstance(values,list) or not values or any(type(x) is not int or x not in allowed for x in values):
            raise ValueError('Invalid domain: '+key)
        if len(set(values))!=len(values):raise ValueError('Duplicate domain entries')
    cap=plan['shared_memory_limit']
    if type(cap) is not int or not 8192<=cap<=49152:raise ValueError('Invalid resource budget')
    if set(plan['weights']) != {'traffic','launches','pipeline','footprint'}:raise ValueError('Incorrect cost weights')
    for x in plan['weights'].values():
        if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or not .001<=x<=100:
            raise ValueError('Invalid cost weight')
    if not isinstance(plan.get('rationale'),str):raise ValueError('Missing rationale')
    return plan


def solve(plan, shape, directory, timeout=20, excluded=(), hardware_limit=None):
    validate(plan); m,n,k=shape; rows=[]; names=list(CATALOG)
    for values in itertools.product(*(plan['domains'][x] for x in names)):
        c=dict(zip(names,values));bm,bn,bk=c['block_M'],c['block_N'],c['block_K']
        if m%bm or n%bn or k%bk:continue
        if c in excluded:continue
        # Explicit barriers separate lifetimes; TileLang aliases these buffers.
        # This estimate is still checked against actual compiled resource usage.
        smem=max(2*bk*(bm+bn)*c['num_stages'],4*bm*bn)
        # Analytical B-tile reuse within a row panel; not measured timing data.
        reuse=min(c['swizzle'] or 1,math.ceil(m/bm))
        features=dict(traffic=2*m*n*k*(1/(bm*reuse)+1/bn)/1e8,launches=math.ceil(m/bm)*math.ceil(n/bn)/1000,
                      pipeline=(k/bk)/c['num_stages']/100,footprint=smem/49152)
        rows.append(dict(config=c,smem_bytes_estimate=smem,features=features,cost=sum(plan['weights'][x]*v for x,v in features.items())))
    if not rows:raise ValueError('No untried tuple divides workload shape; expand or revise domains')
    h=highspy.Highs();h.setOptionValue('output_flag',False);h.setOptionValue('time_limit',timeout)
    for i,r in enumerate(rows):
        h.addVar(0,1);h.changeColIntegrality(i,highspy.HighsVarType.kInteger);h.changeColCost(i,r['cost']);h.passColName(i,'choose_'+str(i))
    h.addRow(1,1,len(rows),list(range(len(rows))),[1.]*len(rows))
    cap=min(plan['shared_memory_limit'],hardware_limit) if hardware_limit else plan['shared_memory_limit']
    h.addRow(-highspy.kHighsInf,cap,len(rows),list(range(len(rows))),[r['smem_bytes_estimate'] for r in rows])
    h.writeModel(str(directory/'model.lp'));h.writeModel(str(directory/'model.mps'));h.run()
    info=h.getInfo();status=h.modelStatusToString(h.getModelStatus());solution=h.getSolution()
    chosen=next((i for i,x in enumerate(solution.col_value) if x>.5),None) if solution.value_valid else None
    finite=lambda x:float(x) if math.isfinite(x) else None
    report=dict(status=status,incumbent=finite(info.objective_function_value),bound=finite(info.mip_dual_bound),
                gap=finite(info.mip_gap),selected_index=chosen,candidates=rows,
                cost_unit='dimensionless analytic proxy; not latency',highs_version=h.version(),
                effective_shared_memory_limit=cap,excluded_previously_evaluated=excluded)
    return (rows[chosen]['config'] if chosen is not None else None),report


def index_proof(shape, config, code_hash):
    results=[]
    for dimension,extent,tile in zip(['M','N','K'],shape,[config['block_M'],config['block_N'],config['block_K']]):
        block,local=z3.Ints('block_'+dimension+' local_'+dimension)
        index=block*tile+local;s=z3.Solver();s.set(timeout=3000)
        s.add(block>=0,block<(extent+tile-1)//tile,local>=0,local<tile)
        s.add(z3.Or(index<0,index>=extent));status=s.check()
        results.append(dict(dimension=dimension,status=str(status),smt2=s.to_smt2()))
    m,n,k=shape;rows=(m+config['block_M']-1)//config['block_M'];cols=(n+config['block_N']-1)//config['block_N']
    panel=config.get('swizzle',0)
    if panel:
        launch=z3.Int('launch_index');index=launch/(panel*cols);offset=launch%(panel*cols)
        panels=(rows+panel-1)//panel
        stride=z3.If(index<panels-1,panel,rows-(panels-1)*panel)
        row=index*panel+offset%stride;col=z3.If(index%2==1,cols-1-offset/stride,offset/stride)
        s=z3.Solver();s.set(timeout=3000);s.add(launch>=0,launch<rows*cols,z3.Or(row<0,row>=rows,col<0,col>=cols))
        results.append(dict(dimension='swizzled_tile_bounds',status=str(s.check()),smt2=s.to_smt2()))
    mapping=[launch_coordinate(i,rows,cols,panel) for i in range(rows*cols)]
    bijective=len(set(mapping))==rows*cols and all(0<=r<rows and 0<=c<cols for r,c in mapping)
    return dict(property='Tiled indices stay in bounds; serpentine launch map covers each tile exactly once',assumptions='Fixed shape/config; selected tiles divide tensor dimensions; physical launch indices cover the whole grid',
                shape=shape,config=config,code_sha256=code_hash,results=results,
                finite_mapping_check=dict(bijective=bijective,grid_tiles=rows*cols,method='Complete finite enumeration of coordinates, not performance evaluation'),
                scope='Integer bounds only; not floating-point correctness, pipeline safety or lowering correctness')


def launch_coordinate(index,rows,cols,panel):
    if not panel:return index//cols,index%cols
    group,offset=divmod(index,panel*cols)
    stride=min(panel,rows-group*panel)
    col=offset//stride
    return group*panel+offset%stride,cols-1-col if group%2 else col
