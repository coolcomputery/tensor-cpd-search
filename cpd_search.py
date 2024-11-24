import numpy as np, itertools as it, time
import functools

def mexp(n,k,MOD):
    '''
    return n^k modulo MOD
    '''
    assert k>=0
    if k==0:
        return 1
    h=mexp(n,k//2,MOD)
    h=(h*h)%MOD
    return h if k%2==0 else (n*h)%MOD
def minv(n,MOD):
    '''
    return the multiplicative inverse of n modulo MOD
    assumes MOD is prime
    '''
    assert n!=0
    return mexp(n,MOD-2,MOD)

# ==== matrix utils ====
def mat_row_reduce(M,MOD):
    '''
    return (rref(M),Q,iQ,r) s.t. Q@M=rref(M), Q@iQ=identity, r=rank(M)
    assumes MOD is prime
    '''
    M=np.array(M)  # avoid mutating object passed in as parameter M
    assert M.dtype==np.dtype('int64') and ((M>=0)&(M<MOD)).all(), (M.dtype,M)
    
    m,n=M.shape
    Q,iQ=np.eye(m,dtype=int),np.eye(m,dtype=int)
    
    r=0
    for j in range(n):
        if (M[r:,j]!=0).any():
            i=min(i for i in range(r,m) if M[i][j]!=0)
            
            M[[r,i],:]=M[[i,r],:]
            Q[[r,i],:]=Q[[i,r],:]
            iQ[:,[r,i]]=iQ[:,[i,r]]
            
            s=M[r][j]
            i_s=minv(s,MOD)
            M[r,:]=(M[r,:]*i_s)%MOD
            Q[r,:]=(Q[r,:]*i_s)%MOD
            iQ[:,r]=(iQ[:,r]*s)%MOD
            
            for i in range(m):
                if i!=r:
                    s=-M[i][j]
                    M[i,:]=(M[i,:]+s*M[r,:])%MOD
                    Q[i,:]=(Q[i,:]+s*Q[r,:])%MOD
                    iQ[:,r]=(iQ[:,r]-s*iQ[:,i])%MOD
            r+=1
    
    return (M,Q,iQ,r)

def mat_rank(M,MOD):
    _,_,_,r = mat_row_reduce(M,MOD)
    return r
def mat_rankfac(M,MOD):
    rM,_,S,r=mat_row_reduce(M,MOD)
    return S[:,:r],rM[:r,:]
def mat_inv(M,MOD):
    assert M.shape[0]==M.shape[1], 'not square'
    _,Q,_,r=mat_row_reduce(M,MOD)
    assert r==M.shape[0], 'not invertible'
    return Q

'''
==== tensor utils ====
throughout this file, M @_d T is shorthand for the axis-d operation on tensor T by matrix M
'''
def rotate(T,sh):
    '''
    rotate axes of T s.t. axis 0 of the result is axis sh of the input
    '''
    return np.transpose(T,axes=np.roll(np.arange(len(T.shape)),-sh))

def axis_op0(M,T,MOD):
    assert len(M.shape)==2
    return np.tensordot(M,T,axes=[(-1,),(0,)])%MOD
def axis_op(M,T,ax,MOD):
    '''
    return tensor T' such that
    T'[i_0,...,i'_ax,...,i_{D-1}] = sum_{i_ax} M[i'_ax,i_ax]*T[i_0,...,i_{D-1}]
    '''
    return rotate(axis_op0(M,rotate(T,ax),MOD),-ax)
def axis_ops(Ms,T,MOD):
    assert len(Ms)==len(T.shape)
    for ax,M in enumerate(Ms):
        T=axis_op(M,T,ax,MOD)
    return T

def unfold(T,ax):
    T=rotate(T,ax)
    return T.reshape((T.shape[0],np.prod(T.shape[1:])))
def axis_reduce(T,MOD):
    '''
    return (T', S) such that T' is concise and T==S[0] @_0 (S[1] @_1 ... (S[D-1] @_{D-1} T') ... )
    '''
    D=len(T.shape)
    info=[]
    for ax in range(D):
        _,Q,iQ,r=mat_row_reduce(unfold(T,ax),MOD)
        info.append((Q[:r,:],iQ[:,:r]))
    return axis_ops([Q for Q,_ in info], T, MOD), [S for _,S in info]

'''
==== CPD utils ====
A rank-R CPD consisting of factor matrices A_0,...,A_{D-1}
is stored as the list of tuples
[
    tuple(A_d[:,r] for d in range(D))
    for r in range(R)
]
'''
def cpd_eval(cpd,MOD,shape=None):
    '''
    return the tensor that the CPD `cpd` evaluates to

    if `cpd` is an empty list:
    * returns an all-zeros tensor of shape `shape` if specified
    * else, throws error
    '''
    if len(cpd)==0:
        assert shape is not None, 'shape must be specified for empty CPD'
    return sum(
        (functools.reduce(np.multiply.outer,vec_tuple) for vec_tuple in cpd),
        start=(0 if shape is None else np.zeros(shape,dtype=int))
    )%MOD
def cpd_transform(cpd,Qs,MOD):
    '''
    return a CPD such that
    the d-th vector v in each tuple has been replaced with Qs[d]@v

    if Qs[d] is None, replace Qs[d]@v with v, i.e. treat Qs[d] as an identity matrix
    '''
    assert all(len(Qs)==len(vs) for vs in cpd), f'incorrect number of slice-op transformers: {Qs}'
    return [
        tuple((v if Q is None else (Q@v)%MOD) for Q,v in zip(Qs,vs))
        for vs in cpd
    ]
def cpd_rotate(cpd,sh):
    '''
    return a CPD such that
    each tuple has been rotated to the left by sh units

    satisfies the relation cpd_eval(cpd_rotate(cpd,sh),MOD) == rotate(cpd_eval(cpd,MOD),sh)
    '''
    if len(cpd)==0:
        return cpd
    D=len(cpd[0])
    sh%=D
    return [tupl[sh:]+tupl[:sh] for tupl in cpd]

'''
==== CPD DFS ====
allows arbitrary list of *pruners*

pruner: given `(T,R,MOD)`, outputs
* a CPD of `T` over the integers mod `MOD` with rank $\le$`R`, if it manages to find one
* `False` if it can prove that no rank-`R` decomp of `T` over the integers mod `MOD` exists
* `None` otherwise

implementation detail: pruner functions are allowed to assume that any tensor they receive is already axis-reduced
'''
NO_CPD_EXISTS=False
CANNOT_DETERMINE_IF_CPD_EXISTS=None
assert NO_CPD_EXISTS!=CANNOT_DETERMINE_IF_CPD_EXISTS

def assert_cpd_valid(T,R,MOD,cpd):
    assert cpd!=NO_CPD_EXISTS and cpd!=CANNOT_DETERMINE_IF_CPD_EXISTS and len(cpd)<=R, (T,R,MOD,cpd)
    ret=cpd_eval(cpd,MOD,shape=T.shape)
    assert np.array_equal(T,ret), (T,cpd,ret)

def vec_1hot(n,i):
    assert 0<=i and i<n, (n,i)
    vec=np.zeros(n,dtype=int)
    vec[i]=1
    return vec

def all_vecs(n,MOD):
    return (np.array(v,dtype=int) for v in it.product(range(MOD),repeat=n))

def cpd_search_help(T_target,R,MOD,pruners=None,verbose=True):
    ''' assume T_target is concise '''
    D=len(T_target.shape)
    N0=T_target.shape[0]
    
    if pruners is None:
        pruners={}
    assert isinstance(pruners,dict), (type(pruners),pruners)
    
    work_stats={
        key:[0]*(R-N0+1)
        for key in ['work']+[f'prune::{name}' for name in pruners]
    }
    timer_info={
        'start':time.perf_counter(),'mark':0,'elapsed':0,
    }
    def update_stats(force_display=False):
        timer_info['elapsed']=time.perf_counter()-timer_info['start']
        if force_display or (timer_info['elapsed']>timer_info['mark'] and timer_info['mark']>0):
            print(f"t={timer_info['elapsed']}")
            print(work_stats)
        while timer_info['elapsed']>timer_info['mark']:
            timer_info['mark']+=10**int(np.log10(max(10,timer_info['mark'])))
    
    def dfs(T,rem):
        ''' assume all shape lengths of T are <=R '''
        if verbose:
            update_stats()
        work_stats['work'][rem]+=1
        
        T_compressed,Ss_uncompress=axis_reduce(T,MOD)
        for name,pruner in pruners.items():
            ret=pruner(T_compressed,N0+rem,MOD)
            if ret!=CANNOT_DETERMINE_IF_CPD_EXISTS:
                work_stats[f'prune::{name}'][rem]+=1
                if ret==NO_CPD_EXISTS:
                    return None
                else:
                    return cpd_transform(ret,Ss_uncompress,MOD)
        
        assert rem>=0, rem
        if rem==0:
            B=[]
            fac_tuples=[]
            for v in all_vecs(N0,MOD):
                T_slice=axis_op(v.reshape((1,N0)),T,0,MOD)[0]
                T_slice_reduce,Ss=axis_reduce(T_slice,MOD)
                if (
                    all(r<=1 for r in T_slice_reduce.shape)
                    and mat_rank(np.array(B+[v]),MOD)>len(B)
                ):
                    slice_idx=len(B)
                    B.append(v)
                    if all(r>=1 for r in T_slice_reduce.shape):
                        fac_tuples.append(
                            (vec_1hot(N0,slice_idx)*T_slice_reduce.flatten()[0],)
                            +tuple(Ss[d-1][:,0] for d in range(1,D))
                        )
            if len(B)<N0:
                return None
            B_mat=np.array(B,dtype=int)
            return cpd_transform(fac_tuples,(mat_inv(B_mat,MOD),)+(None,)*(D-1),MOD)
        
        # fix the first rank of factor vectors in CPD
        # case: tensor product of fixed factor vectors is the all-zero tensor
        ret=dfs(T,rem-1)
        if ret is not None:
            return ret
        
        # all other cases of factor vectors
        for fac_vecs in it.product(*[
            (v for v in all_vecs(n,MOD) if (v!=0).any())
            for n in T.shape
        ]):
            ret=dfs((T-cpd_eval([fac_vecs],MOD))%MOD,rem-1)
            if ret is not None:
                return ret+[fac_vecs]
        return None
    
    cpd=(
        dfs(T_target,R-N0)
        if max(T_target.shape)<=R else
        None
    )
    if verbose:
        update_stats(force_display=True)
    return {
        'cpd':cpd,
        'stats':work_stats,
    }

def cpd_search(T_target,R,MOD,pruners=None,verbose=True,return_stats=False):
    D=len(T_target.shape)
    T_reduced,Ss=axis_reduce(T_target,MOD)
    d_max=max(range(D),key=(lambda ax:T_reduced.shape[ax]))
    raw_output=cpd_search_help(rotate(T_reduced,d_max),R,MOD,pruners=pruners,verbose=verbose)
    
    cpd_compressed_rotated=raw_output['cpd']
    cpd=None
    if cpd_compressed_rotated is not None:
        cpd=cpd_transform(cpd_rotate(cpd_compressed_rotated,-d_max),Ss,MOD)
        assert_cpd_valid(T_target,R,MOD,cpd)
    return (
        {
            'cpd':cpd,
            'stats':raw_output['stats']
        }
        if return_stats else
        cpd
    )

'''
==== pruners ====
* rref (reduced row-echelon form) pruning
* rank-sum pruning
'''
def expand_pruner_rotate(pruner):
    '''
    apply a pruner to all rotations of an input tensor,
    so that the original pruner function does not have to
    '''
    def _f(T,R,MOD,**kwargs):
        for sh in range(len(T.shape)):
            ret=pruner(rotate(T,sh),R,MOD,**kwargs)
            if ret!=CANNOT_DETERMINE_IF_CPD_EXISTS:
                if ret==NO_CPD_EXISTS:
                    return NO_CPD_EXISTS
                else:
                    return [tupl[-sh:]+tupl[:-sh] for tupl in ret]
        return CANNOT_DETERMINE_IF_CPD_EXISTS
    return _f

def bound_slice_decomps_help(T,R,MOD):
    '''
    rank-factorize each axis-0 slice of a tensor T
    requires T to be 3-dimensional
    '''
    assert len(T.shape)==3, T.shape
    n0=T.shape[0]
    if sum(mat_rank(M,MOD) for M in T)<=R:
        out=[]
        for i in range(n0):
            v_1hot=np.zeros(n0,dtype=int)
            v_1hot[i]=1
            C,F=mat_rankfac(T[i,:,:],MOD)
            out.extend([
                (v_1hot,C[:,r],F[r,:])
                for r in range(C.shape[1])
            ])
        return out
    return CANNOT_DETERMINE_IF_CPD_EXISTS
bound_slice_decomps=expand_pruner_rotate(bound_slice_decomps_help)

'''
## rref-pruning

suppose T has shape n_0 x ... x n_{D-1}, is concise, and has a rank-R CPD [[A_0, ... A_{D-1}]]
construct some Q such that Q@A_0 = rref(A_0)
then Q @_0 T = [[Q@A_0, ... A_{D-1}]] has the property that each slice of the form i,:,...,: has rank <= R-n_0+1,
    because each row of Q@A_0 has at most that many nonzeros
so there exist n_0 many linearly independent row vectors v such that rk(v @_0 T) <= R-n_0+1

checking this condition is only efficient if D=3 (so each slice is a matrix, for which rank can be calculated in polynomial time),
so we enforce this requirement

additionally, we run a heuristic for finding a low-rank CPD of T:
find n_0 many lin. indep. row vectors v_0,...,v_{n_0} such that sum_i rk(v_i @_0 T) is minimized; then construct a CPD out of this
we use a greedy algorithm to find such a set of vectors:
* sort v by ascending value of rk(v @_0 T)
* for each v, add it to a set if doing so increases the dimension of the span of that set
'''
def prune_rref0(T,R,MOD,verbose=False):
    '''
    requires T to be 3-dimensional and to be concise along axis 0
    '''
    assert len(T.shape)==3, T.shape
    assert R>=0, R
    n0=T.shape[0]
    if n0==0:
        return []
    # guaranteed to not return `CANNOT_DETERMINE_IF_CPD_EXISTS` if n0==R
    assert mat_rank(T.reshape((n0,np.prod(T.shape[1:]))),MOD)==n0, f'not concise along axis 0: {T=}'
    s=R-n0+1
    if verbose: print(f'{T.shape=} {s=}')
    
    pairs=(
        (mat_rank(axis_op(v.reshape((1,n0)),T,0,MOD)[0],MOD), v)
        for v in all_vecs(n0,MOD)
    )
    good_pairs=sorted(
        [(weight,v) for weight,v in pairs if weight<=s],
        key=(lambda p:p[0]),
    )
    if verbose: print(good_pairs)
    
    # greedy: add next cheapest vector, if it is linearly independent from all previously chosen vectors
    good_vecs=[]
    for _,v in good_pairs:
        if mat_rank(np.array(good_vecs+[v]),MOD)>len(good_vecs):
            good_vecs.append(v)
            if len(good_vecs)==n0:
                break
    if len(good_vecs)<n0:
        return NO_CPD_EXISTS
    Q=np.array(good_vecs)
    ret=bound_slice_decomps(axis_op(Q,T,0,MOD),R,MOD)
    if ret!=CANNOT_DETERMINE_IF_CPD_EXISTS:
        return cpd_transform(ret,(mat_inv(Q,MOD),)+(None,)*(len(T.shape)-1),MOD)
    return CANNOT_DETERMINE_IF_CPD_EXISTS
prune_rref=expand_pruner_rotate(prune_rref0)

'''
## rank-sum pruning

Let T be a tensor with elements in a finite field F
suppose has a rank-R CPD [[A_0,...,A_{D-1}]]

for any row vector v, rk(v @_0 T) is at most the number of nonzeros in v@A_0

choose arbitrary row vectors v, w;
consider the quantity x := sum_{s in F} rk((w+s*v) @_0 T)
we have x <= sum_{s in F} (# of nonzeros in (w+s*v) @ A_0)

some algebra shows that the right hand side is at most R*|F| - (# of nonzeros in v@A)
this expression is at most R*|F| - rk(v @_0 T)

so we check the inequality
    sum_{s in F} rk((w+s*v) @_0 T) <= R*|F| - rk(v @_0 T)
for all pairs of row vectors v, w

checking this condition is only efficient if D=3 (so each slice is a matrix, for which rank can be calculated in polynomial time),
so we enforce this requirement
'''

def prune_contract_ranksum0(T,R,MOD,verbose=False):
    assert len(T.shape)==3
    n0=T.shape[0]
    ranks=np.zeros((MOD,)*n0,dtype=int)
    for v in all_vecs(n0,MOD):
        ranks[tuple(v)]=mat_rank(axis_op(v.reshape((1,n0)),T,0,MOD)[0],MOD)
    if verbose:
        print(ranks)
        for v,w in it.product(all_vecs(n0,MOD),repeat=2):
            ret=sum(ranks[tuple((w+s*v)%MOD)] for s in range(MOD))
            thresh=R*MOD-ranks[tuple(v)]
            if ret>thresh:
                print(f'ruled out by {(v,w)=}; {ret=} > {thresh=}')
                break
    return CANNOT_DETERMINE_IF_CPD_EXISTS if all(
        sum(ranks[tuple((w+s*v)%MOD)] for s in range(MOD))<=R*MOD-ranks[tuple(v)]
        for v in all_vecs(n0,MOD)
        for w in all_vecs(n0,MOD)
    ) else NO_CPD_EXISTS

prune_contract_ranksum=expand_pruner_rotate(prune_contract_ranksum0)

# ==== convenience functions for CLI ====
def MMTf(p,q=None,r=None):
    '''
    return tensor of shape p*q x q*r x r*p
    corresponding to matrix multiplication between shapes p x q and q x r

    if both q and r are not specified, both default to p
    if exactly one of q and r is not specified, throws error
    '''
    assert sum(1 for v in [q,r] if v is None)!=1, (p,q,r)
    if q is None and r is None:
        q,r=p,p
    out=np.zeros((p,q,q,r,r,p),dtype=int)
    for i in range(p):
        for j in range(q):
            for k in range(r):
                out[i,j,j,k,k,i]=1
    return out.reshape((p*q,q*r,r*p))

# ==== CLI ====
import argparse
def parse_int_tuple(s):
    # s must be a comma-delimited string of integers with no whitespace
    return tuple(map(int,s.split(',')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Tensor CPD (Canonical Polyadic Decomposition) DFS over integers mod a prime')
    parser.add_argument('mod', nargs=1, type=int, help='modulo of finite field')
    parser.add_argument('rank', nargs=1, type=int, help='desired rank threshold of CPD')
    
    parser.add_argument('--shape', nargs=1, type=parse_int_tuple, help='shape of tensor to be searched over')
    parser.add_argument('--data', nargs=1, type=parse_int_tuple, help='elements of tensor flattened in row-major order')
    parser.add_argument('--matmul', nargs=3, type=int, help='matrix multiplication tensor <m,k,n> parameters (original 6-D tensor has adjacent pairs of dimensions flattened)')
    
    parser.add_argument('--pruners', nargs='*', type=str, help='list of extra DFS pruners to use (worst-case, slows down search by multiplicative factor MOD^(max shape length along one axis))')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='display search progress')
    
    args=parser.parse_args()
    MOD=args.mod[0]
    assert MOD>=2 and all(MOD%i!=0 for i in range(2,MOD)), f'{MOD} is not prime'
    rank_thresh=args.rank[0]
    
    T_target=None
    if args.shape is not None:
        assert args.data is not None, 'data is required if shape is specified'
        shape=args.shape[0]
        data=args.data[0]
        T_target=np.array(data,dtype=int).reshape(shape)
        if not ((T_target>=0)&(T_target<MOD)).all():
            print(f'WARNING: reducing data modulo {MOD}')
            T_target%=MOD
    else:
        assert args.matmul is not None, 'matmul parameters is required if default specification of tensor is not provided'
        T_target=MMTf(*args.matmul)
    print('target tensor:')
    print(T_target)
    
    pruner_name2func={
        'rref':prune_rref,
        'ranksum':prune_contract_ranksum,
    }
    pruners=None
    if args.pruners is not None:
        pruners={
            name:pruner_name2func[name]
            for name in args.pruners
        }
    
    cpd=cpd_search(T_target, rank_thresh, MOD, pruners=pruners, verbose=args.verbose)
    
    print('='*32)
    if cpd is None:
        print(f'no CPD of rank <={rank_thresh} found')
    else:
        print(f'found CPD of rank <={rank_thresh}')
        print(cpd)
