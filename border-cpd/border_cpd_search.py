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

class BPoly:
    '''
    Implementation of the polynomial ring (Z_MOD)[x]/(x^H).
    assumes MOD is prime
    '''
    def __init__(self,coeffs,MOD,H):
        assert isinstance(H,int) and H>0, f'{H=}'
        assert isinstance(MOD,int) and MOD>0, f'{MOD=}'
        self.MOD=MOD
        self.H=H
        assert len(coeffs)<=H, f'too many coeffs: {coeffs=} {H=}'
#         assert all(0<=v and v<MOD for v in coeffs), f'not all coeffs in [0,{MOD=}): {coeffs=}'
        self.coeffs=[0]*H
        for i,c in enumerate(coeffs):
            self.coeffs[i]=c%MOD
    def __repr__(self):
        return f'{self.coeffs}_{self.MOD}'.replace(' ','')
    
    def __eq__(self,o):
        if isinstance(o,int):
            o=BPoly([o],self.MOD,self.H)
        assert isinstance(o,BPoly), type(o)
        return self.MOD==o.MOD and self.H==o.H and self.coeffs==o.coeffs
    
    def __pos__(self):
        return self
    def __neg__(self):
        return BPoly([(-a)%self.MOD for a in self.coeffs],self.MOD,self.H)
    def __add__(self,o):
        if isinstance(o,int):
            o=BPoly([o],self.MOD,self.H)
        assert isinstance(o,BPoly), type(o)
        assert self.MOD==o.MOD and self.H==o.H, (self,o)
        return BPoly([a+b for a,b in zip(self.coeffs,o.coeffs)],self.MOD,self.H)
    def __radd__(self,o):
        return self+o
    def __sub__(self,o):
        return self+(-o)
    def __mul__(self,o):
        if isinstance(o,int):
            o=BPoly([o],self.MOD,self.H)
        if isinstance(o,np.ndarray):
            return o*self
        assert isinstance(o,BPoly), type(o)
        assert self.MOD==o.MOD and self.H==o.H, (self,o)
        return BPoly([sum(self.coeffs[i]*o.coeffs[h-i] for i in range(h+1)) for h in range(self.H)],self.MOD,self.H)
    def __rmul__(self,o):
        return self*o
    def _inv(self):
        '''
        return the multiplicative inverse of self
        '''
        assert self.coeffs[0]!=0, f'not invertible: {self}'
        inv_a0=minv(self.coeffs[0],self.MOD)
        out=[inv_a0]
        for d in range(1,self.H):
            out.append((-sum(self.coeffs[d-h]*out[h] for h in range(d))*inv_a0)%self.MOD)
        return BPoly(out,self.MOD,self.H)
    def __truediv__(self,o):
        if isinstance(o,np.ndarray):
            return np.array([
                self/val
                for val in o.flatten()
            ]).reshape(o.shape)
        if isinstance(o,int):
            o=BPoly([o],self.MOD,self.H)
        assert isinstance(o,BPoly), type(o)
        return self*o._inv()
    def __rtruediv__(self,o):
        return o*self._inv()
    
    def xpow(self):
        '''
        if self==0: return H
        else: return maximum h s.t. self is a multiple of x^h
        '''
        return min([deg for deg,c in enumerate(self.coeffs) if c!=0]+[self.H])
        
    def xfac(self):
        '''
        return h,y s.t. self==x^h * y, for maximal h
        such y is guaranteed to be invertible, but not guaranteed to be unique
        '''
        h=self.xpow()
        return h,BPoly(self.coeffs[h:],self.MOD,self.H)

'''
==== Matrix utils ====
'''
def border_eye(n,MOD,H):
    '''
    return the identity matrix over the ring (Z_MOD)[x]/(x^H)
    '''
    return np.array([
        [
            BPoly([(1 if i==j else 0)],MOD,H)
            for j in range(n)
        ]
        for i in range(n)
    ])

def border_row_reduc(M,debug=False):
    '''
    M: a matrix with elements of the form BPoly(...,MOD,H) for constant MOD, H

    return (M',Q,iQ,r) s.t.
    * Q@M=M'
    * Q@iQ=identity
    * r=rank(M)
    * M'[r:,:] is all-zeros
    '''
    assert isinstance(M,np.ndarray) and len(M.shape)==2 and all(n>0 for n in M.shape)
    M=np.array(M)
    MOD,H=M[0,0].MOD,M[0,0].H
    nR,nC=M.shape
    Q=border_eye(nR,MOD,H)
    iQ=border_eye(nR,MOD,H)
    
    rk=0
    good_clms=set(range(nC))
    while rk<min(nR,nC):
        # find a pivot element
        pt=None
        best_xpow=H
        for r,c in it.product(range(rk,nR),good_clms):
            h=M[r,c].xpow()
            if h<best_xpow:
                best_xpow=h
                pt=(r,c)
        if debug:
            print('pivot')
            print((pt,best_xpow,M))
        if pt is None:
            break
        
        r0,c0=pt
        pval=M[r0,c0]
        # swap rows rk,r0 to move pivot to row rk
        M[[rk,r0],:]=M[[r0,rk],:]
        Q[[rk,r0],:]=Q[[r0,rk],:]
        iQ[:,[rk,r0]]=iQ[:,[r0,rk]]
        if debug:
            print('after swap')
            print(M,Q,iQ)

        # normalize pivot
        h,invle_part=pval.xfac()
        assert h==best_xpow
        M[rk,:]/=invle_part
        Q[rk,:]/=invle_part
        iQ[:,rk]*=invle_part
        if debug:
            print('after normalization')
            print(M,Q,iQ)

        # clear terms below the pivot
        for r in range(rk+1,nR):
            s_to_elim=M[r,c0]
            s=BPoly(s_to_elim.coeffs[best_xpow:],MOD,H)
            assert s*BPoly([0]*best_xpow+[1],MOD,H)==s_to_elim, f'{s_to_elim=} not a multiple of x**{best_xpow}'
            M[r,:]-=s*M[rk,:]
            Q[r,:]-=s*Q[rk,:]
            iQ[:,rk]+=s*iQ[:,r]
        if debug:
            print('after elimination')
            print(M,Q,iQ)
        
        good_clms.remove(c0)
        rk+=1
    return M,Q,iQ,rk

'''
==== Tensor utils ====
'''
def rotate(T,sh):
    '''
    rotate axes of T s.t. axis 0 of the result is axis sh of the input
    '''
    return np.transpose(T,axes=np.roll(np.arange(len(T.shape)),-sh))

def axis_op0(M,T):
    assert len(M.shape)==2
    return np.tensordot(M,T,axes=[(-1,),(0,)])
def axis_op(M,T,ax):
    '''
    return tensor T' such that
    T'[i_0,...,i'_ax,...,i_{D-1}] = sum_{i_ax} M[i'_ax,i_ax]*T[i_0,...,i_{D-1}]
    '''
    return rotate(axis_op0(M,rotate(T,ax)),-ax)
def axis_ops(Ms,T):
    assert len(Ms)==len(T.shape)
    for ax,M in enumerate(Ms):
        T=axis_op(M,T,ax)
    return T

def unfold(T,ax):
    T=rotate(T,ax)
    return T.reshape((T.shape[0],np.prod(T.shape[1:])))
def axis_reduce(T):
    '''
    return (T', S) such that T' is concise and T==S[0] @_0 (S[1] @_1 ... (S[D-1] @_{D-1} T') ... )
    '''
    D=len(T.shape)
    info=[]
    for ax in range(D):
        _,Q,iQ,r=border_row_reduc(unfold(T,ax))
        info.append((Q[:r,:],iQ[:,:r]))
    return axis_ops([Q for Q,_ in info],T), [S for _,S in info]

'''
==== CPD utils ====
A rank-R CPD consisting of factor matrices A_0,...,A_{D-1}
is stored as the list of tuples
[
    tuple(A_d[:,r] for d in range(D))
    for r in range(R)
]
'''
def cpd_eval(cpd,shape=None):
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
    )
def cpd_transform(Qs,cpd):
    '''
    return a CPD such that
    the d-th vector v in each tuple has been replaced with Qs[d]@v

    if Qs[d] is None, replace Qs[d]@v with v, i.e. treat Qs[d] as an identity matrix
    '''
    assert all(len(Qs)==len(vs) for vs in cpd), f'incorrect number of slice-op transformers: {Qs}'
    return [
        tuple((v if Q is None else (Q@v)) for Q,v in zip(Qs,vs))
        for vs in cpd
    ]
def cpd_rotate(cpd,sh):
    '''
    return a CPD such that
    each tuple has been rotated to the left by sh units

    satisfies the relation cpd_eval(cpd_rotate(cpd,sh)) == rotate(cpd_eval(cpd),sh)
    '''
    if len(cpd)==0:
        return cpd
    D=len(cpd[0])
    sh%=D
    return [tupl[sh:]+tupl[:sh] for tupl in cpd]

'''
==== Border CPD DFS ====
currently does not support pruners
'''
def all_border_vecs(n,MOD,H):
    return (
        np.array([BPoly(coeffs,MOD,H) for coeffs in coeffss])
        for coeffss in it.product(it.product(range(MOD),repeat=H),repeat=n)
    )
def border_vec_1hot(n,i0,MOD,H):
    assert 0<=i0 and i0<n, (n,i0)
    return np.array([BPoly(([1] if i==i0 else []),MOD,H) for i in range(n)])

def border_cpd_search(T_target,R,verbose=True):
    tmp=T_target.flatten()[0]
    MOD,H=tmp.MOD,tmp.H
    D=len(T_target.shape)
    assert D>=2
    
    stats={
        'st':time.perf_counter(),
        'elapsed':0,
        'mark':0,
        'work':[0]*(R+1),
    }
    def update_stats(force_show=False):
        stats['elapsed']=time.perf_counter()-stats['st']
        if force_show or (stats['elapsed']>=stats['mark'] and stats['mark']>0):
            print(f"t={stats['elapsed']} work={stats['work']} (tot={sum(stats['work'])})")
        while stats['elapsed']>=stats['mark']:
            stats['mark']+=10**int(np.log10(max(10,stats['mark'])))
    def dfs(T_target,R):
        if verbose:
            update_stats()
        stats['work'][R]+=1
        T_concise,S_to_raw=axis_reduce(T_target)
        if any(s==0 for s in T_concise.shape):
            return []
        if any(s>R for s in T_concise.shape):
            return None
        for tupl in it.product(*[all_border_vecs(n,MOD,H) for n in T_concise.shape]):
            ret=dfs(T_concise-cpd_eval([tupl]),R-1)
            if ret is not None:
                return cpd_transform(S_to_raw,ret+[tupl])
        return None
    
    ret=dfs(T_target,R)
    if verbose:
        update_stats(force_show=True)
    if ret is not None:
        assert np.array_equal(T_target,cpd_eval(ret)), (T_target,cpd_eval(ret))
    return ret

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
    parser = argparse.ArgumentParser(description = 'Tensor CPD (Canonical Polyadic Decomposition) DFS over rings F[x]/(x^H), where F is the integers mod a prime')
    parser.add_argument('mod', nargs=1, type=int, help='modulo of coefficient field')
    parser.add_argument('H', nargs=1, type=int, help='exponent threshold')
    parser.add_argument('rank', nargs=1, type=int, help='desired rank threshold of CPD')
    
    parser.add_argument('--shape', nargs=1, type=parse_int_tuple, help='shape of tensor to be searched over')
    parser.add_argument('--coeffs', nargs=1, type=parse_int_tuple, help='direct input: coefficients of elements of tensor, with coefficients listed little-endian and elements listed in row-major order')
    parser.add_argument('--ground', nargs=1, type=parse_int_tuple, help='implicitly multiply input by x^{H-1}: elements of original tensor listed in row-major order, where all elements must be in the field F')
    parser.add_argument('--matmul', nargs=3, type=int, help='matrix multiplication tensor <m,k,n> parameters, multiplied by x**(H-1)')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='display search progress')
    
    args=parser.parse_args()
    MOD=args.mod[0]
    assert MOD>=2 and all(MOD%i!=0 for i in range(2,MOD)), f'{MOD} is not prime'
    H=args.H[0]
    rank_thresh=args.rank[0]
    
    T_target=None
    x_pow_Hm1=BPoly([0]*(H-1)+[1], MOD, H)
    if args.shape is not None:
        shape=args.shape[0]
        if args.coeffs is not None:
            coeffs=args.coeffs[0]
            coeffs=np.array(coeffs,dtype=int).reshape((-1,H))
            elems=[BPoly(vec,MOD,H) for vec in coeffs]
            T_target=np.array(elems).reshape(shape)
        elif args.ground is not None:
            T=np.array(args.ground[0]).reshape(shape)
            if not ((T>=0)&(T<MOD)).all():
                print(f'WARNING: reducing data modulo {MOD}')
                T%=MOD
            T_target=T * x_pow_Hm1
        else:
            raise ValueError('at least one of --coeffs and --ground is required')
    else:
        assert args.matmul is not None, 'matmul parameters is required if default specification of tensor is not provided'
        T_target=MMTf(*args.matmul) * x_pow_Hm1
    print('target tensor:')
    print(T_target)
    
    cpd=border_cpd_search(T_target, rank_thresh, verbose=args.verbose)
    
    print('='*32)
    if cpd is None:
        print(f'no CPD of rank <={rank_thresh} found')
    else:
        print(f'found CPD of rank <={rank_thresh}')
        print(cpd)
