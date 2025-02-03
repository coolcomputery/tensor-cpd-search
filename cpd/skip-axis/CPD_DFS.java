import java.util.*;

class ModArith {
    public static int mod(long n, int MOD) {
        return (int)( (n%MOD + MOD)%MOD );
    }
    public static int add(int a, int b, int MOD) {
        return mod(a+(long)b,MOD);
    }
    public static int mul(int a, int b, int MOD) {
        return mod(a*(long)b,MOD);
    }
    public static int exp(int n, int k, int MOD) {
        if (k<0)
            throw new RuntimeException(""+k);
        if (k==0)
            return 1;
        int out=exp(n,k/2,MOD);
        out=mul(out,out,MOD);
        return mod((k%2==0?out:mul(out,n,MOD)),MOD);
    }
    public static int inv(int n, int MOD) {
        // for efficiency, assume MOD is prime
        return exp(n,MOD-2,MOD);
    }
    public static boolean is_prime(int n) {
        if (n<=0)
            throw new RuntimeException();
        if (n==1)
            return false;
        for (int k=2; k*(long)k<=n; k++)
            if (n%k==0)
                return false;
        return true;
    }
}

interface Tensor {
    int[] shape();
    int len(int ax);
    int ndims();
    int mod();
    int at(int[] coord);

    /**
     * @return list of elements of this in row-major order
     */
    int[] flattened();

    /**
     *
     * @param perm array
     * @return T such that axis ax of T is axis perm[ax] of this
     * @throws Exception if perm is not permutation of {0,...,ndims}
     */
    Tensor permutedims(int[] perm);

    /**
     * @param v row vector
     * @return [sum_{i_0} v[i_0]*this[i_0,...,i_{D-1} for i_1,...,i_{D-1}]
     */
    Tensor contract0(int[] v);

    /**
     * @param M tensor
     * @return T such that T[i,i_1,...] = sum_{i_0} M[i,i_0]*this[i_0,i_1,...]
     * @throws Exception if M not 2-dim or has incompatible shape with this
     */
    Tensor op0(Tensor M);

    /**
     * @return boolean (rank(this) <= 1)
     */
    boolean is_separable();

    /**
     * @return
     *  if rank(this)==0: null
     *  if rank(this)==1: a rank-1 CPD of this tensor
     * @throws Exception if this tensor is 0-dimensional
     * @throws Exception if this tensor has rank > 1
     */
    int[][] separable_factors();

    /**
     * @param weights array of ints *not necessarily* in {0,...,mod-1}
     * @param vecss list of arrays of vectors, elements *not necessarily* in {0,...,mod-1}
     * @return this + sum_i (weight[i] * outer_product(vecss.get(i)))
     * @throws Exception if any outer_product(vecss.get(i)) has different shape from this
     */
    Tensor add_weighted_outerprods(int[] weights, List<int[][]> vecss);

    class ReduceRet {
        Tensor reduced;
        Tensor[] inv_ops;
        public ReduceRet(Tensor reduced, Tensor[] inv_ops) {
            this.reduced=reduced;
            this.inv_ops=inv_ops;
        }
        public String toString() {
            return String.format("{reduced=%s inv_ops=%s",reduced,Arrays.toString(inv_ops));
        }
    }
    /**
     * @return reduced,inv_ops s.t.
     *  reduced has simultaneously minimal length among all axes
     *  axis-contracting reduced with inv_ops[ax] along axis ax, for each ax, results in this
     */
    ReduceRet reduce();

    /**
     * @return rank of this
     * @throws Exception if ndims!=2
     */
    int mat_rank();

    /**
     * @return matrix M such that M*this=I and this*M=I
     * @throws Exception if ndims!=2, shape is not square, or this is not full-rank
     */
    Tensor mat_inv();

    /**
     * @return an arbitrary matrix M such that
     *  M@this=0
     *  M has maximal rank
     * @throws Exception if ndims!=2
     */
    Tensor left_nullmat();

    // helper methods for element indexing
    static int prod(int[] A) {
        for (int v:A)
            if (v==0)
                return 0;

        // test for overflow
        int test=Integer.MAX_VALUE;
        for (int v:A)
            test/=Math.abs(v);
        if (test==0)
            throw new RuntimeException(String.format("abs(prod(%s)) > Integer.MAX_VALUE",Arrays.toString(A)));

        int out=1;
        for (int v:A)
            out*=v;
        return out;
    }
    static int[] idx2coord(int[] shape, int i) {
        if (i<0)
            throw new RuntimeException(Arrays.toString(shape)+" "+i);
        int[] out=new int[shape.length];
        int h=i;
        for (int ax=shape.length-1; ax>=0; ax--) {
            out[ax]=h%shape[ax];
            h/=shape[ax];
        }
        if (h>0)
            throw new RuntimeException(Arrays.toString(shape)+" "+i);
        return out;
    }
    static int coord2idx(int[] shape, int[] coord) {
        if (coord.length!=shape.length)
            throw new RuntimeException(String.format("shape=%s coord=%s",Arrays.toString(shape),Arrays.toString(coord)));
        for (int ax=0; ax<shape.length; ax++) {
            int c=coord[ax], len=shape[ax];
            if (c<0 || c>=len)
                throw new RuntimeException(Arrays.toString(shape)+" "+Arrays.toString(coord));
        }
        int out=0;
        for (int ax=0; ax<shape.length; ax++)
            out=out*shape[ax]+coord[ax];
        return out;
    }
    static int[] rotate_arr(int[] A, int sh) {
        int[] out=new int[A.length];
        for (int ax=0; ax<A.length; ax++)
            out[ax]=A[ModArith.add(ax,sh,A.length)];
        return out;
    }
    static int[] permute_arr(int[] A, int[] perm) {
        if (perm.length!=A.length)
            throw new RuntimeException();
        for (int v:perm)
            if (v<0 || v>=A.length)
                throw new RuntimeException();
        boolean[] seen=new boolean[A.length];
        for (int v:perm)
            seen[v]=true;
        for (boolean b:seen)
            if (!b)
                throw new RuntimeException();

        int[] out=new int[A.length];
        for (int i=0; i<A.length; i++)
            out[i]=A[perm[i]];
        return out;
    }
    static int[] mat_clm(Tensor M, int j) {
        if (M.ndims()!=2)
            throw new RuntimeException("not 2-dim: "+Arrays.toString(M.shape()));
        int R=M.len(0);
        int[] out=new int[R];
        for (int i=0; i<R; i++)
            out[i]=M.at(new int[] {i,j});
        return out;
    }
    static int[] flatten(int[][] M) {
        int R=M.length;
        if (R==0)
            return new int[0];
        int C=M[0].length;
        for (int[] row:M)
            if (row.length!=C)
                throw new RuntimeException();
        int[] out=new int[R*C];
        for (int r=0; r<R; r++)
            System.arraycopy(M[r],0,out,r*C,C);
        return out;
    }
    static Tensor make_tensor(int[] shape, int[] data, int MOD) {
        return (
                MOD==2?new Mod2Tensor(shape,data):
                        new GenTensor(shape,data,MOD)
        );
    }
}
class GenTensor implements Tensor {
    private final int[] shape, data;
    private final int ndims, MOD;
    public GenTensor(int[] shape, int[] data, int MOD) {
        for (int len:shape)
            if (len<0)
                throw new RuntimeException();
        int size=Tensor.prod(shape);
        if (data.length!=size)
            throw new RuntimeException(String.format("%d != prod(%s) = %d",data.length,Arrays.toString(shape),size));

        if (!ModArith.is_prime(MOD))
            throw new RuntimeException(MOD+" is not prime");
        this.MOD=MOD;
        this.ndims=shape.length;
        this.shape=Arrays.copyOf(shape,ndims);
        this.data=new int[size];
        for (int i=0; i<size; i++)
            this.data[i]=ModArith.mod(data[i],MOD);
    }
    public String toString() {
        return String.format("{%s : %s mod %d}",Arrays.toString(shape),Arrays.toString(data),MOD);
    }

    public int[] shape() {
        return Arrays.copyOf(shape,ndims);
    }
    public int len(int ax) {
        return shape[ax];
    }
    public int ndims() {
        return ndims;
    }
    public int mod() {
        return MOD;
    }
    public int at(int[] coord) {
        return data[Tensor.coord2idx(shape,coord)];
    }
    public int[] flattened() {
        return Arrays.copyOf(data,data.length);
    }
    public GenTensor permutedims(int[] perm) {
        int[] nshape=Tensor.permute_arr(shape,perm);
        int[] ndata=new int[data.length];
        for (int i=0; i<data.length; i++)
            ndata[Tensor.coord2idx(nshape,Tensor.permute_arr(Tensor.idx2coord(shape,i),perm))]=data[i];
        return new GenTensor(nshape,ndata,MOD);
    }
    private GenTensor rotate(int sh) {
        // permute dimensions so that axis ax of the output is axis (ax+sh)%ndims of the input
        sh=ModArith.mod(sh,ndims);
        if (sh==0)
            return this;
        int[] nshape=Tensor.rotate_arr(shape,sh);
        int[] ndata=new int[data.length];
        for (int i=0; i<data.length; i++)
            ndata[Tensor.coord2idx(nshape,Tensor.rotate_arr(Tensor.idx2coord(shape,i),sh))]=data[i];
        return new GenTensor(nshape,ndata,MOD);
    }
    public GenTensor contract0(int[] v) {
        if (ndims==0)
            throw new RuntimeException();
        if (v.length!=shape[0])
            throw new RuntimeException(v.length+"!="+shape[0]);

        int n0=shape[0];
        int[] nshape=Arrays.copyOfRange(shape,1,ndims);
        int L=Tensor.prod(nshape);
        int[] out=new int[L];
        for (int i0=0; i0<n0; i0++)
            for (int i_rem=0; i_rem<L; i_rem++)
                out[i_rem]=ModArith.add(out[i_rem],ModArith.mul(v[i0],data[i0*L+i_rem],MOD),MOD);
        return new GenTensor(nshape,out,MOD);
    }
    public GenTensor add_weighted_outerprods(int[] weights, List<int[][]> vecss) {
        if (weights.length!=vecss.size())
            throw new RuntimeException();
        int[] ndata=Arrays.copyOf(data,data.length);

        for (int pi=0; pi<vecss.size(); pi++) {
            int[][] vecs=vecss.get(pi);
            int weight=weights[pi];
            if (vecs.length!=ndims)
                throw new RuntimeException(vecs.length+"!="+ndims);
            for (int ax=0; ax<ndims; ax++)
                if (vecs[ax].length!=shape[ax])
                    throw new RuntimeException();

            for (int i=0; i<data.length; i++) {
                int[] coord=Tensor.idx2coord(shape,i);
                int prod=ModArith.mod(weight,MOD);
                for (int ax=0; ax<ndims; ax++)
                    prod=ModArith.mul(prod,vecs[ax][coord[ax]],MOD);
                ndata[i]=ModArith.add(ndata[i],prod,MOD);
            }
        }
        return new GenTensor(shape,ndata,MOD);
    }

    /*
    row operation helper methods
    assume data has a matrix shape (i.e. is not a jagged array)
     */
    private static void swap_rows(int[][] data, int r0, int r1) {
        int[] row0=Arrays.copyOf(data[r0],data[r0].length);
        int[] row1=Arrays.copyOf(data[r1],data[r1].length);
        data[r0]=row1;
        data[r1]=row0;
    }
    private static void scale_row(int[][] data, int r, int v, int MOD) {
        int n=data[r].length;
        for (int c=0; c<n; c++)
            data[r][c]=ModArith.mul(v,data[r][c],MOD);
    }
    private static void shear_rows(int[][] data, int r0, int r1, int v, int MOD) {
        int n=data[r1].length;
        for (int c=0; c<n; c++)
            data[r1][c]=ModArith.add(data[r1][c],ModArith.mul(v,data[r0][c],MOD),MOD);
    }
    private static int[][] eye(int n) {
        int[][] out=new int[n][n];
        for (int i=0; i<n; i++)
            out[i][i]=1;
        return out;
    }
    static class RrefRet {
        int[][] rref, left, ileftT;
        int rk;
        public RrefRet(int[][] rref, int[][] left, int[][] ileftT, int rk) {
            this.rref=rref;
            this.left=left;
            this.ileftT=ileftT;
            this.rk=rk;
        }
    }
    /**
     * @return
     *  if rank(this) <= max_rank: rref,left,ileftT,rk s.t. left@this=rref, ileftT is transpose of inverse of left, and rk=rank(this)
     *  else: null
     */
    private RrefRet row_reduce(int max_rank) {
        if (ndims!=2)
            throw new RuntimeException(String.format("must be matrix, not shape %s",Arrays.toString(shape)));
        if (max_rank<0)
            return null;
        int R=shape[0], C=shape[1];
        int[][] rref=new int[R][];
        for (int r=0; r<R; r++)
            rref[r]=Arrays.copyOfRange(data,r*C,(r+1)*C);
        int[][] left=eye(R), ileftT=eye(R);
        int leadr=0;
        for (int c=0; c<C && leadr<R; c++) {
            {  // swap a valid row to the leadr-th position
                int nr=-1;
                for (int r=leadr; r<R; r++) {
                    if (rref[r][c]!=0) {
                        nr=r;
                        break;
                    }
                }
                if (nr==-1)
                    continue;
                swap_rows(rref,leadr,nr);
                swap_rows(left,leadr,nr);
                swap_rows(ileftT,leadr,nr);
            }
            {  //
                int v=rref[leadr][c], iv=ModArith.inv(v,MOD);
                scale_row(rref,leadr,iv,MOD);
                scale_row(left,leadr,iv,MOD);
                scale_row(ileftT,leadr,v,MOD);
            }

            for (int r=0; r<R; r++) {
                if (r!=leadr) {
                    int v=rref[r][c], mv=ModArith.mod(-v,MOD);
                    shear_rows(rref,leadr,r,mv,MOD);
                    shear_rows(left,leadr,r,mv,MOD);
                    shear_rows(ileftT,r,leadr,v,MOD);
                }
            }
            leadr++;
            if (leadr>max_rank)
                return null;
        }
        return new RrefRet(
                rref,
                left,
                ileftT,
                leadr
        );
    }
    private RrefRet row_reduce() {
        return row_reduce(Integer.MAX_VALUE);
    }
    public int mat_rank() {
        return row_reduce().rk;
    }
    public GenTensor left_nullmat() {
        RrefRet ret=row_reduce();
        return new GenTensor(
                new int[] {shape[0]-ret.rk,shape[0]},
                Tensor.flatten(Arrays.copyOfRange(ret.left,ret.rk,shape[0])),
                MOD
        );
    }
    private GenTensor unfold0() {
        return new GenTensor(
                new int[] {shape[0],Tensor.prod(Arrays.copyOfRange(shape,1,ndims))},
                data,
                MOD
        );
    }
    public GenTensor mat_inv() {
        if (ndims!=2 || shape[0]!=shape[1])
            throw new RuntimeException(String.format("must be square matrix, not have shape %s",Arrays.toString(shape)));
        int R=shape[0];
        RrefRet ret=row_reduce();
        if (ret.rk!=R)
            throw new RuntimeException("matrix not invertible: "+this);
        return new GenTensor(new int[] {R,R},Tensor.flatten(ret.left),MOD);
    }

    public GenTensor op0(Tensor op_T) {
        if (ndims==0)
            throw new RuntimeException();
        if (op_T.mod()!=MOD)
            throw new RuntimeException();
        if (op_T.ndims()!=2)
            throw new RuntimeException();
        int R=op_T.len(0), C=op_T.len(1);
        if (C!=shape[0])
            throw new RuntimeException();
        int[][] op_data=new int[R][C];
        for (int r=0; r<R; r++)
            for (int c=0; c<C; c++)
                op_data[r][c]=op_T.at(new int[] {r,c});
        int[][] out=new int[R][];
        for (int r=0; r<R; r++)
            out[r]=contract0(op_data[r]).data;
        int[] nshape=Arrays.copyOf(shape,ndims);
        nshape[0]=R;
        return new GenTensor(nshape,Tensor.flatten(out),MOD);
    }
    public ReduceRet reduce() {
        if (ndims==0)
            throw new RuntimeException();
        Tensor[] ops=new Tensor[ndims], inv_ops=new Tensor[ndims];
        for (int ax=0; ax<ndims; ax++) {
            int n=shape[ax];
            RrefRet reduc=rotate(ax).unfold0().row_reduce();
            ops[ax]=new GenTensor(
                    new int[] {reduc.rk,n},
                    Tensor.flatten(Arrays.copyOf(reduc.left,reduc.rk)),
                    MOD
            );
            inv_ops[ax]=new GenTensor(
                    new int[] {reduc.rk,n},
                    Tensor.flatten(Arrays.copyOf(reduc.ileftT,reduc.rk)),
                    MOD
            ).rotate(1);  // transpose
        }
        GenTensor out=this;
        for (int ax=0; ax<ndims; ax++)
            out=out.op0(ops[ax]).rotate(1);
        return new ReduceRet(out,inv_ops);
    }
    public boolean is_separable() {
        if (ndims==0)
            throw new RuntimeException();
        for (int ax=0; ax<ndims; ax++)
            if (rotate(ax).unfold0().row_reduce(1)==null)
                return false;
        return true;
    }
    public int[][] separable_factors() {
        // only called when a CPD is called, so does not have to be efficient
        if (ndims==0)
            throw new RuntimeException();
        ReduceRet reduc=reduce();
        for (int ax=0; ax<ndims; ax++)
            if (reduc.reduced.len(ax)>1)
                throw new RuntimeException();
        if (reduc.reduced.len(0)==0)
            return null;
        // reduc.reduced must have shape 1x...x1
        int scale_v=reduc.reduced.at(new int[ndims]);
        int[][] tup=new int[ndims][];
        tup[0]=VecUtils.scaled_vec(scale_v,Tensor.mat_clm(reduc.inv_ops[0],0),MOD);
        for (int ax=1; ax<ndims; ax++)
            tup[ax]=Tensor.mat_clm(reduc.inv_ops[ax],0);
        return tup;
    }
}

class Mod2Tensor implements Tensor {
    private int[] shape, data;
    private int ndims, slice0_size;
    private final int MOD=2;
    public Mod2Tensor(int[] shape, int[] data) {
        for (int len:shape)
            if (len<0)
                throw new RuntimeException();
        int size=Tensor.prod(shape);
        if (data.length!=size)
            throw new RuntimeException(String.format("%d != prod(%s) = %d",data.length,Arrays.toString(shape),size));

        this.ndims=shape.length;
        this.slice0_size=Tensor.prod(Arrays.copyOfRange(shape,1,ndims));
        this.shape=Arrays.copyOf(shape,ndims);
        this.data=new int[size];
        for (int i=0; i<size; i++)
            this.data[i]=data[i]&1;
    }

    /**
     * private constructor that avoids deep-copying shape and data
     *  assume shape and data satisfy all constraints that are required in the public constructor
     * @param shape shape of tensor
     * @param data elements of tensor in row-major order
     * @param ref_copy unused param that makes this constructor have a different signature than the public constructor
     */
    private Mod2Tensor(int[] shape, int[] data, boolean ref_copy) {
        this.ndims=shape.length;
        this.slice0_size=Tensor.prod(Arrays.copyOfRange(shape,1,ndims));
        this.shape=shape;
        this.data=data;
    }
    public String toString() {
        return String.format("{optimized %s : %s mod %d}",Arrays.toString(shape),Arrays.toString(data),MOD);
    }

    public int[] shape() {
        return Arrays.copyOf(shape,ndims);
    }
    public int len(int ax) {
        return shape[ax];
    }
    public int ndims() {
        return ndims;
    }
    public int mod() {
        return MOD;
    }
    public int at(int[] coord) {
        return data[Tensor.coord2idx(shape,coord)];
    }
    public int[] flattened() {
        return Arrays.copyOf(data,data.length);
    }
    public Mod2Tensor permutedims(int[] perm) {
        int[] nshape=Tensor.permute_arr(shape,perm);
        int[] ndata=new int[data.length];
        for (int i=0; i<data.length; i++)
            ndata[Tensor.coord2idx(nshape,Tensor.permute_arr(Tensor.idx2coord(shape,i),perm))]=data[i];
        return new Mod2Tensor(nshape,ndata);
    }
    private Mod2Tensor rotate(int sh) {
        if (ndims==0)
            throw new RuntimeException();
        sh=ModArith.mod(sh,ndims);
        if (sh==0)
            return this;
        int sz_l=Tensor.prod(Arrays.copyOfRange(shape,0,sh)),
                sz_r=Tensor.prod(Arrays.copyOfRange(shape,sh,ndims));
        int[] ndata=new int[data.length];
        for (int i_l=0; i_l<sz_l; i_l++)
            for (int i_r=0; i_r<sz_r; i_r++)
                ndata[i_r*sz_l+i_l]=data[i_l*sz_r+i_r];
        int[] nshape=Tensor.rotate_arr(shape,sh);
        return new Mod2Tensor(nshape,ndata,true);
    }
    public Mod2Tensor contract0(int[] v) {
        if (ndims==0)
            throw new RuntimeException();
        if (v.length!=shape[0])
            throw new RuntimeException(v.length+"!="+shape[0]);

        int n0=shape[0];
        int[] nshape=Arrays.copyOfRange(shape,1,ndims);
        int L=slice0_size;
        int[] out=new int[L];
        for (int i0=0; i0<n0; i0++)
            for (int i_rem=0; i_rem<L; i_rem++)
                out[i_rem]^=(v[i0]&1)&data[i0*L+i_rem];
        return new Mod2Tensor(nshape,out,true);
    }
    public Mod2Tensor op0(Tensor op_T) {
        if (ndims==0)
            throw new RuntimeException();
        if (op_T.mod()!=MOD)
            throw new RuntimeException();
        if (op_T.ndims()!=2)
            throw new RuntimeException();
        int R=op_T.len(0), C=op_T.len(1);
        if (C!=shape[0])
            throw new RuntimeException();
        int[][] op_data=new int[R][C];
        for (int r=0; r<R; r++)
            for (int c=0; c<C; c++)
                op_data[r][c]=op_T.at(new int[] {r,c});
        int[][] out=new int[R][];
        for (int r=0; r<R; r++)
            out[r]=contract0(op_data[r]).data;
        int[] nshape=Arrays.copyOf(shape,ndims);
        nshape[0]=R;
        return new Mod2Tensor(nshape,Tensor.flatten(out));
    }

    public Mod2Tensor add_weighted_outerprods(int[] weights, List<int[][]> vecss) {
        if (weights.length!=vecss.size())
            throw new RuntimeException();
        int[] ndata=Arrays.copyOf(data,data.length);
        for (int pi=0; pi<vecss.size(); pi++) {
            int[][] vecs=vecss.get(pi);
            int weight=weights[pi];
            if (vecs.length!=ndims)
                throw new RuntimeException(vecs.length+"!="+ndims);
            for (int ax=0; ax<ndims; ax++)
                if (vecs[ax].length!=shape[ax])
                    throw new RuntimeException();
            for (int i=0; i<data.length; i++) {
                int prod=weight&1;
                for (int ax=ndims-1, h=i; ax>=0; h/=shape[ax], ax--)
                    prod&=vecs[ax][h%shape[ax]];
                ndata[i]^=prod;
            }
        }
        return new Mod2Tensor(shape,ndata,true);
    }

    /*
    row operation helper methods
    assume data has a matrix shape (i.e. is not a jagged array)
     */
    private static void swap_rows(int[][] data, int r0, int r1) {
        int[] row0=Arrays.copyOf(data[r0],data[r0].length);
        int[] row1=Arrays.copyOf(data[r1],data[r1].length);
        data[r0]=row1;
        data[r1]=row0;
    }
    private static void shear_rows(int[][] data, int r0, int r1, int v) {
        v&=1;
        int n=data[r1].length;
        for (int c=0; c<n; c++)
            data[r1][c]^=v&data[r0][c];
    }
    private static int[][] eye(int n) {
        int[][] out=new int[n][n];
        for (int i=0; i<n; i++)
            out[i][i]=1;
        return out;
    }
    static class RrefRet {
        int[][] reduced, left, ileftT;
        int rk;
        public RrefRet(int[][] reduced, int[][] left, int[][] ileftT, int rk) {
            this.reduced=reduced;
            this.left=left;
            this.ileftT=ileftT;
            this.rk=rk;
        }
    }
    /**
     * @return
     *  if rank(this) <= max_rank: rref,left,ileftT,rk s.t. left@this=rref, ileftT is transpose of inverse of left, and rk=rank(this)
     *  else: null
     */
    private RrefRet reduce0() {
        if (ndims<1)
            throw new RuntimeException();
        int R=shape[0], C=slice0_size;
        int[][] reduced=new int[R][];
        for (int r=0; r<R; r++)
            reduced[r]=Arrays.copyOfRange(data,r*C,(r+1)*C);
        int[][] left=eye(R), ileftT=eye(R);
        int leadr=0;
        for (int c=0; c<C && leadr<R; c++) {
            {  // swap a valid row to the leadr-th position
                int nr=-1;
                for (int r=leadr; r<R; r++) {
                    if (reduced[r][c]!=0) {
                        nr=r;
                        break;
                    }
                }
                if (nr==-1)
                    continue;
                swap_rows(reduced,leadr,nr);
                swap_rows(left,leadr,nr);
                swap_rows(ileftT,leadr,nr);
            }

            for (int r=0; r<R; r++) {
                if (r!=leadr) {
                    int v=reduced[r][c];
                    shear_rows(reduced,leadr,r,v);
                    shear_rows(left,leadr,r,v);
                    shear_rows(ileftT,r,leadr,v);
                }
            }
            leadr++;
        }
        return new RrefRet(
                reduced,
                left,
                ileftT,
                leadr
        );
    }
    private int rank0() {
        if (ndims<1)
            throw new RuntimeException();
        int R=shape[0], C=slice0_size;
        int[][] rref=new int[R][];
        for (int r=0; r<R; r++)
            rref[r]=Arrays.copyOfRange(data,r*C,(r+1)*C);
        int leadr=0;
        for (int c=0; c<C && leadr<R; c++) {
            {  // swap a valid row to the leadr-th position
                int nr=-1;
                for (int r=leadr; r<R; r++) {
                    if (rref[r][c]!=0) {
                        nr=r;
                        break;
                    }
                }
                if (nr==-1)
                    continue;
                swap_rows(rref,leadr,nr);
            }

            for (int r=leadr+1; r<R; r++) {
                int v=rref[r][c];
                shear_rows(rref,leadr,r,v);
            }
            leadr++;
        }
        return leadr;
    }
    public int mat_rank() {
        if (ndims!=2)
            throw new RuntimeException(String.format("must be matrix, not shape %s",Arrays.toString(shape)));
        return rank0();
    }
    public Mod2Tensor mat_inv() {
        if (ndims!=2 || shape[0]!=shape[1])
            throw new RuntimeException(String.format("must be square matrix, not have shape %s",Arrays.toString(shape)));
        int R=shape[0];
        RrefRet ret=reduce0();
        if (ret.rk!=R)
            throw new RuntimeException("matrix not invertible: "+this);
        return new Mod2Tensor(new int[] {R,R},Tensor.flatten(ret.left),true);
    }
    public GenTensor left_nullmat() {
        RrefRet ret=reduce0();
        return new GenTensor(
                new int[] {shape[0]-ret.rk,shape[0]},
                Tensor.flatten(Arrays.copyOfRange(ret.left,ret.rk,shape[0])),
                MOD
        );
    }

    public ReduceRet reduce() {
        if (ndims==0)
            throw new RuntimeException();
        Tensor[] inv_ops=new Tensor[ndims];
        Mod2Tensor out=this;
        for (int ax=0; ax<ndims; ax++) {
            int n=shape[ax];
            RrefRet reduc=out.reduce0();
            if (reduc.rk==n) {
                // no need to apply an invertible axis-op, just use the identity
                inv_ops[ax]=null;
                out=out.rotate(1);
            }
            else {
                inv_ops[ax]=new Mod2Tensor(
                        new int[] {reduc.rk,n},
                        Tensor.flatten(Arrays.copyOf(reduc.ileftT,reduc.rk))
                ).rotate(1);  // transpose
                int[] nshape=Arrays.copyOf(out.shape,ndims);
                nshape[0]=reduc.rk;
//                System.out.println(Arrays.toString(nshape));
                out=new Mod2Tensor(
                        nshape,
                        Tensor.flatten(Arrays.copyOf(reduc.reduced,reduc.rk)),
                        true
                ).rotate(1);
            }
        }
        return new ReduceRet(out,inv_ops);
    }
    public boolean is_separable() {
        if (ndims==0)
            throw new RuntimeException();
        if (data.length==0)
            return true;  // a zero-sized tensor has rank zero
        int i_principal=-1;
        for (int i=0; i<data.length; i++)
            if (data[i]!=0) {
                i_principal=i;
                break;
            }
        if (i_principal==-1)
            return true; // all zeros
        for (int ax=ndims-1, h=i_principal, sz_rem=1; ax>=0; h/=shape[ax], sz_rem*=shape[ax], ax--) {
            int sz_top=(data.length/sz_rem)/shape[ax];
            int i_ax_principal=h%shape[ax];  // coordinate of i_principal along axis ax
            for (int i_ax=0; i_ax<shape[ax]; i_ax++) {
                boolean all_zero=true, eq_principal=true;
                for (int i_top=0; i_top<sz_top; i_top++)
                    for (int i_rem=0; i_rem<sz_rem; i_rem++) {
                        int v=data[(i_top*shape[ax]+i_ax)*sz_rem+i_rem];
                        all_zero&=v==0;
                        eq_principal&=v==data[(i_top*shape[ax]+i_ax_principal)*sz_rem+i_rem];
                        if (!all_zero && !eq_principal)
                            return false;
                    }
            }
        }
        return true;

        // too slow, does not allow early termination
//        int[][] facs=new int[ndims][];
//        for (int ax=0; ax<ndims; ax++)
//            facs[ax]=new int[shape[ax]];
//        for (int i=0; i<data.length; i++)
//            for (int ax=ndims-1, h=i; ax>=0; h/=shape[ax], ax--)
//                facs[ax][h%shape[ax]]|=data[i];
//        for (int i=0; i<data.length; i++) {
//            int prod=1;
//            for (int ax=ndims-1, h=i; ax>=0; h/=shape[ax], ax--)
//                prod&=facs[ax][h%shape[ax]];
//            if (data[i]!=prod)
//                return false;
//        }
//        return true;
    }
    public int[][] separable_factors() {
        // only called when a CPD is found, so does not have to be efficient
        if (ndims==0)
            throw new RuntimeException();
        ReduceRet reduc=reduce();
        for (int ax=0; ax<ndims; ax++)
            if (reduc.reduced.len(ax)>1)
                throw new RuntimeException();
        if (reduc.reduced.len(0)==0)
            return null;
        // reduc.reduced must have shape 1 x ... x 1
        int scale_v=reduc.reduced.at(new int[ndims]);
        int[][] tup=new int[ndims][];
        int[] I1={1};
        tup[0]=VecUtils.scaled_vec_mod2(
                scale_v,
                (
                        reduc.inv_ops[0]==null?
                                I1:
                                Tensor.mat_clm(reduc.inv_ops[0],0)
                )
        );
        for (int ax=1; ax<ndims; ax++)
            tup[ax]=(
                    reduc.inv_ops[ax]==null?
                            I1:
                            Tensor.mat_clm(reduc.inv_ops[ax],0)
            );
        return tup;
    }
}

class CPDUtils {
    /**
     * matrix-vector multiplication
     *  only used in dfs() when a CPD has been found,
     *  so this method does not have to be efficient
     * @param M tensor
     * @param v vector
     * @return M@v
     * @throws Exception if M is not 2-dim
     */
    private static int[] mv(Tensor M, int[] v) {
        if (M==null)
            return v;
        if (M.ndims()!=2)
            throw new RuntimeException();
        if (v.length!=M.len(1))
            throw new RuntimeException("incompatible shapes for mat@vec: "+Arrays.toString(M.shape())+" "+v.length);
        Tensor ret=Tensor.make_tensor(new int[] {v.length,1},v,M.mod()).op0(M);
        int R=M.len(0);
        int[] out=new int[R];
        for (int i=0; i<R; i++)
            out[i]=ret.at(new int[] {i,0});
        return out;
    }

    /**
     * axis-operations on a CPD
     *  only used in dfs() when a CPD has been found,
     *  so this method does not have to be efficient
     */
    public static List<int[][]> ops(List<int[][]> cpd, Tensor[] ops) {
        int ndims=ops.length;
        List<int[][]> out=new ArrayList<>();
        for (int[][] tup:cpd) {
            if (tup.length!=ndims)
                throw new RuntimeException();
            int[][] ntup=new int[ndims][];
            for (int ax=0; ax<ndims; ax++)
                ntup[ax]=mv(ops[ax],tup[ax]);
            out.add(ntup);
        }
        return out;
    }
}

class VecUtils {
    public static int[] scaled_vec(int s, int[] v, int MOD) {
        int[] out=new int[v.length];
        for (int i=0; i<v.length; i++)
            out[i]=ModArith.mul(s,v[i],MOD);
        return out;
    }
    public static int[] scaled_vec_mod2(int s, int[] v) {
        int[] out=new int[v.length];
        for (int i=0; i<v.length; i++)
            out[i]=(s&v[i])&1;
        return out;
    }

    public static int[] init_vec(int n) {
        return new int[n];
    }

    /**
     * @param vec vector of integers {0,...,MOD-1}
     * @param MOD modulo
     * @return lexicographically next vector of integers {0,...,MOD-1}, null if no such vector exists
     */
    public static int[] next_vec(int[] vec, int MOD) {
        for (int v:vec)
            if (v<0 || v>=MOD)
                throw new RuntimeException();
        if (vec.length==0)
            return null;
        int[] out=Arrays.copyOf(vec,vec.length);
        out[vec.length-1]++;
        for (int i=vec.length-1; i>=0 && out[i]==MOD; i--) {
            if (i==0)
                return null;
            out[i]=0;
            out[i-1]++;
        }
        return out;
    }

    public static int[] init_normalized_vec(int n) {
        if (n==0)
            throw new RuntimeException();
        int[] out=new int[n];
        out[n-1]=1;
        return out;
    }
    /**
     * @param vec vector of integers {0,...,MOD-1}
     * @param MOD modulo
     * @return lexicographically next normalized vector of integers {0,...,MOD-1}, null if no such vector exists
     * @throws Exception if vec not normalized
     */
    public static int[] next_normalized_vec(int[] vec, int MOD) {
        if (vec.length==0)
            throw new RuntimeException();
        int[] out=Arrays.copyOf(vec,vec.length);
        int lead=0;
        while (lead<vec.length && out[lead]==0)
            lead++;
        if (lead>=vec.length || out[lead]!=1)
            throw new RuntimeException("not normalized: vec="+Arrays.toString(vec));

        out[vec.length-1]++;
        for (int i=vec.length-1; i>=0; i--) {
            if (out[i]==MOD || (i==lead && out[i]!=1)) {
                if (i==0)
                    return null;
                out[i]=0;
                out[i-1]++;
            }
            else
                break;
        }
        return out;
    }

    public static int[][] init_normalized_vec_tuple(int[] shape) {
        int ndims=shape.length;
        int[][] tup=new int[ndims][];
        for (int ax=0; ax<ndims; ax++)
            tup[ax]=VecUtils.init_normalized_vec(shape[ax]);
        return tup;
    }

    /**
     * increment to lexicographically next highest vector tuple
     *  where all vectors are normalized
     * @param tup vector tuple to increment (mutated)
     * @param shape list of lengths of each vector
     * @param MOD modulo
     */
    public static void mut_incr_normalized_vec_tuple(int[][] tup, int[] shape, int MOD) {
        int ndims=shape.length;
        tup[ndims-1]=VecUtils.next_normalized_vec(tup[ndims-1],MOD);
        for (int ax=ndims-1; ax>0 && tup[ax]==null; ax--) {
            tup[ax]=VecUtils.init_normalized_vec(shape[ax]);
            tup[ax-1]=VecUtils.next_normalized_vec(tup[ax-1],MOD);
        }
    }
}

/**
 * CPD search algorithm
 * each item partial_tups.get(r) corresponds to the column tuple ((Y_d)_{:,r})_{d\ne 0}
 */
public class CPD_DFS {
    private static final long SEC_TO_NANO=1000_000_000;
    private static long tick, time_st, time_mark;
    private static long[] work;
    public static long[] work_stats() {
        return Arrays.copyOf(work,work.length);
    }

    private static void log_progress(boolean force_print) {
        long time_elapsed=System.nanoTime()-time_st;
        if (time_elapsed>time_mark || force_print) {
            if (time_mark>0 || force_print) {
                System.out.printf("t=%.3f work=%s\n",time_elapsed/(float)SEC_TO_NANO,Arrays.toString(work));
            }
            while (time_elapsed>time_mark)
                time_mark+=Math.pow(10,(int)(Math.log10(Math.max(10,time_mark/(float)SEC_TO_NANO))))*SEC_TO_NANO;
        }
    }

    private static Tensor derived_slice(Tensor T, List<int[][]> partial_tups, int[] v, int[] c) {
        if (c.length!=partial_tups.size())
            throw new RuntimeException();
        int[] weights=new int[c.length];
        for (int i=0; i<c.length; i++)
            weights[i]=-c[i];
        return T.contract0(v).add_weighted_outerprods(weights,partial_tups);
    }
    private static int[] flattened_outerprod(int[] shape, int weight, int[][] tups, int MOD) {
        Tensor tmp=Tensor.make_tensor(shape,new int[Tensor.prod(shape)],MOD);
        tmp=tmp.add_weighted_outerprods(new int[] {weight}, Collections.singletonList(tups));
        return tmp.flattened();
    }

    // could be implemented as an iterator or generator,
    //  but doing so in Java is annoying and would probably incur high constant factors
    private static List<int[][]> good_pairs(Tensor T, int R, List<int[][]> partial_tups) {
        int ndims=T.ndims(), MOD=T.mod();
        int[] sh=T.shape();
        int n0=sh[0];
        List<int[][]> out=new ArrayList<>();
        int cnt=0;
        for (int ax=2; ax<ndims; ax++)
            cnt+=sh[ax];
        if (R<=cnt) {
            for (int[] v=VecUtils.init_normalized_vec(n0); v!=null; v=VecUtils.next_normalized_vec(v,MOD))
                for (int[] c=VecUtils.init_vec(R-n0); c!=null; c=VecUtils.next_vec(c,MOD)) {
                    Tensor Ts=derived_slice(T,partial_tups,v,c);
                    if (Ts.is_separable())
                        out.add(new int[][] {v,c});
                }
        }
        else {
            int[] sh_1=Arrays.copyOfRange(sh,1,ndims), sh_2=Arrays.copyOfRange(sh,2,ndims);
            int vol_1=Tensor.prod(sh_1), vol_2=Tensor.prod(sh_2);
            int[][] T_slices_flattened=new int[sh[0]][vol_1]; {
                int[] T_flattened=T.flattened();
                for (int i0=0; i0<sh[0]; i0++)
                    System.arraycopy(T_flattened,i0*vol_1,T_slices_flattened[i0],0,vol_1);
            }
            int[][] partial_tup_prods_flattened=new int[partial_tups.size()][];
            for (int pi=0; pi<partial_tups.size(); pi++)
                partial_tup_prods_flattened[pi]=flattened_outerprod(sh_1,-1,partial_tups.get(pi),MOD);

            for (
                    int[][] rk1_facs=VecUtils.init_normalized_vec_tuple(sh_2);
                    rk1_facs[0]!=null;
                    VecUtils.mut_incr_normalized_vec_tuple(rk1_facs,sh_2,MOD)
            ) {
                // construct matrix s.t. [v | c | u] @ M = flattened(v @_0 T - [[c,Y_1,...,Y_{D-1}]] - u x u_2 x ... x u_{D-1})
                int[] rk1_prod_flattened=flattened_outerprod(sh_2,-1,rk1_facs,MOD);
                int[][] M=new int[sh[0]+partial_tups.size()+sh[1]][];
                System.arraycopy(T_slices_flattened,0,M,0,sh[0]);
                System.arraycopy(partial_tup_prods_flattened,0,M,sh[0],partial_tups.size());
                for (int i1=0; i1<sh[1]; i1++) {
                    M[sh[0]+partial_tups.size()+i1]=new int[vol_1];
                    System.arraycopy(rk1_prod_flattened,0,M[sh[0]+partial_tups.size()+i1],i1*vol_2,vol_2);
                }

                Tensor M_tensor=Tensor.make_tensor(new int[] {M.length,vol_1},Tensor.flatten(M),MOD);
                Tensor ker=M_tensor.left_nullmat();

                int[] ker_data=ker.flattened();
                for (int ki=0; ki<ker.len(0); ki++) {
                    int[] ker_row=Arrays.copyOfRange(ker_data,ki*M.length,(ki+1)*M.length);
                    out.add(new int[][] {Arrays.copyOfRange(ker_row,0,sh[0]),Arrays.copyOfRange(ker_row,sh[0],sh[0]+partial_tups.size())});
                }
            }
        }
        return out;
    }
    private static List<int[][]> dfs_base(Tensor T, List<int[][]> partial_tups) {
        int ndims=T.ndims(), MOD=T.mod();
        int n0=T.len(0);
        int R=n0+partial_tups.size();
        int[][] Q_rows=new int[n0][], C_rows=new int[n0][];
        int nrows=0;
        for (int[][] pair:good_pairs(T,R,partial_tups)) {
            int[] Q_row=pair[0], C_row=pair[1];
            int[][] nQ_rows=new int[nrows+1][];
            System.arraycopy(Q_rows,0,nQ_rows,0,nrows);
            nQ_rows[nrows]=Q_row;
            if (Tensor.make_tensor(new int[] {nrows+1,n0},Tensor.flatten(nQ_rows),MOD).mat_rank() > nrows) {
                Q_rows[nrows]=Q_row;
                C_rows[nrows]=C_row;
                nrows++;
            }
        }
        if (nrows<n0)
            return null;

        Tensor Q=Tensor.make_tensor(new int[] {n0,n0},Tensor.flatten(Q_rows),MOD);

        Tensor iQ=Q.mat_inv();
        List<int[][]> cpd=new ArrayList<>();
        for (int i=0; i<n0; i++) {
            int[][] tup=new int[ndims][];
            tup[0]=Tensor.mat_clm(iQ,i);
            Tensor Tc=derived_slice(T,partial_tups,Q_rows[i],C_rows[i]);
            System.arraycopy(Tc.separable_factors(),0,tup,1,ndims-1);
            cpd.add(tup);
        }

        Tensor C=Tensor.make_tensor(new int[] {n0,R-n0},Tensor.flatten(C_rows),MOD);
        Tensor iQ_C=C.op0(iQ);
        for (int y=0; y<R-n0; y++) {
            int[][] tup=new int[ndims][];
            tup[0]=Tensor.mat_clm(iQ_C,y);
            System.arraycopy(partial_tups.get(y),0,tup,1,ndims-1);
            cpd.add(tup);
        }

        return cpd;
    }

    private static List<int[][]> dfs(Tensor T, int R, List<int[][]> partial_tups, int[][] prev_ptup) {
        // T concise, shape (n_0,n_1,...) s.t. n_0 >= n_1, n_2, ... >= 1
        tick++;
        if (tick%128==0)
            log_progress(false);
        work[partial_tups.size()]++;

        int ndims=T.ndims(), MOD=T.mod();
        int[] sh=T.shape();
        int n0=sh[0];
        if (partial_tups.size()>R-n0)
            throw new RuntimeException();
        // try base case
        {
            List<int[][]> base_ret=dfs_base(T,partial_tups);
            if (base_ret!=null)
                return base_ret;
        }
        if (partial_tups.size()==R-n0)
            return null;

        int[] sh_1=Arrays.copyOfRange(sh,1,ndims);
        int[][] partial_tup;
        if (prev_ptup==null)
            partial_tup=VecUtils.init_normalized_vec_tuple(sh_1);
        else {
            partial_tup=new int[prev_ptup.length][];
            for (int i=0; i<prev_ptup.length; i++)
                partial_tup[i]=Arrays.copyOf(prev_ptup[i],prev_ptup[i].length);
            VecUtils.mut_incr_normalized_vec_tuple(partial_tup,sh_1,MOD);
        }
        for (; partial_tup[0]!=null; VecUtils.mut_incr_normalized_vec_tuple(partial_tup,sh_1,MOD)) {
            List<int[][]> nYs=new ArrayList<>(partial_tups);
            nYs.add(partial_tup);
            List<int[][]> ret=dfs(T,R,nYs,partial_tup);
            if (ret!=null)
                return ret;
        }
        return null;
    }

    public static List<int[][]> search(Tensor T, int R) {
        // T arbitrary
        if (T.ndims()<3)
            throw new RuntimeException();
        Tensor.ReduceRet reduction=T.reduce();
        Tensor T_reduced=reduction.reduced;
        // sort axes of T_reduced from longest to shortest length
        int D=T.ndims();
        int[] axes_perm=new int[D]; {
            Integer[] axes=new Integer[D];
            for (int i=0; i<D; i++)
                axes[i]=i;
            Arrays.sort(axes, new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    int n1=T_reduced.len(o1), n2=T_reduced.len(o2);
                    if (n1==n2)
                        return Integer.compare(o1,o2);
                    return Integer.compare(n2,n1);
                }
            });
            for (int i=0; i<D; i++)
                axes_perm[i]=axes[i];
        }
        int max_len=T_reduced.len(axes_perm[0]);
        if (max_len>R)
            return null;
        if (max_len==0)
            return new ArrayList<>();
        Tensor T_input=T_reduced.permutedims(axes_perm);
//        System.out.println("applied axis permutation "+Arrays.toString(axes_perm));

        tick=0;
        time_st=System.nanoTime();
        time_mark=0;
        int max_level=R-max_len+1;
        work=new long[max_level];
        List<int[][]> cpd_raw=dfs(T_input,R,new ArrayList<>(),null);
        log_progress(true);

        if (cpd_raw==null)
            return null;
        List<int[][]> cpd_unpermuted=new ArrayList<>();
        for (int[][] tup:cpd_raw) {
            int[][] unpermd_tup=new int[D][];
            for (int ax=0; ax<D; ax++)
                unpermd_tup[axes_perm[ax]]=tup[ax];
            cpd_unpermuted.add(unpermd_tup);
        }
        return CPDUtils.ops(cpd_unpermuted,reduction.inv_ops);
    }
}