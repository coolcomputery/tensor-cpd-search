/**
 * modified from cpd/skip-axis/CPD_DFS.java
 * uses a more efficient implementation of tensors over the integers modulo 2
 */

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

    /**
     * @return list of all elements in row-major order
     * only for correctness tests, not actually used in the CPD search
     */
    int[] flattened();

    /**
     *
     * @param perm array
     * @return T such that axis ax of T is axis perm[ax] of this
     * @throws Exception if perm is not permutation of {0,...,ndims}
     * does not have to be fast
     */
    Tensor permutedims(int[] perm);

    /**
     * @param M tensor
     * @param ax
     * @return T such that T[i_0,...,i'_ax,...,i_{D-1}] = sum_{i_ax} M[i'_ax,i_ax]*this[i_0,...,i_{D-1}]
     * @throws Exception if M not 2-dim or has incompatible shape with this
     * @throws Exception if this has dimension 0
     * does not have to be fast
     */
    Tensor op(Tensor M, int ax);

    /**
     * @param v
     * @return rank(v @_0 this)
     * @throws Exception if this has dimension !=3
     */
    int contraction0_rank(int[] v);

    /**
     * @param v
     * @return (rank(v @_0 this) <= 1)
     * equivalent to contraction0_rank(v)<=1, but works for any tensor and is more efficient
     */
    boolean contraction0_is_separable(int[] v);

    /**
     * @return [v_0,v_1,...] such that outer product of v_0,... equals Tc:=(v @_0 this)
     * @throws Exception if this tensor has dimension <=1
     * @throws Exception if Tc has rank > 1
     */
    int[][] contraction0_fac_vecs(int[] v);

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
     *  applying op(inv_ops[ax],ax) to reduced, for each ax, results in this
     * does not have to be fast
     */
    ReduceRet reduce();

    /**
     * @param tups_1_D
     * @return this concatenated with [outer_product(tups.get(i))]_i along axis 0
     */
    Tensor cat_oprods(List<int[][]> tups_1_D);

    /**
     * @param tup_2_D
     * @return matrix whose rows spans the kernel of the map (v,u) --> v @_0 T + outer_product(u, tup_2_D...)
     */
    Tensor oprod_ker_basis(int[][] tup_2_D);

    // matrix-specific operations

    /**
     * @param r
     * @return this[r,:]
     * @throws Exception if this has dimension !=2
     */
    int[] mat_row(int r);

    /**
     * @param c
     * @return this[:,c]
     * @throws Exception if this has dimension !=2
     */
    int[] mat_clm(int c);

    /**
     * @param v
     * @return scale*this@v
     * @throws Exception if this has dimension !=2
     */
    int[] scaled_mat_vec(int[] v, int scale);

    /**
     * @return matrix M such that M*this=I and this*M=I
     * @throws Exception if ndims!=2, shape is not square, or this is not full-rank
     * does not have to be fast
     */
    Tensor mat_inv();

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
    static Tensor make_tensor(int[] shape, int[] data, int MOD) {
        if (MOD==2)
            return new Mod2Tensor(shape,data);
        throw new RuntimeException("does not support mod!=2");
    }
    static Tensor matrix_from_rows(int[] shape, List<int[]> rows, int MOD) {
        if (shape.length!=2)
            throw new RuntimeException();
        int R=shape[0], C=shape[1];
        if (rows.size()!=R)
            throw new RuntimeException();
        for (int[] row:rows)
            if (row.length!=C)
                throw new RuntimeException();
        int[] data=new int[R*C];
        for (int r=0; r<R; r++)
            System.arraycopy(rows.get(r),0,data,r*C,C);
        return make_tensor(shape,data,MOD);
    }
}
class BitpackingUtils {
    public static final int WORD_SIZE=64;
    public static int num_words(int n) {
        return (n+WORD_SIZE-1)/WORD_SIZE;
    }
    public static long[][] zeros(int R, int C) {
        return new long[R][num_words(C)];
    }
    public static int bit_at(long[] A, int i) {
        return (int)( (A[i/WORD_SIZE]>>>(i%WORD_SIZE))&1L );
    }
    public static void or_bit(long[] A, int i, int v) {
        A[i/WORD_SIZE]|=((long)v&1L)<<(i%WORD_SIZE);
    }
}
class Mod2Tensor implements Tensor {
    private static int outer_product_at(int[][] tup, int i) {
        int size=1;
        for (int[] vec:tup)
            size*=vec.length;
        if (i<0 || i>=size)
            throw new RuntimeException();
        int prod=1;
        for (int ax=tup.length-1, h=i; ax>=0; h/=tup[ax].length, ax--)
            prod&=tup[ax][h%tup[ax].length]&1;
        return prod;
    }
    private static int row_reduce_mut(long[][] to_reduce, int C) {
        int R=to_reduce.length;
        int rk=0;
        for (int j=0; j<C && rk<Math.min(R,C); j++) {
            int pivot=-1;
            for (int r=rk; r<R; r++)
                if (BitpackingUtils.bit_at(to_reduce[r],j)!=0) {
                    pivot=r;
                    break;
                }
            if (pivot>=0) {
                swap_rows_words(to_reduce,rk,pivot);
                for (int r=0; r<R; r++)
                    if (r!=rk)
                        shear_rows_words(to_reduce,rk,r,BitpackingUtils.bit_at(to_reduce[r],j));
                rk++;
            }
        }
        return rk;
    }

    private int[] shape;
    private long[][] data_words;
    private int ndims, slice0_size, size;
    private final int MOD=2;
    private long[][] ker_buffer;  // buffer for oprod_ker_basis, to avoid having to allocate new memory every time
    private void check_invariants() {
        if (shape.length<2)
            throw new RuntimeException();
        for (int len:shape)
            if (len<0)
                throw new RuntimeException();
        if (ndims!=shape.length)
            throw new RuntimeException();
        if (slice0_size!=Tensor.prod(Arrays.copyOfRange(shape,1,ndims)))
            throw new RuntimeException();
        if (size!=shape[0]*slice0_size)
            throw new RuntimeException();

        if (data_words.length!=shape[0])
            throw new RuntimeException();
        for (long[] row:data_words)
            if (row.length!=BitpackingUtils.num_words(slice0_size))
                throw new RuntimeException();

        if (ker_buffer!=null) {
            if (ker_buffer.length!=shape[0]+shape[1])
                throw new RuntimeException();
            for (long[] row:ker_buffer)
                if (row.length!=BitpackingUtils.num_words(slice0_size + shape[0]+shape[1]))
                    throw new RuntimeException();
        }
    }
    public Mod2Tensor(int[] shape, int[] data) {
        int size=Tensor.prod(shape);
        if (data.length!=size)
            throw new RuntimeException(String.format("%d != prod(%s) = %d",data.length,Arrays.toString(shape),size));

        this.shape=Arrays.copyOf(shape,shape.length);
        this.ndims=shape.length;
        this.slice0_size=Tensor.prod(Arrays.copyOfRange(shape,1,ndims));
        this.size=size;
        this.data_words=BitpackingUtils.zeros(shape[0],slice0_size);
        for (int i0=0; i0<shape[0]; i0++)
            for (int i_rem=0; i_rem<slice0_size; i_rem++)
                BitpackingUtils.or_bit(data_words[i0],i_rem,data[i0*slice0_size+i_rem]);
        check_invariants();
    }
    private Mod2Tensor(int[] shape, long[][] data_words) {
        this.size=Tensor.prod(shape);
        this.shape=Arrays.copyOf(shape,shape.length);
        this.ndims=shape.length;
        this.slice0_size=Tensor.prod(Arrays.copyOfRange(shape,1,ndims));
        this.data_words=data_words;
        check_invariants();
    }
    private int at(int i) {
        if (i<0 || i>=size)
            throw new RuntimeException();
        return BitpackingUtils.bit_at(data_words[i/slice0_size],i%slice0_size);
    }
    public int[] flattened() {
        int[] out=new int[size];
        for (int i=0; i<size; i++)
            out[i]=at(i);
        return out;
    }
    public String toString() {
        StringBuilder s=new StringBuilder();
        for (int i=0; i<size; i++) {
            int[] coord=Tensor.idx2coord(shape,i);
            for (int ax=ndims-1; ax>=0; ax--)
                if (coord[ax]==0)
                    s.append("[");
                else
                    break;
            s.append(at(i));
            for (int ax=ndims-1; ax>=0; ax--)
                if (coord[ax]==shape[ax]-1)
                    s.append("]");
                else
                    break;
        }
        return String.format("Mod2{shape=%s : %s}",Arrays.toString(shape),s);
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

    public int[] mat_row(int r) {
        if (ndims!=2)
            throw new RuntimeException();
        int R=shape[0], C=shape[1];
        if (r<0 || r>=R)
            throw new RuntimeException();
        int[] out=new int[C];
        for (int c=0; c<C; c++)
            out[c]=at(r*C+c);
        return out;
    }
    public int[] mat_clm(int c) {
        if (ndims!=2)
            throw new RuntimeException();
        int R=shape[0], C=shape[1];
        if (c<0 || c>=C)
            throw new RuntimeException();
        int[] out=new int[R];
        for (int r=0; r<R; r++)
            out[r]=at(r*C+c);
        return out;
    }
    public int[] scaled_mat_vec(int[] v, int scale) {
        if (ndims!=2)
            throw new RuntimeException();
        int R=shape[0], C=shape[1];
        if (v.length!=C)
            throw new RuntimeException();
        int[] out=new int[R];
        for (int r=0; r<R; r++)
            for (int c=0; c<C; c++)
                out[r]^=(at(r*C+c)&(long)v[c])&1;
        for (int r=0; r<R; r++)
            out[r]&=scale&1;
        return out;
    }

    public Mod2Tensor permutedims(int[] perm) {
        int[] nshape=Tensor.permute_arr(shape,perm);
        int[] ndata=new int[size];
        for (int i=0; i<size; i++)
            ndata[Tensor.coord2idx(nshape,Tensor.permute_arr(Tensor.idx2coord(shape,i),perm))]=at(i);
        return new Mod2Tensor(nshape,ndata);
    }
    public Tensor op(Tensor M, int ax) {
        if (ax<0 || ax>=ndims)
            throw new RuntimeException();
        if (M.ndims()!=2)
            throw new RuntimeException();
        if (M.len(1)!=shape[ax])
            throw new RuntimeException();
        if (M.mod()!=MOD)
            throw new RuntimeException();
        int n_len_ax=M.len(0);
        int[] nshape=Arrays.copyOf(shape,ndims);
        nshape[ax]=n_len_ax;
        int[] ndata=new int[Tensor.prod(nshape)];
        for (int ni=0; ni<n_len_ax; ni++) {
            int[] M_row=M.mat_row(ni);
            for (int idx=0; idx<size; idx++) {
                int[] ncoord=Tensor.idx2coord(shape,idx);
                int o_ax=ncoord[ax];
                ncoord[ax]=ni;
                int nidx=Tensor.coord2idx(nshape,ncoord);
                ndata[nidx]^=(M_row[o_ax]&at(idx))&1L;
            }
        }
        return new Mod2Tensor(nshape,ndata);
    }
    private long[] contract0(int[] v) {
        if (v.length!=shape[0])
            throw new RuntimeException();
        long[] out=new long[BitpackingUtils.num_words(slice0_size)];
        for (int i0=0; i0<shape[0]; i0++) {
            long mask=-(long)(v[i0]&1L);
            for (int w_rem=0; w_rem<BitpackingUtils.num_words(slice0_size); w_rem++)
                out[w_rem]^=mask&data_words[i0][w_rem];
        }
        return out;
    }
    private int[][] slice0_fac_vecs(long[] slice0) {
        int[][] fac_vecs=new int[ndims-1][];
        for (int ax=1; ax<ndims; ax++)
            fac_vecs[ax-1]=new int[shape[ax]];
        for (int i=0; i<slice0_size; i++)
            if (BitpackingUtils.bit_at(slice0,i)!=0) {
                for (int ax=ndims-1, h=i; ax>=1; h/=shape[ax], ax--)
                    fac_vecs[ax-1][h%shape[ax]]=1;
            }
        for (int i=0; i<slice0_size; i++) {
            int actual=BitpackingUtils.bit_at(slice0,i);
            int expected=1;
            for (int ax=ndims-1, h=i; ax>=1; h/=shape[ax], ax--)
                expected&=fac_vecs[ax-1][h%shape[ax]];
            if (actual!=expected)
                return null;
        }
        return fac_vecs;
    }
    public boolean contraction0_is_separable(int[] v) {
        return slice0_fac_vecs(contract0(v))!=null;
    }
    public int[][] contraction0_fac_vecs(int[] v) {
        int[][] ret=slice0_fac_vecs(contract0(v));
        if (ret==null)
            throw new RuntimeException();
        return ret;
    }
    public int contraction0_rank(int[] v) {
        if (ndims!=3)
            throw new RuntimeException();
        long[] con0=contract0(v);
        long[][] to_reduce=BitpackingUtils.zeros(shape[1],shape[2]);
        for (int i1=0; i1<shape[1]; i1++)
            for (int i2=0; i2<shape[2]; i2++)
                BitpackingUtils.or_bit(to_reduce[i1],i2,BitpackingUtils.bit_at(con0,i1*shape[2]+i2));
//        System.out.println(shape[1]+" "+shape[2]);
//        for (long[] row:to_reduce)
//            System.out.println(Arrays.toString(row));
        return row_reduce_mut(to_reduce,shape[2]);
    }
    public Tensor cat_oprods(List<int[][]> tups_1_D) {
        for (int[][] tup:tups_1_D) {
            if (tup.length!=ndims-1)
                throw new RuntimeException();
            for (int ax=1; ax<ndims; ax++)
                if (tup[ax-1].length!=shape[ax])
                    throw new RuntimeException();
        }

        int n_tups=tups_1_D.size();
        long[][] out_data=BitpackingUtils.zeros(shape[0]+n_tups,slice0_size);
        for (int i0=0; i0<shape[0]; i0++)
            System.arraycopy(this.data_words[i0],0,out_data[i0],0,BitpackingUtils.num_words(slice0_size));
        for (int tup_i=0; tup_i<n_tups; tup_i++)
            for (int i_rem=0; i_rem<slice0_size; i_rem++)
                BitpackingUtils.or_bit(out_data[shape[0]+tup_i],i_rem,outer_product_at(tups_1_D.get(tup_i),i_rem));

        int[] nshape=Arrays.copyOf(this.shape,this.ndims);
        nshape[0]+=n_tups;
        return new Mod2Tensor(nshape,out_data);
    }

    private static void swap_rows_words(long[][] data, int r0, int r1) {
        if (data[r0].length!=data[r1].length)
            throw new RuntimeException();
        for (int c=0; c<data[r0].length; c++) {
            long tmp=data[r0][c];
            data[r0][c]=data[r1][c];
            data[r1][c]=tmp;
        }
    }
    private static void shear_rows_words(long[][] data, int r0, int r1, int v) {
        if (r0==r1)
            throw new RuntimeException();
        if (data[r0].length!=data[r1].length)
            throw new RuntimeException();
        v&=1;
        if (v!=0) {  // faster than trying to make it branchless by using a mask
            int n=data[r0].length;
            for (int c=0; c<n; c++)
                data[r1][c]^=data[r0][c];
        }
    }

    private long[][] aug_I() {
        // return [(axis-0 unfolding of this) | I]
        long[][] aug=BitpackingUtils.zeros(shape[0],slice0_size+shape[0]);
        for (int i0=0; i0<shape[0]; i0++)
            System.arraycopy(this.data_words[i0],0,aug[i0],0,BitpackingUtils.num_words(slice0_size));
        for (int i0=0; i0<shape[0]; i0++)
            BitpackingUtils.or_bit(aug[i0],slice0_size+i0,1);
        return aug;
    }
    public Tensor mat_inv() {
        if (ndims!=2)
            throw new RuntimeException();
        int R=shape[0], C=shape[1];
        if (R!=C)
            throw new RuntimeException();
        long[][] aug=aug_I();
        int rk=row_reduce_mut(aug,C);
        if (rk!=R)
            throw new RuntimeException();
        int[] out=new int[R*R];
        for (int i0=0; i0<R; i0++)
            for (int j=0; j<R; j++)
                out[i0*R+j]=BitpackingUtils.bit_at(aug[i0],C+j);
        return new Mod2Tensor(new int[] {R,R},out);
    }

    public Tensor oprod_ker_basis(int[][] tup_2_D) {
        if (tup_2_D.length!=ndims-2)
            throw new RuntimeException();
        for (int ax=2; ax<ndims; ax++)
            if (tup_2_D[ax-2].length!=shape[ax])
                throw new RuntimeException();

        int sz_2_D=1;
        for (int ax=2; ax<ndims; ax++)
            sz_2_D*=shape[ax];
        int domain_dim=shape[0]+shape[1];
        if (ker_buffer==null)
            ker_buffer=BitpackingUtils.zeros(domain_dim,slice0_size+domain_dim);
        else {
            for (long[] row:ker_buffer)
                Arrays.fill(row,0L);
        }
        for (int i0=0; i0<shape[0]; i0++)
            System.arraycopy(data_words[i0],0,ker_buffer[i0],0,BitpackingUtils.num_words(slice0_size));
        for (int i2_D=0; i2_D<sz_2_D; i2_D++) {
            int oprod_elem=outer_product_at(tup_2_D,i2_D);
            for (int i1=0; i1<shape[1]; i1++) {
                int j=i1*sz_2_D+i2_D;
                BitpackingUtils.or_bit(ker_buffer[shape[0]+i1],j,oprod_elem);
            }
        }
        for (int i=0; i<domain_dim; i++)
            BitpackingUtils.or_bit(ker_buffer[i],slice0_size+i,1);

        int rk=row_reduce_mut(ker_buffer,slice0_size);

        long[][] out_data=BitpackingUtils.zeros(domain_dim-rk,domain_dim);
        for (int i=0; i<domain_dim-rk; i++)
            for (int j=0; j<domain_dim; j++)
                BitpackingUtils.or_bit(out_data[i],j,BitpackingUtils.bit_at(ker_buffer[i+rk],slice0_size+j));
        return new Mod2Tensor(new int[] {domain_dim-rk,domain_dim},out_data);
    }

    static class RrefRet {
        long[][] reduced, ileftT;
        int rk;
        public RrefRet(long[][] reduced, long[][] ileftT, int rk) {
            this.reduced=reduced;
            this.ileftT=ileftT;
            this.rk=rk;
        }
    }
    private RrefRet reduce0() {
        if (ndims<1)
            throw new RuntimeException();
        int n0=shape[0];
        long[][] reduced=new long[n0][];
        for (int i0=0; i0<n0; i0++)
            reduced[i0]=Arrays.copyOf(data_words[i0],BitpackingUtils.num_words(slice0_size));
        long[][] inv_T_transform=new long[n0][BitpackingUtils.num_words(n0)];
        for (int i=0; i<n0; i++)
            BitpackingUtils.or_bit(inv_T_transform[i],i,1);
        int rk0=0;
        for (int j=0; j<slice0_size; j++) {
            int pivot=-1;
            for (int r=rk0; r<n0; r++)
                if (BitpackingUtils.bit_at(reduced[r],j)!=0) {
                    pivot=r;
                    break;
                }
            if (pivot>=0) {
                swap_rows_words(reduced,rk0,pivot);
                swap_rows_words(inv_T_transform,rk0,pivot);
                for (int r=0; r<n0; r++)
                    if (r!=rk0) {
                        int v=BitpackingUtils.bit_at(reduced[r],j);
                        shear_rows_words(reduced,rk0,r,v);
                        shear_rows_words(inv_T_transform,r,rk0,v);
                    }
                rk0++;
            }
        }
        return new RrefRet(reduced,inv_T_transform,rk0);
    }

    public ReduceRet reduce() {
        if (ndims==0)
            throw new RuntimeException();
        Tensor[] inv_ops=new Tensor[ndims];
        Mod2Tensor out=this;
        int[] rot1=new int[ndims];
        for (int ax=0; ax<ndims; ax++)
            rot1[ax]=(ax+1)%ndims;
        for (int ax=0; ax<ndims; ax++, out=out.permutedims(rot1)) {
            int n=shape[ax];
            RrefRet reduc=out.reduce0();
            if (reduc.rk==n) {
                int[] eye=new int[n*n];
                for (int i=0; i<n; i++)
                    eye[i*n+i]=1;
                inv_ops[ax]=new Mod2Tensor(new int[] {n,n},eye);
            }
            else {
                inv_ops[ax]=new Mod2Tensor(
                        new int[] {reduc.rk,n},
                        Arrays.copyOf(reduc.ileftT,reduc.rk)
                ).permutedims(new int[] {1,0});  // transpose
                int[] nshape=Arrays.copyOf(out.shape,ndims);
                nshape[0]=reduc.rk;
//                System.out.println(Arrays.toString(nshape));
                out=new Mod2Tensor(
                        nshape,
                        Arrays.copyOf(reduc.reduced,reduc.rk)
                );
            }
        }
        return new ReduceRet(out,inv_ops);
    }
}

/**
 * dynamically maintains a linear span of vectors in a fixed-dimensional space
 */
interface LinearSpan {
    int mod();

    /**
     * @return dimension of space that span lies in
     */
    int dim();

    /**
     * @return dimension of span itself, i.e. number of vectors in a basis of this span
     */
    int rank();

    /**
     * @param v vector, elements not nec. modded by mod()
     * @return (v contained in this span)
     * @throws Exception if v.length != dim()
     */
    boolean contains_vec(int[] v);

    /**
     * add a new vector to this span
     * @param v vector, elements not nec. modded by mod()
     * @throws Exception if v.length != dim()
     */
    void add_vec(int[] v);

    static LinearSpan make_linspan(int dim, int MOD) {
        if (MOD==2)
            return new Mod2LinearSpan(dim);
        throw new RuntimeException();
    }
}
class Mod2LinearSpan implements LinearSpan {
    private final int MOD=2;
    private int D, rank;
    /*
    collectively, all non-null items in vec_with_lead
    form the rref matrix whose rowspan is the linear span that this object represents
     */
    private long[][] vec_with_lead;  // vec_with_lead[i]=vector with leading 1 at index i
    public Mod2LinearSpan(int dim) {
        D=dim;
        rank=0;
        vec_with_lead=new long[D][];
    }
    public int mod() {
        return MOD;
    }
    public int dim() {
        return D;
    }
    public int rank() {
        return rank;
    }
    public long[] subtracted_word(int[] v) {
        long[] subtract=new long[BitpackingUtils.num_words(D)];
        for (int i=0; i<D; i++)
            BitpackingUtils.or_bit(subtract,i,v[i]);
        for (int i=0; i<D; i++)
            if (vec_with_lead[i]!=null && (v[i]&1)!=0)
                for (int w=0; w<BitpackingUtils.num_words(D); w++)
                    subtract[w]^=vec_with_lead[i][w];
        return subtract;
    }
    public boolean contains_vec(int[] v) {
        if (v.length!=D)
            throw new RuntimeException();
        long[] sub=subtracted_word(v);
        for (long word:sub)
            if (word!=0)
                return false;
        return true;
    }
    public void add_vec(int[] v) {
        if (v.length!=D)
            throw new RuntimeException();
        long[] sub=subtracted_word(v);
        int nlead=0;
        while (nlead<D && BitpackingUtils.bit_at(sub,nlead)==0)
            nlead++;
        if (nlead>=D)
            return;  // v already contained in this linear span
        if (vec_with_lead[nlead]!=null) {
            System.out.println(nlead+" "+Arrays.toString(v)+" "+Arrays.toString(sub));
            for (int i=0; i<D; i++)
                System.out.println(i+" "+Arrays.toString(vec_with_lead[i]));
            throw new RuntimeException();
        }
        vec_with_lead[nlead]=sub;
        rank++;

        // reduce previously added vectors
        for (int i=0; i<D; i++)
            if (i!=nlead && vec_with_lead[i]!=null && BitpackingUtils.bit_at(vec_with_lead[i],nlead)!=0)
                for (int w=0; w<BitpackingUtils.num_words(D); w++)
                    vec_with_lead[i][w]^=sub[w];
    }
}

interface Pruner {
    class Result {
        enum Decision {
            YES, NO, MAYBE;
        }
        private Decision decision;
        private List<int[][]> cpd;
        private Result(Decision decision, List<int[][]> cpd) {
            this.decision=decision;
            this.cpd=cpd;
        }
        public static Result yes(List<int[][]> cpd) {
            return new Result(Decision.YES,cpd);
        }
        public static Result no() {
            return new Result(Decision.NO,null);
        }
        public static Result maybe() {
            return new Result(Decision.MAYBE,null);
        }
        public boolean is_yes() {
            return decision==Decision.YES;
        }
        public boolean is_no() {
            return decision==Decision.NO;
        }
        public List<int[][]> cpd() {
            if (is_yes())
                return cpd;
            throw new RuntimeException();
        }
    }
    Result prune(Tensor T, int R, List<int[][]> partial_tups);
}
class RrefPruner implements Pruner {
    public Result prune(Tensor T, int R, List<int[][]> partial_tups) {
        if (T.ndims()!=3)
            return Result.maybe();
        int n0=T.len(0), ndims=T.ndims(), MOD=T.mod();
        Tensor T_help=T.cat_oprods(partial_tups);
        /*
        necessary condition for rk(T)<=R:
        there must exist vectors v_0,...,v_{n_0-1} and c_0,...c_{n_0-1} such that
            {v_i}_i is linearly independent AND
            for all i: rk( [v_i | c_i] @_0 T_help ) <= R-partial_tups.size()-n0+1

        sufficient condition:
        if there exist such vector v_i, c_i with the additional condition that
            sum_i rk( [v_i | c_i] @_0 T_help ) <= R
        conjecture: optimal to pick vectors [v_i | c_i] greedily
         */
        LinearSpan S_nec=LinearSpan.make_linspan(n0,T.mod());
        LinearSpan[] S_sufs=new LinearSpan[Math.min(T.len(1),T.len(2))+1];
        List<List<int[][]>> pairs_at_rank=new ArrayList<>();
        for (int rk=0; rk<S_sufs.length; rk++) {
            S_sufs[rk]=LinearSpan.make_linspan(n0,T.mod());
            pairs_at_rank.add(new ArrayList<>());
        }
        for (int[] v=VecUtils.init_normalized_vec(n0); v!=null; v=VecUtils.next_normalized_vec(v,MOD)) {
            int best_rk=Integer.MAX_VALUE;
            int[] best_c=null;
            for (int[] c=VecUtils.init_vec(partial_tups.size()); c!=null; c=VecUtils.next_vec(c,MOD)) {
                int rk=T_help.contraction0_rank(VecUtils.concat(v,c));
                if (rk<best_rk) {
                    best_rk=rk;
                    best_c=c;
                }
            }
            if (best_rk<=R-partial_tups.size()-n0+1) {
                S_nec.add_vec(v);
            }
            if (best_rk>=S_sufs.length)
                throw new RuntimeException();
            if (!S_sufs[best_rk].contains_vec(v)) {
                S_sufs[best_rk].add_vec(v);
                pairs_at_rank.get(best_rk).add(new int[][] {v,best_c});
            }
        }
        if (S_nec.rank()<n0)
            return Result.no();

        List<int[]> Q_rows=new ArrayList<>(), C_rows=new ArrayList<>();
        LinearSpan S_suf_greedy=LinearSpan.make_linspan(n0,T.mod());
        int tot_rk=0;
        for (int rk=0; rk<S_sufs.length; rk++) {
            for (int[][] pair:pairs_at_rank.get(rk)) {
                int[] v=pair[0], c=pair[1];
                if (!S_suf_greedy.contains_vec(v)) {
                    S_suf_greedy.add_vec(v);
                    Q_rows.add(v);
                    C_rows.add(c);
                    tot_rk+=rk;
                }
            }
        }
        if (S_suf_greedy.rank()==n0 && tot_rk+partial_tups.size()<=R) {
//            System.out.printf("tot_rk=%d R=%d\n",tot_rk,R);
            Tensor Q=Tensor.matrix_from_rows(new int[] {n0,n0},Q_rows,MOD),
                    C=Tensor.matrix_from_rows(new int[] {n0,partial_tups.size()},C_rows,MOD);
            Tensor iQ=Q.mat_inv();
            List<int[][]> cpd=new ArrayList<>();
            for (int i=0; i<n0; i++) {
                int[] vc=VecUtils.concat(Q.mat_row(i),C.mat_row(i));
                Tensor vc_tensor=Tensor.make_tensor(new int[] {1,n0+partial_tups.size()},vc,MOD);
                Tensor con0=T_help.op(vc_tensor,0);
                Tensor to_reduce=Tensor.make_tensor(new int[] {T.len(1),T.len(2)},con0.flattened(),MOD);  // flatten con0 along axis 0
                Tensor.ReduceRet reduc=to_reduce.reduce();
                if (reduc.reduced.len(0)!=reduc.reduced.len(1))
                    throw new RuntimeException();
                if (reduc.reduced.len(0)!=T_help.contraction0_rank(vc))
                    throw new RuntimeException(reduc.reduced.len(0)+" "+T_help.contraction0_rank(vc)+" "+Arrays.toString(vc)+" "+T_help);
                Tensor left_fac=reduc.reduced.op(reduc.inv_ops[0],0), right_fac=reduc.inv_ops[1];
                for (int rk=0; rk<reduc.reduced.len(0); rk++)
                    cpd.add(new int[][] {iQ.mat_clm(i),left_fac.mat_clm(rk),right_fac.mat_clm(rk)});
            }
            if (cpd.size()!=tot_rk)
                throw new RuntimeException(cpd.size()+" "+tot_rk);
            for (int y=0; y<partial_tups.size(); y++) {
                int[][] tup=new int[ndims][];
                tup[0]=iQ.scaled_mat_vec(C.mat_clm(y),-1);
                System.arraycopy(partial_tups.get(y),0,tup,1,ndims-1);
                cpd.add(tup);
            }
            return Result.yes(cpd);
        }
        return Result.maybe();
    }
}
class LaskPruner implements Pruner {
    public Result prune(Tensor T, int R, List<int[][]> partial_tups) {
        if (T.ndims()!=3)
            return Result.maybe();
        int n0=T.len(0), MOD=T.mod();
        Tensor T_help=T.cat_oprods(partial_tups);

        long tot_rk=0, n_normalized_vecs=0;
        for (int[] v=VecUtils.init_normalized_vec(n0); v!=null; v=VecUtils.next_normalized_vec(v,MOD)) {
            int best_rk=Integer.MAX_VALUE;
            for (int[] c=VecUtils.init_vec(partial_tups.size()); c!=null; c=VecUtils.next_vec(c,MOD))
                best_rk=Math.min(best_rk,T_help.contraction0_rank(VecUtils.concat(v,c)));
            tot_rk+=best_rk;
            n_normalized_vecs++;
        }
        long mod_pow_n0=1;
        for (int rep=0; rep<n0; rep++)
            mod_pow_n0*=MOD;
        long thresh=(R-partial_tups.size()) * (mod_pow_n0 - mod_pow_n0/MOD) / (MOD-1);
        if (tot_rk<=thresh)
            return Result.maybe();
        return Result.no();
    }
}

class VecUtils {
    public static int[] concat(int[] A, int[] B) {
        int[] out=new int[A.length+B.length];
        System.arraycopy(A,0,out,0,A.length);
        System.arraycopy(B,0,out,A.length,B.length);
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
 */
public class CPD_DFS {
    public static List<int[][]> cpd_ops(List<int[][]> cpd, Tensor[] ops) {
        int ndims=ops.length;
        List<int[][]> out=new ArrayList<>();
        for (int[][] tup:cpd) {
            if (tup.length!=ndims)
                throw new RuntimeException();
            int[][] ntup=new int[ndims][];
            for (int ax=0; ax<ndims; ax++) {
                ntup[ax]=(
                        ops[ax]==null?
                                tup[ax]:
                                ops[ax].scaled_mat_vec(tup[ax],1)
                );
            }
            out.add(ntup);
        }
        return out;
    }

    public static boolean verbose=true;
    private static final long SEC_TO_NANO=1000_000_000;
    private static long tick, time_st, time_mark;
    private static long[] work;
    private static Map<String,Pruner> pruners;
    private static Map<String,long[]> prune_cnts;
    public static long[] work_stats() {
        return Arrays.copyOf(work,work.length);
    }

    private static void log_progress(boolean force_print) {
        long time_elapsed=System.nanoTime()-time_st;
        if (time_elapsed>time_mark || force_print) {
            if (time_mark>0 || force_print) {
                System.out.printf("t=%.3f work=%s\n",time_elapsed/(float)SEC_TO_NANO,Arrays.toString(work));
                for (String prune_name:prune_cnts.keySet())
                    System.out.printf("\t%s pruning: %s\n",prune_name,Arrays.toString(prune_cnts.get(prune_name)));
            }
            while (time_elapsed>time_mark)
                time_mark+=Math.pow(10,(int)(Math.log10(Math.max(10,time_mark/(float)SEC_TO_NANO))))*SEC_TO_NANO;
        }
    }

    // each item partial_tups.get(r) corresponds to the column tuple ((Y_d)_{:,r})_{d\ne 0}

    // could be implemented as an iterator or generator,
    //  but doing so in Java is annoying and would probably incur high constant factors
    private static List<int[][]> good_pairs(Tensor T, List<int[][]> partial_tups) {
        int ndims=T.ndims(), MOD=T.mod();
        int[] sh=T.shape();
        int n0=sh[0];
        int R=n0+partial_tups.size();
        Tensor T_help=T.cat_oprods(partial_tups);

        List<int[][]> out=new ArrayList<>();

        int cnt=0;
        for (int ax=2; ax<ndims; ax++)
            cnt+=sh[ax];
        // TODO allow Tester to force this branching
        if (R<=cnt) {
            for (int[] v=VecUtils.init_normalized_vec(n0); v!=null; v=VecUtils.next_normalized_vec(v,MOD))
                for (int[] c=VecUtils.init_vec(partial_tups.size()); c!=null; c=VecUtils.next_vec(c,MOD)) {
                    if (T_help.contraction0_is_separable(VecUtils.concat(v,c))) {
                        out.add(new int[][] {v,c});
                        break;
                    }
                }
        }
        else {
            int[] sh_2=Arrays.copyOfRange(sh,2,ndims);
            for (
                    int[][] tup_2_D=VecUtils.init_normalized_vec_tuple(sh_2);
                    tup_2_D[0]!=null;
                    VecUtils.mut_incr_normalized_vec_tuple(tup_2_D,sh_2,MOD)
            ) {
                Tensor ker=T_help.oprod_ker_basis(tup_2_D);
                for (int i=0; i<ker.len(0); i++) {
                    int[] row=ker.mat_row(i);
                    out.add(new int[][] {Arrays.copyOfRange(row,0,n0),Arrays.copyOfRange(row,n0,R)});
                }
            }
        }
        return out;
    }
    private static List<int[][]> dfs_base(Tensor T, List<int[][]> partial_tups) {
        int ndims=T.ndims();
        int n0=T.len(0), MOD=T.mod();
        List<int[]> Q_rows=new ArrayList<>(), C_rows=new ArrayList<>();
        LinearSpan Qspan=LinearSpan.make_linspan(n0,MOD);
        for (int[][] pair:good_pairs(T,partial_tups)) {
            int[] v=pair[0], c=pair[1];
            if (!Qspan.contains_vec(v)) {
                Qspan.add_vec(v);
                Q_rows.add(v);
                C_rows.add(c);
            }
        }
        if (Qspan.rank()<n0)
            return null;

        Tensor Q=Tensor.matrix_from_rows(new int[] {n0,n0},Q_rows,MOD),
                C=Tensor.matrix_from_rows(new int[] {n0,partial_tups.size()},C_rows,MOD);
        Tensor iQ=Q.mat_inv();
        Tensor T_help=T.cat_oprods(partial_tups);
        List<int[][]> cpd=new ArrayList<>();
        for (int i=0; i<n0; i++) {
            int[][] facs=T_help.contraction0_fac_vecs(VecUtils.concat(Q.mat_row(i),C.mat_row(i)));
            int[][] tup=new int[ndims][];
            tup[0]=iQ.mat_clm(i);
            System.arraycopy(facs,0,tup,1,ndims-1);
            cpd.add(tup);
        }

        for (int y=0; y<partial_tups.size(); y++) {
            int[][] tup=new int[ndims][];
            tup[0]=iQ.scaled_mat_vec(C.mat_clm(y),-1);
            System.arraycopy(partial_tups.get(y),0,tup,1,ndims-1);
            cpd.add(tup);
        }
        return cpd;
    }

    private static List<int[][]> dfs(Tensor T, int R, List<int[][]> partial_tups, int[][] prev_ptup) {
        // T concise, shape (n_0,n_1,...) s.t. n_0 >= n_1, n_2, ... >= 1
        tick++;
        if (tick%128==0 && CPD_DFS.verbose)
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

        for (String prune_name:pruners.keySet()) {
            Pruner pruner=pruners.get(prune_name);
            Pruner.Result prune_ret=pruner.prune(T,R,partial_tups);
            if (prune_ret.is_no()) {
                prune_cnts.get(prune_name)[partial_tups.size()]++;
                return null;
            }
            if (prune_ret.is_yes()) {
                prune_cnts.get(prune_name)[partial_tups.size()]++;
                return prune_ret.cpd();
            }
        }

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

    public static List<int[][]> search(Tensor T, int R, Map<String,Pruner> pruners) {
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
        CPD_DFS.pruners=pruners;
        prune_cnts=new TreeMap<>();
        for (String prune_name:pruners.keySet())
            prune_cnts.put(prune_name,new long[max_level]);
        List<int[][]> cpd_raw=dfs(T_input,R,new ArrayList<>(),null);
        if (CPD_DFS.verbose)
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
        return cpd_ops(cpd_unpermuted,reduction.inv_ops);
    }
    public static List<int[][]> search(Tensor T, int R) {
        return search(T,R,new HashMap<>());
    }
}