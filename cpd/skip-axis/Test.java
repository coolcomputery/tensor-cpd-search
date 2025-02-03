import java.util.*;

/**
 * incomplete set of unit and integration tests
 */
public class Test {
    private static void assert_equals(int a, int b) {
        if (a!=b)
            throw new RuntimeException(a+" != "+b);
    }
    private static boolean int2_equals(int[][] A, int[][] B) {
        if (A.length!=B.length)
            return false;
        for (int i=0; i<A.length; i++)
            if (!Arrays.equals(A[i],B[i]))
                return false;
        return true;
    }

    private static String int2_str(int[][] A) {
        StringBuilder out=new StringBuilder("[");
        for (int i=0; i<A.length; i++)
            out.append(i>0?" ":"").append(Arrays.toString(A[i]));
        out.append("]");
        return out.toString();
    }
    private static void assert_equals(int[][] A, int[][] B) {
        if (!int2_equals(A,B))
            throw new RuntimeException(int2_str(A)+" "+int2_str(B));
    }

    private static GenTensor rnd_gentensor(int[] shape, int MOD, SplittableRandom rnd) {
        int[] data=new int[Tensor.prod(shape)];
        for (int i=0; i<data.length; i++)
            data[i]=rnd.nextInt(MOD);
        return new GenTensor(shape,data,MOD);
    }
    private static Mod2Tensor rnd_mod2tensor(int[] shape, SplittableRandom rnd) {
        int[] data=new int[Tensor.prod(shape)];
        for (int i=0; i<data.length; i++)
            data[i]=rnd.nextInt(2);
        return new Mod2Tensor(shape,data);
    }
    private static boolean coord_in_bounds(int[] sh, int[] coord) {
        for (int ax=0; ax<sh.length; ax++)
            if (coord[ax]<0 || coord[ax]>=sh[ax])
                return false;
        return true;
    }
    private static void incr_tensor_coord(int[] sh, int[] coord) {
        if (!coord_in_bounds(sh,coord))
            throw new RuntimeException(Arrays.toString(sh)+" "+Arrays.toString(coord));
        coord[sh.length-1]++;
        for (int ax=sh.length-1; ax>0 && coord[ax]==sh[ax]; ax--) {
            coord[ax]=0;
            coord[ax-1]++;
        }
    }
    private static boolean tensor_equals(Tensor A, Tensor B) {
        if (!Arrays.equals(A.shape(),B.shape()))
            return false;
        if (A.mod()!=B.mod())
            return false;
        int[] sh=A.shape();
        if (Tensor.prod(sh)==0)
            return true;
        int[] coord=new int[sh.length];
        while (coord_in_bounds(sh,coord)) {
            if (A.at(coord)!=B.at(coord))
                return false;
            incr_tensor_coord(sh,coord);
        }
        return true;
    }
    private static void assert_tensor_equals(Tensor A, Tensor B) {
        if (!tensor_equals(A,B))
            throw new RuntimeException(A+" != "+B);
    }
    private static int[][] matrix_data(Tensor A) {
        if (A.ndims()!=2)
            throw new RuntimeException();
        int R=A.len(0), C=A.len(1);
        int[][] out=new int[R][C];
        for (int i=0; i<R; i++)
            for (int j=0; j<C; j++)
                out[i][j]=A.at(new int[] {i,j});
        return out;
    }
    private static int[][] eye(int n) {
        int[][] out=new int[n][n];
        for (int i=0; i<n; i++)
            out[i][i]=1;
        return out;
    }
    private static int[][] mm(int I, int J, int K, int[][] A, int[][] B, int MOD) {
        // multiplying a IxJ matrix by a JxK matrix
        int[][] out=new int[I][K];
        for (int i=0; i<I; i++)
            for (int j=0; j<J; j++)
                for (int k=0; k<K; k++)
                    out[i][k]=ModArith.add(out[i][k],ModArith.mul(A[i][j],B[j][k],MOD),MOD);
        return out;
    }
    private static void assert_inv_correct(Tensor M) {
        if (M.ndims()!=2 || M.len(0)!=M.len(1))
            throw new RuntimeException("invalid test: must be square matrix, not shape "+Arrays.toString(M.shape()));
        int n=M.len(0);
        if (M.mat_rank()!=n)
            throw new RuntimeException("invalid test: matrix not invertible");
        assert_equals(mm(n,n,n,matrix_data(M.mat_inv()),matrix_data(M),M.mod()),eye(n));
    }
    private static GenTensor op0(Tensor T, Tensor M) {
        if (T.ndims()<1)
            throw new RuntimeException();
        int MOD=T.mod();
        int[][] op_data=matrix_data(M);
        int[] sh=T.shape();
        int[] nsh=Arrays.copyOf(sh,sh.length);
        nsh[0]=M.len(0);
        int[] out=new int[Tensor.prod(nsh)];
        int[] coord=new int[T.ndims()];
        while (coord_in_bounds(sh,coord)) {
            for (int nr=0; nr<nsh[0]; nr++) {
                int[] ncoord=Arrays.copyOf(coord,sh.length);
                ncoord[0]=nr;
                int ni=Tensor.coord2idx(nsh,ncoord);
                out[ni]=ModArith.add(out[ni],ModArith.mul(op_data[nr][coord[0]],T.at(coord),MOD),MOD);
            }
            incr_tensor_coord(sh,coord);
        }
        return new GenTensor(nsh,out,MOD);
    }

    private static abstract class ExceptionTester {
        abstract void run();
        ExceptionTester() {
            boolean thrown=false;
            try {
                run();
            }
            catch (Exception e) {
                thrown=true;
            }
            if (!thrown)
                throw new RuntimeException();
        }
    }

    public static void test_utils() {
        {  // modular arithmetic
            int MOD=1000_000_007;
            // make sure to cover cases where naively applying operands would result in overflow for signed 32-bit
            assert_equals(ModArith.mod(3L*MOD+1,MOD),1);
            assert_equals(ModArith.mod(-3L*MOD+1,MOD),1);

            assert_equals(ModArith.add(2*MOD+2,2*MOD+3,MOD),5);
            assert_equals(ModArith.mul(2*MOD+2,2*MOD+3,MOD),6);

            assert_equals(ModArith.exp(2,3,MOD),8);
            assert_equals(ModArith.exp(2,MOD-1,MOD),1);

            assert_equals(ModArith.inv(2,MOD),500_000_004);

            if (!ModArith.is_prime(5))
                throw new RuntimeException();
            if (ModArith.is_prime(6))
                throw new RuntimeException();
        }

        // Tensor static methods
        {  // prod
            assert_equals(Tensor.prod(new int[] {2,3,4}),24);
        }

        // Tensor instance methods
        {  // permute
            {
                int MOD=Integer.MAX_VALUE;
                Tensor T=new GenTensor(new int[] {1,2,3},new int[] {0,1,2,3,4,5},MOD);
                Tensor expected=new GenTensor(new int[] {2,3,1},new int[] {0,1,2,3,4,5},MOD);
                Tensor actual=T.permutedims(new int[] {1,2,0});
                assert_tensor_equals(actual,expected);
            }
            {
                Tensor T=new Mod2Tensor(new int[] {1,2,3},new int[] {0,1,2,3,4,5});
                Tensor expected=new Mod2Tensor(new int[] {2,3,1},new int[] {0,1,2,3,4,5});
                Tensor actual=T.permutedims(new int[] {1,2,0});
                assert_tensor_equals(actual,expected);
            }
        }
        {  // contract0
            {  // zero-length test cases
                assert_tensor_equals(
                        new GenTensor(new int[] {0,2,2},new int[0],2).contract0(new int[0]),
                        new GenTensor(new int[] {2,2},new int[4],2)
                );
                assert_tensor_equals(
                        new GenTensor(new int[] {1,0,2},new int[0],2).contract0(new int[1]),
                        new GenTensor(new int[] {0,2},new int[0],2)
                );

                assert_tensor_equals(
                        new Mod2Tensor(new int[] {0,2,2},new int[0]).contract0(new int[0]),
                        new Mod2Tensor(new int[] {2,2},new int[4])
                );
                assert_tensor_equals(
                        new Mod2Tensor(new int[] {1,0,2},new int[0]).contract0(new int[1]),
                        new Mod2Tensor(new int[] {0,2},new int[0])
                );
            }
            {  // use small negative numbers and a large modulo to check correct arithmetic
                {
                    int[] sh={3,2,2};
                    int[] data=new int[12];
                    for (int i=0; i<data.length; i++)
                        data[i]=-(i+1);
                    int[] expected_sh={2,2};
                    int[] expected_data={-(1*1+2*5+3*9),-(1*2+2*6+3*10),-(1*3+2*7+3*11),-(1*4+2*8+3*12)};
                    int MOD=Integer.MAX_VALUE;
                    Tensor T=new GenTensor(sh,data,MOD);
                    int[] v=new int[] {1,2,3};
                    Tensor expected=new GenTensor(expected_sh,expected_data,MOD);
                    Tensor actual=T.contract0(v);
                    assert_tensor_equals(actual,expected);
                }
                {  // make sure at least one tube has more than one set bit, to catch the * vs & mistake
                    Tensor T=new Mod2Tensor(new int[] {2,2,3},new int[] {0,0,0, 0,1,1, 1,0,1, 1,1,1});
                    int[] v=new int[] {1,1};
                    Tensor expected=new Mod2Tensor(new int[] {2,3},new int[] {1,0,1, 1,0,0});
                    Tensor actual=T.contract0(v);
                    assert_tensor_equals(actual,expected);
                }
            }
        }
        {  // subtract_outerprod
            int[] sh={2,2,2}, data={1,2,3,4,5,6,7,8};
            int[][] tup={
                    {-1,-2},
                    {-3,-4},
                    {-5,-6},
            };
            int[] expected_data={1+1*3*5,2+1*3*6,3+1*4*5,4+1*4*6,5+2*3*5,6+2*3*6,7+2*4*5,8+2*4*6};

            {
                int MOD=Integer.MAX_VALUE;
                Tensor T=new GenTensor(sh,data,MOD);
                Tensor expected=new GenTensor(sh,expected_data,MOD);
                Tensor actual=T.add_weighted_outerprods(new int[] {-1},Collections.singletonList(tup));
                assert_tensor_equals(actual,expected);
            }
            {
                Tensor T=new Mod2Tensor(sh,data);
                Tensor expected=new Mod2Tensor(sh,expected_data);
                Tensor actual=T.add_weighted_outerprods(new int[] {-1},Collections.singletonList(tup));
                assert_tensor_equals(actual,expected);
            }
        }
        {  // is_separable, separable_factors
            Tensor[] Ts_rank_0={
                    new GenTensor(new int[] {2,2,2},new int[8],2),

                    new Mod2Tensor(new int[] {2,2,2},new int[8]),
            };
            for (Tensor T:Ts_rank_0) {
                if (!T.is_separable())
                    throw new RuntimeException("detected separable tensor as non-separable: "+T);
                int[][] ret=T.separable_factors();
                if (ret!=null)
                    throw new RuntimeException("non-empty CPD: "+cpd2str(Collections.singletonList(ret)));
            }

            Tensor[] Ts_rank_1={
                    new GenTensor(new int[] {4},new int[] {1,1,1,1},2),
                    new GenTensor(new int[] {2,2},new int[] {1,1,1,1},2),

//                    new Mod2Tensor(new int[] {4},new int[] {1,1,1,1}),
                    new Mod2Tensor(new int[] {2,2},new int[] {1,1,1,1}),
            };
            for (Tensor T:Ts_rank_1) {
                if (!T.is_separable())
                    throw new RuntimeException("detected separable tensor as non-separable: "+T);
                int[][] ret=T.separable_factors();
                if (ret==null)
                    throw new RuntimeException();
                if (!is_all_zeros(T.add_weighted_outerprods(new int[] {-1},Collections.singletonList(ret))))
                    throw new RuntimeException(cpd2str(Collections.singletonList(ret))+" does not evaluate to "+T);
            }

            Tensor[] Ts_rank_g_1={
                    new GenTensor(new int[] {2,2},new int[] {1,0,0,1},2),

                    new Mod2Tensor(new int[] {2,2},new int[] {1,0,0,1}),
            };
            for (Tensor T:Ts_rank_g_1) {
                if (T.is_separable())
                    throw new RuntimeException("detected non-separable tensor as separable: "+T);
                new ExceptionTester() {
                    void run() {
                        int[][] ret=T.separable_factors();
                        System.err.println("returned "+cpd2str(Collections.singletonList(ret))+" instead of throwing an exception");
                    }
                };
            }
        }

        {  // mat_rank
            Tensor[] Ms={
                    new GenTensor(new int[] {2,2},new int[] {1,0,0,1},2),
                    new GenTensor(new int[] {4,2},new int[] {0,1,0,1,1,1,1,0},2),

                    new Mod2Tensor(new int[] {2,2},new int[] {1,0,0,1}),
                    new Mod2Tensor(new int[] {4,2},new int[] {0,1,0,1,1,1,1,0}),
            };
            int[] expected_ranks={2,2,2,2};
            for (int i=0; i<Ms.length; i++)
                assert_equals(Ms[i].mat_rank(),expected_ranks[i]);
        }
        {  // mat_inv
            // test with a random matrix that is invertible with high probability
            SplittableRandom rnd=new SplittableRandom(1);
            Tensor[] inv_Ms={
                    rnd_gentensor(new int[] {10,10},Integer.MAX_VALUE,rnd),

                    new Mod2Tensor(new int[] {3,3},new int[] {11,0,0, 11,11,11, 0,0,11}),
            };
            for (Tensor M:inv_Ms)
                assert_inv_correct(M);

            Tensor[] non_inv_Ms={
                    new GenTensor(new int[] {3,3},new int[] {1,0,0, 1,1,1, 2,1,1},3),
                    new GenTensor(new int[] {3,2},new int[] {1,0, 0,1, 0,0},2),


                    new Mod2Tensor(new int[] {3,3},new int[] {1,0,0, 1,1,1, 2,1,1}),
                    new Mod2Tensor(new int[] {3,2},new int[] {1,0, 0,1, 0,0}),
            };
            for (Tensor M:non_inv_Ms)
                new ExceptionTester() {
                    void run() {
                        Tensor ret=M.mat_inv();
                        System.err.println(ret);
                    }
                };
        }
        {  // reduce
            int MOD=Integer.MAX_VALUE;
            Tensor[] Ts={
                    // all-zeros tensor, to test handling of tensors with length 0 along an axis
                    new GenTensor(new int[] {2,2,2},new int[8],MOD),
                    new GenTensor(
                            new int[] {3,3,3},
                            new int[] {
                                    0,1,1, 1,0,0, 1,0,0,
                                    1,0,0, 0,0,0, 0,0,0,
                                    1,2,2, 2,0,0, 2,0,0,
                            },
                            MOD
                    ),

                    new Mod2Tensor(new int[] {2,2,2},new int[8]),
                    new Mod2Tensor(
                            new int[] {3,3,3},
                            new int[] {
                                    0,1,1, 1,0,0, 1,0,0,
                                    1,0,0, 0,0,0, 0,0,0,
                                    1,2,2, 2,0,0, 2,0,0,
                            }
                    ),
            };
            int[][] expected_reduced_sizes={
                    {0,0,0},
                    {2,2,2},

                    {0,0,0},
                    {2,2,2},
            };
            for (int ti=0; ti<Ts.length; ti++) {
                Tensor T=Ts[ti];
                int[] expected_rsize=expected_reduced_sizes[ti];

                Tensor.ReduceRet ret=T.reduce();

                if (!Arrays.equals(ret.reduced.shape(),expected_rsize))
                    throw new RuntimeException();
                Tensor uncompressed=ret.reduced;

                int[] perm_rot1=new int[T.ndims()];
                for (int ax=0; ax<T.ndims(); ax++)
                    perm_rot1[ax]=(ax+1)%T.ndims();
                for (int ax=0; ax<T.ndims(); ax++)
                    uncompressed=op0(uncompressed,ret.inv_ops[ax]).permutedims(perm_rot1);
                if (!tensor_equals(uncompressed,T))
                    throw new RuntimeException(ret+"\n"+uncompressed+" != "+T);
            }
        }

        // VecUtils
        // enumerating vectors
        {
            for (int n:new int[] {0,1,3})
                for (int mod:new int[] {2,3,5}) {
                    int[] v=VecUtils.init_vec(n);
                    for (int i=0; v!=null; v=VecUtils.next_vec(v,mod), i++) {
                        if (v.length!=n)
                            throw new RuntimeException(String.format("length !=%d: %s",n,Arrays.toString(v)));
                        int code=0;
                        for (int digit:v)
                            code=code*mod+digit;
                        if (i!=code)
                            throw new RuntimeException(String.format("i=%d != code=%d; v=%s",i,code,Arrays.toString(v)));
                    }
                }
        }
        // enumerating normalized vectors
        {
            for (int n:new int[] {1,3})
                for (int mod:new int[] {2,3,5}) {
                    int[] v=VecUtils.init_normalized_vec(n);
                    int[] prev_v=null;
                    int cnt=0;
                    Set<String> S=new HashSet<>();
                    for (int prev_code=0; v!=null; v=VecUtils.next_normalized_vec(v,mod), cnt++) {
                        S.add(Arrays.toString(v));
                        if (v.length!=n)
                            throw new RuntimeException(String.format("length !=%d: %s",n,Arrays.toString(v)));
                        // check if v is normalized
                        {
                            int lead=0;
                            while (lead<n && v[lead]==0)
                                lead++;
                            if (lead>=n || v[lead]!=1)
                                throw new RuntimeException("not normalized: "+Arrays.toString(v));
                        }
                        int code=0;
                        for (int digit:v)
                            code=code*mod+digit;
                        if (code<=prev_code)
                            throw new RuntimeException(String.format(
                                    "(v,c)=(%s,%d) not lexicographically greater than (prev_v,prev_code)=(%s,%d)",
                                    Arrays.toString(v),code, Arrays.toString(prev_v),prev_code
                            ));
                        prev_v=Arrays.copyOf(v,n);
                        prev_code=code;
                    }
                    int expected_cnt=1;
                    for (int rep=0; rep<n; rep++)
                        expected_cnt*=mod;
                    expected_cnt=(expected_cnt-1)/(mod-1);
                    if (cnt!=expected_cnt)
                        throw new RuntimeException(String.format("incorrect number of normalized vectors for n=%d,mod=%d\nS=%s",n,mod,S));
                }
        }
    }

    private static String cpd2str(List<int[][]> cpd) {
        StringBuilder out=new StringBuilder("CPD {");
        for (int[][] tup:cpd) {
            out.append("\n");
            for (int[] vec:tup)
                out.append(" ").append(Arrays.toString(vec));
        }
        out.append("\n}");
        return out.toString();
    }
    private static int[] flatten(int n0, int n1, int n2, int[][][] data) {
        int[] out=new int[n0*n1*n2];
        for (int i0=0; i0<n0; i0++)
            for (int i1=0; i1<n1; i1++)
                for (int i2=0; i2<n2; i2++)
                    out[i0*(n1*n2) + i1*n2 + i2]=data[i0][i1][i2];
        return out;
    }
    public static Tensor MMTf(int m, int p, int n, int MOD, boolean mod2) {
        int[][][] out=new int[m*p][p*n][n*m];
        for (int i=0; i<m; i++)
            for (int j=0; j<p; j++)
                for (int k=0; k<n; k++)
                    out[i*p+j][j*n+k][k*m+i]=1;
        return (
                mod2?
                        new Mod2Tensor(new int[] {m*p,p*n,n*m},flatten(m*p,p*n,n*m,out)):
                        new GenTensor(new int[] {m*p,p*n,n*m},flatten(m*p,p*n,n*m,out),MOD)
        );
    }
    private static boolean is_all_zeros(Tensor T) {
        int[] data=T.flattened();
        for (int v:data)
            if (v!=0)
                return false;
        return true;
    }
    private static void assert_cpd_correct(Tensor T, List<int[][]> cpd) {
        int[] weights=new int[cpd.size()];
        Arrays.fill(weights,-1);
        Tensor rem=T.add_weighted_outerprods(weights,cpd);
        if (!is_all_zeros(rem))
            throw new RuntimeException(String.format("CPD does not evaluate to tensor: %s\n=!=>\n%s",cpd2str(cpd),T));
    }
    public static void test_search() {
        Tensor[] Ts={
                // GenTensor, supports arbitrary prime modulus
                new GenTensor(new int[] {2,2,2,2},new int[] {2,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,2},1009),
                new GenTensor(new int[] {2,2,2},new int[] {1,0,0,0, 0,1,1,0},2),
                new GenTensor(new int[] {2,2,2},new int[] {1,0,0,1, 0,1,1,0},2),
                new GenTensor(new int[] {2,2,2},new int[] {1,0,0,1, 0,1,1,0},3),
                new GenTensor(new int[] {3,3,3},new int[] {
                        1,0,0, 0,0,0, 0,0,0,
                        0,1,0, 1,0,0, 0,0,0,
                        0,0,1, 0,1,0, 1,0,0,
                },2),
                // shape (2,6,3) --> sorting lengths in descending order requires applying a permutation that is not self-inverting
                MMTf(1,2,3,3,false),

                // Mod2Tensor, only supports mod 2 but is much faster than GenTensor
                new Mod2Tensor(new int[] {2,2,2,2},new int[16]),
                new Mod2Tensor(new int[] {2,2,2,2},new int[] {1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1}),
                new Mod2Tensor(new int[] {2,2,2},new int[] {1,0,0,0, 0,1,1,0}),
                new Mod2Tensor(new int[] {2,2,2},new int[] {1,0,0,1, 0,1,1,0}),
                new Mod2Tensor(new int[] {3,3,3},new int[] {
                        1,0,0, 0,0,0, 0,0,0,
                        0,1,0, 1,0,0, 0,0,0,
                        0,0,1, 0,1,0, 1,0,0,
                }),
                // Bini tensor
                new Mod2Tensor(new int[] {3,4,4},new int[] {
                        1,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,0,
                        0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,1,0,
                        0,1,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0,
                }),
                MMTf(2,2,2,2,true),

                /*
                MMTf(2,2,2) axis-op'd with
                    axis 0: new int[][] {{0,0,0,1},{0,1,0,0},{1,0,0,1},{0,1,1,1}}
                    axis 1: new int[][] {{1,0,1,0},{0,0,0,1},{1,1,0,1},{1,0,0,0}}
                    axis 2: new int[][] {{0,1,0,0},{1,1,1,0},{0,0,1,0},{0,0,0,1}}
                 */
                new Mod2Tensor(new int[] {4,4,4},new int[] {
                        1,1,0,0, 0,0,0,1, 0,0,0,1, 0,0,0,0,
                        0,1,0,0, 0,1,1,0, 0,1,1,0, 0,0,0,0,
                        1,0,0,0, 0,0,0,1, 0,0,1,1, 0,1,0,0,
                        0,1,0,0, 0,1,1,1, 1,0,1,0, 1,1,0,0,
                }),
        };
        int[] exact_ranks={
                2,3,3,2,5,6,
                0,2,3,3,5,6,7,7
        };
        if (Ts.length!=exact_ranks.length)
            throw new RuntimeException();
        for (int i=0; i<Ts.length; i++) {
            Tensor T=Ts[i];
            System.out.println(T);
            for (int r=0; r<=exact_ranks[i]+1; r++) {
                System.out.printf("r=%d\n",r);
                List<int[][]> ret=CPD_DFS.search(T,r);
                if (r<exact_ranks[i]) {
                    if (ret!=null)
                        throw new RuntimeException(cpd2str(ret));
                }
                else {
                    if (ret==null)
                        throw new RuntimeException();
                    System.out.println(cpd2str(ret));
                    assert_cpd_correct(T,ret);
                }
            }
        }
    }
    public static void main(String[] args) {
        test_utils();
        test_search();
    }
}