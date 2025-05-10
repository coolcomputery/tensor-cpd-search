import java.util.*;

/**
 * incomplete set of unit and integration tests
 */
public class Test {
    private static void assert_equals(int a, int b) {
        if (a!=b)
            throw new RuntimeException(a+" != "+b);
    }
    private static void assert_equals(boolean a, boolean b) {
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
    private static boolean tensor_equals(Tensor A, Tensor B) {
        if (!Arrays.equals(A.shape(),B.shape()))
            return false;
        if (A.mod()!=B.mod())
            return false;
        return Arrays.equals(A.flattened(),B.flattened());
    }
    private static void assert_tensor_equals(Tensor A, Tensor B) {
        if (!tensor_equals(A,B))
            throw new RuntimeException(A+" != "+B);
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
    public static void test_ModArith() {
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
    public static void test_Tensor() {
        // Tensor static methods
        {  // prod
            assert_equals(Tensor.prod(new int[] {2,3,4}),24);
            new ExceptionTester() {
                void run() {
                    Tensor.prod(new int[] {1<<16,1<<16});
                }
            };
        }

        // Tensor instance methods
        {  // permutedims
            Tensor T=new Mod2Tensor(new int[] {1,2,3},new int[] {0,1,2,3,4,5});
            int[] perm=new int[] {1,2,0};
            Tensor expected=new Mod2Tensor(new int[] {2,3,1},new int[] {0,1,2,3,4,5});

            Tensor actual=T.permutedims(perm);

            assert_tensor_equals(actual,expected);
        }
        {  // scaled_mat_vec
            Tensor T=new Mod2Tensor(
                    new int[] {2,4},
                    new int[] {
                            0,0,1,1,
                            0,1,0,1,
                    }
            );
            int[] vec={0,3,0,5};
            int[] expected={1,0};

            int[] actual=T.scaled_mat_vec(vec,7);

            if (!Arrays.equals(expected,actual))
                throw new RuntimeException();
        }
        {  // op
            Tensor T=new Mod2Tensor(
                    new int[] {2,3,4},
                    new int[] {
                            1,1,0,0,
                            1,0,1,0,
                            0,0,1,0,

                            1,0,0,0,
                            0,1,0,0,
                            0,0,1,0,
                    }
            );
            Tensor M=new Mod2Tensor(
                    new int[] {2,3},
                    new int[] {
                            1,0,0,
                            1,1,1,
                    }
            );
            int ax=1;
            Tensor expected=new Mod2Tensor(
                    new int[] {2,2,4},
                    new int[] {
                            1,1,0,0,
                            0,1,0,0,

                            1,0,0,0,
                            1,1,1,0,
                    }
            );

            Tensor actual=T.op(M,ax);

            assert_tensor_equals(expected,actual);

            new ExceptionTester() {
                void run() {
                    T.op(new Mod2Tensor(new int[] {2,2},new int[4]),ax);
                }
            };
        }
        {  // contraction0_rank
            new ExceptionTester() {
                // ndims too low
                void run() {
                    new Mod2Tensor(new int[] {2,2},new int[4]).contraction0_rank(new int[] {1,0});
                }
            };
            new ExceptionTester() {
                // ndims too high
                void run() {
                    new Mod2Tensor(new int[] {2,2,2,2},new int[16]).contraction0_rank(new int[] {1,0});
                }
            };
            new ExceptionTester() {
                // v too short
                void run() {
                    new Mod2Tensor(new int[] {2,2,2},new int[8]).contraction0_rank(new int[] {1});
                }
            };
            new ExceptionTester() {
                // v too long
                void run() {
                    new Mod2Tensor(new int[] {2,2,2},new int[8]).contraction0_rank(new int[] {1,0,0});
                }
            };

            Tensor T=new Mod2Tensor(
                    new int[] {2,2,3},
                    new int[] {
                            0,1,1,
                            0,0,0,  // will create error if function indexed i1*shape[1]+i2 instead of i1*shape[2]+i2

                            0,0,1,
                            0,1,1,
                    }
            );
            int[][] vs={
                    {1,0},
                    {0,1},
                    {1,1}
            };
            int[] expected_ranks={1,2,2};
            if (vs.length!=expected_ranks.length)
                throw new RuntimeException();
            for (int i=0; i<vs.length; i++) {
                int[] v=vs[i];
                int expected=expected_ranks[i];
                int actual=T.contraction0_rank(v);
                assert_equals(actual,expected);
            }
        }
        {  // contraction0_is_separable
            Tensor T=new Mod2Tensor(
                    new int[] {2,2,2},
                    new int[] {
                            0,0,
                            1,1,

                            0,1,
                            0,1,
                    }
            );
            assert_equals(T.contraction0_is_separable(new int[2]),true);
            assert_equals(T.contraction0_is_separable(new int[] {1,0}),true);
            assert_equals(T.contraction0_is_separable(new int[] {1,1}),false);
        }
        {  // contraction0_fac_vecs
            Tensor T=new Mod2Tensor(
                    new int[] {2,2,3},
                    new int[] {
                            1,0,0,
                            0,1,1,

                            0,1,0,
                            1,0,1,
                    }
            );
            int[] v={1,1};
            int[][] expected={
                    {1,1},{1,1,0}
            };

            int[][] actual=T.contraction0_fac_vecs(v);
            if (!int2_equals(actual,expected))
                throw new RuntimeException();

            // input vector has wrong shape
            new ExceptionTester() {
                void run() {
                    T.contraction0_fac_vecs(new int[] {1,0,0});
                }
            };
            // result has rank >1
            new ExceptionTester() {
                void run() {
                    T.contraction0_fac_vecs(new int[] {1,0});
                }
            };
        }
        {  // cat_oprods()
            Tensor T=new Mod2Tensor(
                    new int[] {1,2,3},
                    new int[] {
                            1,0,0,
                            0,1,0,
                    }
            );
            List<int[][]> tups_1_D=Arrays.asList(
                    new int[][] {{0,1},{1,1,1}},
                    new int[][] {{1,1},{0,1,1}}
            );
            Tensor expected=new Mod2Tensor(
                    new int[] {3,2,3},
                    new int[] {
                            1,0,0,
                            0,1,0,

                            0,0,0,
                            1,1,1,

                            0,1,1,
                            0,1,1,
                    }
            );

            Tensor actual=T.cat_oprods(tups_1_D);

            assert_tensor_equals(expected,actual);
        }

        {  // mat_inv
            // test with a random matrix that is invertible with high probability
            {
                Tensor[] invle_Ms={
                        new Mod2Tensor(
                                new int[] {3,3},
                                new int[] {
                                        1,0,0,
                                        1,1,1,
                                        0,0,1,
                                }
                        ),
                        new Mod2Tensor(
                                new int[] {3,3},
                                new int[] {
                                        0,1,0,
                                        0,0,1,
                                        1,0,0,
                                }
                        ),
                };
                for (Tensor M:invle_Ms) {
                    int R=M.len(0);
                    int[] data=new int[R*R];
                    for (int i=0; i<R; i++)
                        data[i*R+i]=1;
                    Tensor eye=new Mod2Tensor(new int[] {R,R},data);
                    assert_tensor_equals(M.op(M.mat_inv(),0),eye);
                }
            }

            {
                Tensor[] non_invle_Ms={
                        new Mod2Tensor(new int[] {3,3},new int[] {1,0,0, 1,1,1, 2,1,1}),
                        new Mod2Tensor(new int[] {3,2},new int[] {1,0, 0,1, 0,0}),
                };
                for (Tensor M:non_invle_Ms)
                    new ExceptionTester() {
                        void run() {
                            M.mat_inv();
                        }
                    };
            }
        }
        {  // reduce
            Tensor[] Ts={
                    new Mod2Tensor(new int[] {2,2,2},new int[8]),
                    new Mod2Tensor(
                            new int[] {3,3,3},
                            new int[] {
                                    0,1,1, 1,0,0, 1,0,0,
                                    1,0,0, 0,0,0, 0,0,0,
                                    1,2,2, 2,0,0, 2,0,0,
                            }
                    ),
                    new Mod2Tensor(
                            new int[] {2,3},
                            new int[] {
                                    1,0,1,
                                    0,1,1,
                            }
                    ),
            };
            int[][] expected_reduced_sizes={
                    {0,0,0},
                    {2,2,2},
                    {2,2},
            };
            for (int ti=0; ti<Ts.length; ti++) {
                Tensor T=Ts[ti];
                int[] expected_rsize=expected_reduced_sizes[ti];

                Tensor.ReduceRet ret=T.reduce();

                if (!Arrays.equals(ret.reduced.shape(),expected_rsize))
                    throw new RuntimeException();
                Tensor uncompressed=ret.reduced;
//                System.out.println(T+"\n\t"+ret.reduced);

                for (int ax=0; ax<T.ndims(); ax++)
                    if (expected_rsize[ax]==T.len(ax)) {
                        int n=ret.reduced.len(ax);
                        int[] eye=new int[n*n];
                        for (int i=0; i<n; i++)
                            eye[i*n+i]=1;
                        assert_tensor_equals(ret.inv_ops[ax],Tensor.make_tensor(new int[] {n,n},eye,T.mod()));
                    }

                for (int ax=0; ax<T.ndims(); ax++)
                    uncompressed=uncompressed.op(ret.inv_ops[ax],ax);
                assert_tensor_equals(uncompressed,T);
            }
        }
        {  // oprod_ker_basis
            class TestCase {
                Tensor T;
                int[][] tup_2_D;
                int expected_ker_rank;
                public TestCase(Tensor T, int[][] tup_2_D, int expected_ker_rank) {
                    if (T.ndims()<3)
                        throw new RuntimeException();
                    this.T=T;
                    this.tup_2_D=tup_2_D;
                    this.expected_ker_rank=expected_ker_rank;
                }
                public void fail() {
                    new ExceptionTester() {
                        void run() {
                            T.oprod_ker_basis(tup_2_D);
                        }
                    };
                }
                public void test() {
                    Tensor ret=T.oprod_ker_basis(tup_2_D);
                    if (ret.ndims()!=2)
                        throw new RuntimeException();
                    if (!Arrays.equals(ret.shape(),new int[] {expected_ker_rank,T.len(0)+T.len(1)}))
                        throw new RuntimeException(""+ret);
                    if (ret.reduce().reduced.len(0)!=expected_ker_rank)
                        throw new RuntimeException();
                    Tensor T_help; {
                        List<int[][]> tups_1_D=new ArrayList<>();
                        int n1=T.len(1);
                        for (int i1=0; i1<n1; i1++) {
                            int[] e=new int[n1];
                            e[i1]=1;
                            int[][] tup=new int[T.ndims()-1][];
                            tup[0]=e;
                            for (int ax=2; ax<T.ndims(); ax++)
                                tup[ax-1]=Arrays.copyOf(tup_2_D[ax-2],T.len(ax));
                            tups_1_D.add(tup);
                        }
                        T_help=T.cat_oprods(tups_1_D);
                    }
                    for (int row_i=0; row_i<ret.len(0); row_i++) {
                        int[] sol=ret.mat_row(row_i);
                        if (!T_help.contraction0_is_separable(sol))
                            throw new RuntimeException();
                    }
                }
            }
            TestCase[] passing={
                    new TestCase(
                            new Mod2Tensor(new int[] {0,2,3},new int[0]),
                            new int[][] {
                                    {1,0,0},
                            },
                            0
                    ),
                    new TestCase(
                            new Mod2Tensor(
                                    new int[] {1,2,3},
                                    new int[] {
                                            1,0,0,
                                            1,0,0,
                                    }
                            ),
                            new int[][] {
                                    {1,0,0},
                            },
                            1
                    ),
                    new TestCase(
                            new Mod2Tensor(
                                    new int[] {1,2,3},
                                    new int[] {
                                            1,0,0,
                                            0,1,0,
                                    }
                            ),
                            new int[][] {
                                    {1,0,0},
                            },
                            0
                    ),
                    new TestCase(
                            new Mod2Tensor(
                                    new int[] {2,2,3},
                                    new int[] {
                                            1,0,0,
                                            0,1,0,

                                            0,1,0,
                                            1,0,0,
                                    }
                            ),
                            new int[][] {
                                    {1,1,0},
                            },
                            1
                    ),
            };
            for (TestCase test:passing)
                test.test();

            TestCase[] failing={
                    // incorrect length of tup_2_D
                    new TestCase(
                            new Mod2Tensor(
                                    new int[] {1,2,3},
                                    new int[] {
                                            1,0,0,
                                            0,1,0,
                                    }
                            ),
                            new int[][] {
                                    {1,1},
                            },
                            1
                    ),
                    // incorrect shape of items of tup_2_D
                    new TestCase(
                            new Mod2Tensor(
                                    new int[] {1,2,3},
                                    new int[] {
                                            1,0,0,
                                            0,1,0,
                                    }
                            ),
                            new int[][] {
                                    {1,1},
                                    {1,1,0},
                            },
                            1
                    ),
            };
            for (TestCase test:failing)
                test.fail();
        }
    }

    public static void test_LinearSpan() {
        {  // empty span
            Mod2LinearSpan S=new Mod2LinearSpan(3);
            assert_equals(S.rank(),0);
            assert_equals(S.contains_vec(new int[] {2,2,2}),true);  // checking that the input vector is modded by 2
            assert_equals(S.contains_vec(new int[] {1,0,0}),false);
        }
        {  // contains_vec: invalid inputs
            Mod2LinearSpan S=new Mod2LinearSpan(3);
            new ExceptionTester() {
                void run() {
                    S.contains_vec(new int[2]);
                }
            };
            new ExceptionTester() {
                void run() {
                    S.contains_vec(new int[4]);
                }
            };
        }
        {  // add_vec: invalid inputs
            Mod2LinearSpan S=new Mod2LinearSpan(3);
            new ExceptionTester() {
                void run() {
                    S.add_vec(new int[2]);
                }
            };
            new ExceptionTester() {
                void run() {
                    S.add_vec(new int[4]);
                }
            };
        }
        {  // nonempty
            Mod2LinearSpan S=new Mod2LinearSpan(2);
            S.add_vec(new int[] {1,1});
            S.add_vec(new int[] {0,1});  // <-- this vector must be used to modify the previously added vector,
            //  so that S maintains rref
            assert_equals(S.rank(),2);
            assert_equals(S.contains_vec(new int[] {1,0}),true);
        }
        {  // adding already-contained vector
            Mod2LinearSpan S=new Mod2LinearSpan(3);
            S.add_vec(new int[] {1,0,1});
            S.add_vec(new int[] {1,1,1});
            S.add_vec(new int[] {0,1,0});
            assert_equals(S.rank(),2);
        }
    }

    public static void test_VecUtils() {
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
        StringBuilder out=new StringBuilder("[\n");
        for (int[][] tup:cpd) {
            out.append("\t(");
            for (int[] vec:tup)
                out.append(Arrays.toString(vec)).append(",");
            out.append("),\n");
        }
        out.append("]");
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
    public static Tensor MMTf(int m, int p, int n, int MOD) {
        int[][][] out=new int[m*p][p*n][n*m];
        for (int i=0; i<m; i++)
            for (int j=0; j<p; j++)
                for (int k=0; k<n; k++)
                    out[i*p+j][j*n+k][k*m+i]=1;
        return Tensor.make_tensor(new int[] {m*p,p*n,n*m},flatten(m*p,p*n,n*m,out),MOD);
    }
    private static int[] flattened_oprod(int[][] tup) {
        if (tup.length==0)
            throw new RuntimeException();
        int D=tup.length;
        int[] shape=new int[D];
        for (int ax=0; ax<D; ax++)
            shape[ax]=tup[ax].length;
        int size=Tensor.prod(shape);
        int[] out=new int[size];
        for (int i=0; i<size; i++) {
            int prod=1;
            for (int ax=D-1, h=i; ax>=0; h/=shape[ax], ax--)
                prod*=tup[ax][h%shape[ax]];
            out[i]=prod;
        }
        return out;
    }
    private static void assert_cpd_correct(Tensor T, List<int[][]> cpd) {
        int[] expected=T.flattened();
        int[] actual=new int[expected.length];
        for (int[][] tup:cpd) {
            int[] oprod=flattened_oprod(tup);
            for (int i=0; i<expected.length; i++)
                actual[i]=ModArith.add(actual[i],oprod[i],T.mod());
        }
        if (!Arrays.equals(actual,expected))
            throw new RuntimeException(String.format("T!=CPD: %s\n=!=cpd_eval(\n\t%s\n)",T,cpd2str(cpd)));
    }

    public static void test_CPD_DFS() {
        Tensor[] Ts={
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
                MMTf(2,2,2,2),

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
                0,2,3,3,5,6,7,7
        };
        if (Ts.length!=exact_ranks.length)
            throw new RuntimeException();

        Map<String,Pruner> pruners=new TreeMap<>();
        pruners.put("rref",new RrefPruner());
        pruners.put("lask",new LaskPruner());

        List<String> pruner_names=new ArrayList<>(pruners.keySet());
        for (int prune_i=-1; prune_i<pruner_names.size(); prune_i++) {
            System.out.println("================\npruners="+pruners);
            for (int i=0; i<Ts.length; i++) {
                Tensor T=Ts[i];
                System.out.println(T);
                for (int R=0; R<=exact_ranks[i]+1; R++) {
                    System.out.printf("R=%d\n",R);
                    List<int[][]> ret=CPD_DFS.search(
                            T,R,
                            (
                                    prune_i>=0?
                                            Collections.singletonMap(pruner_names.get(prune_i),pruners.get(pruner_names.get(prune_i))):
                                            new TreeMap<>()
                            )
                    );
                    if (R<exact_ranks[i]) {
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
    }
    public static void main(String[] args) {
        test_ModArith();
        test_Tensor();
        test_LinearSpan();
        test_VecUtils();
        test_CPD_DFS();
    }
}