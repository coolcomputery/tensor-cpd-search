import java.util.*;

/**
 * speedtest of CPD_DFS.java on on select tensors
 */
public class Main {
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
    private static Tensor tensor_from(int[] shape, int[][][] data_3d, int MOD, boolean mod2) {
        if (shape.length!=3)
            throw new RuntimeException();
        int[] data=flatten(shape[0],shape[1],shape[2],data_3d);
        return (
                mod2?
                        new Mod2Tensor(shape,data):
                        new GenTensor(shape,data,MOD)
        );
    }
    public static Tensor strassen_extra2(int n, int MOD, boolean mod2) {
        if (n<=3)
            throw new RuntimeException();
        if (MOD!=2 && mod2)
            throw new RuntimeException();
        int[][][] out=new int[n][n][n];
        for (int i=0; i<n; i++) {
            out[i][i][0]=1;
            out[i][0][i]=1;
        }
        out[n-1][n-2][n-3]=1;
        out[n-1][n-3][n-2]=1;
        return tensor_from(new int[] {n,n,n},out,MOD,mod2);
    }
    public static Tensor MMTf(int m, int p, int n, int MOD, boolean mod2) {
        int[][][] out=new int[m*p][p*n][n*m];
        for (int i=0; i<m; i++)
            for (int j=0; j<p; j++)
                for (int k=0; k<n; k++)
                    out[i*p+j][j*n+k][k*m+i]=1;
        return tensor_from(new int[] {m*p,p*n,n*m},out,MOD,mod2);
    }

    public static Tensor nextInvMat(int n, int MOD, SplittableRandom rnd) {
        while (true) {
            int[] data=new int[n*n];
            for (int i=0; i<data.length; i++)
                data[i]=rnd.nextInt(MOD);
            Tensor out=Tensor.make_tensor(new int[] {n,n},data,MOD);
            if (out.mat_rank()==n)
                return out;
        }
    }

    public static void main(String[] args) {
        Tensor[] Ts={
                MMTf(2,2,2,2,true),
                strassen_extra2(4,2,true),
        };

        int[] n_samples={100,10};
        if (n_samples.length!=Ts.length)
            throw new RuntimeException();

        // seed, chosen by Python secrets.randbits(64), then subtracted by 2**64 to fit in signed 64-bit range
        SplittableRandom rnd=new SplittableRandom(-2928283707445556093L);
        for (int test_i=0; test_i<Ts.length; test_i++) {
            Tensor T=Ts[test_i];
            System.out.println(T);
            int real_rk;
            for (int R=0;; R++) {
                System.out.printf("R=%d\n",R);
                List<int[][]> ret=CPD_DFS.search(T,R);
                if (ret!=null) {
                    real_rk=R;
                    System.out.println(cpd2str(ret));
                    break;
                }
            }

            System.out.printf("testing R=%d on random invertible axis contractions\n",real_rk);
            List<List<Long>> stats=new ArrayList<>();
            for (int rep=0; rep<n_samples[test_i]; rep++) {
                Tensor T_in=T;
                int D=T_in.ndims();
                int[] rot1=new int[D];
                for (int ax=0; ax<D; ax++)
                    rot1[ax]=(ax+1)%D;
                for (int ax=0; ax<D; ax++)
                    T_in=T_in.op0(nextInvMat(T.len(ax),T.mod(),rnd)).permutedims(rot1);

                long time_st=System.nanoTime();
                List<int[][]> ret=CPD_DFS.search(T_in,real_rk);
                long elapsed=System.nanoTime()-time_st;

                long tot_work=0;
                for (long v:CPD_DFS.work_stats())
                    tot_work+=v;

                stats.add(Arrays.asList(elapsed,tot_work));

                if (ret==null)
                    throw new RuntimeException();
            }
            System.out.println("stats: "+stats);
        }
    }
}