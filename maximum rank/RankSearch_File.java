import java.util.*;
import java.io.*;
public class RankSearch_File {
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
    public static void main(String[] args) throws IOException {
        BufferedReader in=new BufferedReader(new FileReader("/Users/coolcomputery/Documents/Tensor Decomposition Over Finite Fields/canonical-3x4x3-first-slice-rank-3.txt"));
        int[] SHAPE={3,4,3};
        int MOD=2;
        int SIZE=Tensor.prod(SHAPE);
        List<Tensor> Ts=new ArrayList<>();
        while (true) {
            String line=in.readLine();
            if (line==null)
                break;
            StringTokenizer tok=new StringTokenizer(line);
            int[] flat=new int[SIZE];
            int i=0;
            for (; i<SIZE; i++)
                flat[i]=Integer.parseInt(tok.nextToken());
            if (i!=SIZE)
                throw new RuntimeException(""+i);
            Ts.add(Tensor.make_tensor(SHAPE,flat,MOD));
        }
        System.out.println(Ts.size()+" tensors");

        Map<String,Pruner> pruners=new TreeMap<>();
        pruners.put("0_rref",new RrefPruner());
        pruners.put("1_lask",new LaskPruner());

        CPD_DFS.verbose=false;

        final long SEC_TO_NANO=1000_000_000;
        long time_start=System.nanoTime(), time_mark=0;
        int cnt=0;
        int max_rk=0;
        for (Tensor T:Ts) {
            {
                long time_elapsed=System.nanoTime()-time_start;
                if (time_elapsed>=time_mark) {
                    System.out.printf("t=%.3f cnt=%d\n",time_elapsed/(float)SEC_TO_NANO,cnt);
                    while (time_elapsed>time_mark)
                        time_mark+=Math.pow(10,(int)(Math.log10(Math.max(10,time_mark/(float)SEC_TO_NANO))))*SEC_TO_NANO;
                }
            }

            int real_rk;
            List<int[][]> ret;
            for (int R=0;; R++) {
                ret=CPD_DFS.search(T,R,pruners);
                if (ret!=null) {
                    real_rk=R;
                    break;
                }
            }
            if (real_rk>max_rk) {
                max_rk=real_rk;
                System.out.println("found tensor with shape "+Arrays.toString(SHAPE)+", rank "+real_rk);
                System.out.println(T);
                System.out.println(cpd2str(ret));
            }
            cnt++;
        }
        {
            long time_elapsed=System.nanoTime()-time_start;
            System.out.printf("t=%.3f cnt=%d\n",time_elapsed/(float)SEC_TO_NANO,cnt);
        }
    }
}
