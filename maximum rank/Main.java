import java.util.*;

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

    public static void main(String[] args) {
        Tensor[] Ts={
                new Mod2Tensor(
                        new int[] {3,3,3},

                        // passes 2-nd order rref-pruning along all axes for rank threshold R=4, but has rank 5>4
                        new int[] {
                                1,1,0,
                                0,1,0,
                                0,0,0,

                                0,0,0,
                                0,1,1,
                                0,0,1,

                                1,0,0,
                                0,0,0,
                                1,0,1,
                        }
                ),
        };

        Map<String,Pruner> pruners=new TreeMap<>();
        pruners.put("0_lask",new LaskPruner());
        pruners.put("1_rref",new RrefPruner());

        for (int test_i=0; test_i<Ts.length; test_i++) {
            Tensor T=Ts[test_i];
            System.out.println(T);
            for (int R=0;; R++) {
                System.out.printf("R=%d\n",R);
                List<int[][]> ret=CPD_DFS.search(T,R,pruners);
                if (ret!=null) {
                    System.out.println(cpd2str(ret));
                    break;
                }
            }
        }
    }
}