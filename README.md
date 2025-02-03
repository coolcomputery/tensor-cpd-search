# Tensor canonical polyadic decomposition search

Given a tensor $T\in \mathcal{P}^{n_0\times\dots\ n_{D-1}}$ over an arbitrary ring $\mathcal{P}$,
a rank - $R$ canonical polyadic decomposition (CPD) is a list of factor matrices $A_d\in\mathcal{P}^{n_d\times R},\ 0\le d<D$, such that $\forall i_0,\dots,i_{D-1}:\ T_{i_0,\dots,i_{D-1}} = \sum_{0\le r<R} \prod_{0\le d<D} (A_d)_{i_d,r}$.

WLOG assume $n_0\ge \dots\ge n_{D-1}$ and $T$ is concise (i.e. for each axis $d$, the set of $T_{\dots,:,i_d,:,\dots}$ over all $i_d$ is linearly independent).

## Standard CPD
Finds CPDs over a finite field $\mathcal{P}=\mathbb{F}$.

### Current fastest algorithm
Java folder: `cpd/skip-axis/` (paper to be released)
* time complexity: $O^*(|\mathbb{F}|^{\min(R,\ \sum_{d\ge 2} n_d) + (R-n_0)(\sum_{d\ne 0} n_d)})$
* **Note**: only supports finite field $\mathbb{F}$ with prime order
* files:
  * `CPD_DFS.java`: search algorithm
  * `Test.java`: incomplete set of unit and integration tests
  * `Main.java`: speedtests

### Older algorithm
Standalone CLI: `cpd/original/cpd_search.py` (following [paper 1](https://arxiv.org/abs/2411.14676))
* time complexity: $O^*(|\mathbb{F}|^{n_0 + (R-n_0)(\sum_d n_d)})$
* **Note**: only supports finite field $\mathbb{F}$ with prime order

#### Example commands
* `python3 cpd_search.py -v 2 5 --shape 3,3,3 --data 0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0`
  * finds a rank-5 CPD of an arbitrary tensor over the field $\mathbb{F}_2$
* `python3 cpd_search.py -v 2 6 --matmul 2 2 2 --pruners rref ranksum`
  * proves that the Strassen tensor does not have a rank-6 CPD over the field $\mathbb{F}_2$
  * uses some pruners to speed up search without affecting correctness

## Border CPD
Finds CPDs over the ring $\mathbb{F}[x]/(x^H)$, for some finite field $\mathbb{F}$ and integer $H\ge 1$.

Standalone CLI: `border-cpd/border_cpd_search.py` (following [paper 1](https://arxiv.org/abs/2411.14676))
* time complexity: $O^*(|\mathbb{F}|^{H\sum_d \sum_{1\le r\le R} \min(n_d,r)})$
* **Note**: only supports finite field $\mathbb{F}$ with prime order
### Example commands
* `python3 border_cpd_search.py -v 2 2 2 --shape 2,2,2 --ground 0,1,1,0,1,0,0,0`
  * finds a rank-2 border CPD of the Waring tensor [[[0,1],[1,0]],[[1,0],[0,0]]], which is commonly used as an example of border rank differing from rank
* `python3 border_cpd_search.py -v 2 2 2 --shape 2,2,2 --coeffs 0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0`
  * finds a CPD of an arbitrary tensor with elements over the ring $\mathbb{F}_2[x]/(x^2)$
