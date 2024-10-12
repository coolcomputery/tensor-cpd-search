# Tensor canonical polyadic decomposition search

Given a tensor $T\in \mathcal{P}^{n_0\times\dots\ n_{D-1}}$ over an arbitrary ring $\mathcal{P}$,
a rank - $R$ canonical polyadic decomposition (CPD) is a list of factor matrices $A_d\in\mathcal{P}^{n_d\times R},\ 0\le d<D$, such that $\forall i_0,\dots,i_{D-1}:\ T_{i_0,\dots,i_{D-1}} = \sum_{0\le r<R} \prod_{0\le d<D} (A_d)_{i_d,r}$.

The "standard" algorithm (`cpd_search.py`) finds CPDs for $\mathcal{P}$ set to a finite field.

The "border" algorithm (`border_cpd_search.py`) finds CPDs for $\mathcal{P}$ set to a ring of the form $\mathbb{F}[x]/(x^H)$, for some finite field $\mathbb{F}$ and integer $H\ge 1$.
* The motivation of the border algorithm is to find CPDs for tensors $x^{H-1}T$, where all elements of $T$ are in $\mathbb{F}$; such CPDs are known as border rank decompositions.
* However, it also supports tensors with arbitrary elements in $\mathbb{F}[x]/(x^H)$.

## CLI Usage
### Standard
```
usage: cpd_search.py [-h] [--shape SHAPE] [--data DATA]
                     [--matmul MATMUL MATMUL MATMUL] [--pruners [PRUNERS ...]]
                     [-v]
                     mod rank

Tensor CPD (Canonical Polyadic Decomposition) DFS over integers mod a prime

positional arguments:
  mod                   modulo of finite field
  rank                  desired rank threshold of CPD

options:
  -h, --help            show this help message and exit
  --shape SHAPE         shape of tensor to be searched over
  --data DATA           elements of tensor flattened in row-major order
  --matmul MATMUL MATMUL MATMUL
                        matrix multiplication tensor <m,k,n> parameters
                        (original 6-D tensor has adjacent pairs of dimensions
                        flattened)
  --pruners [PRUNERS ...]
                        list of extra DFS pruners to use (worst-case, slows
                        down search by multiplicative factor MOD^(max shape
                        length along one axis))
  -v, --verbose         display search progress
```
#### Example commands
* `python3 cpd_search.py -v 2 5 --shape 3,3,3 --data 0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0`
  * finds a rank-5 CPD of an arbitrary tensor over the field $\mathbb{F}_2$
* `python3 cpd_search.py -v 2 6 --matmul 2 2 2 --pruners rref ranksum`
  * proves that the Strassen tensor does not have a rank-6 CPD over the field $\mathbb{F}_2$
  * uses some pruners to speed up search without affecting correctness

### Border
```
usage: border_cpd_search.py [-h] [--shape SHAPE] [--coeffs COEFFS]
                            [--ground GROUND] [--matmul MATMUL MATMUL MATMUL]
                            [-v]
                            mod H rank

Tensor CPD (Canonical Polyadic Decomposition) DFS over rings F[x]/(x^H), where
F is the integers mod a prime

positional arguments:
  mod                   modulo of coefficient field
  H                     exponent threshold
  rank                  desired rank threshold of CPD

options:
  -h, --help            show this help message and exit
  --shape SHAPE         shape of tensor to be searched over
  --coeffs COEFFS       direct input: coefficients of elements of tensor, with
                        coefficients listed little-endian and elements listed
                        in row-major order
  --ground GROUND       implicitly multiply input by x^{H-1}: elements of
                        original tensor listed in row-major order, where all
                        elements must be in the field F
  --matmul MATMUL MATMUL MATMUL
                        matrix multiplication tensor <m,k,n> parameters,
                        multiplied by x**(H-1)
  -v, --verbose         display search progress
```
#### Example commands
* `python3 border_cpd_search.py -v 2 2 2 --shape 2,2,2 --ground 0,1,1,0,1,0,0,0`
  * finds a rank-2 border CPD of the Waring tensor [[[0,1],[1,0]],[[1,0],[0,0]]], which is commonly used as an example of border rank differing from rank
* `python3 border_cpd_search.py -v 2 2 2 --shape 2,2,2 --coeffs 0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0`
  * finds a CPD of an arbitrary tensor with elements over the ring $\mathbb{F}_2[x]/(x^2)$
