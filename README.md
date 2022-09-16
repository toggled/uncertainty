Codes for Uncertainty of Uncertain graphs
-----------
<h> Requirements: </h>
 * Python 3.8
 * Networkx
 * Matplotlib
 * Scipy


<h> Experiments: </h>
TO DO

<h> Demo: </h>
TO DO

Running RelComp/ to generate Induced Subgraphs:
--------

<h> Requirements </h>

- CMAKE 3.0
- gcc-8 g++-8
- To run with g++-7: Uncomment lines 1-2 in CMakeLists.txt
  - `set(CMAKE_C_COMPILER "gcc-7")`
  - `set(CMAKE_CXX_COMPILER "g++-7")`
- Boost (minimum 1.54)

<h> Compile </h>

- `mkdir build`
- `cd build`
- `cmake ..`
- `make`

<h> Run for Reachability query </h>

- Unweighted graph:
  - `./RelComp -i test_graph.txt -l [full path of the directory containing test_graph.txt] -d decomp/ -s test_sourcetarget.txt -m 5 -w 0`
- Weighted graph:
  - `./RelComp -i test_wgraph.txt -l [full path of the directory containing test_wgraph.txt] -d decomp/ -s test_sourcetarget.txt -m 5 -w 1`
