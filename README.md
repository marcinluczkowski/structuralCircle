[![DOI](https://zenodo.org/badge/574263139.svg)](https://zenodo.org/badge/latestdoi/574263139)
# structuralCircle
Sustainable design from used structural elements. A project about building design from used elements in Grasshopper and Rhino. 

The repository contains data, tests and solutions related to the research project at Norwegian University of Science and Technology (NTNU).

The **structuralCircle** project is about aiding the design process with reclaimed building components. In particular, the algorithm is matching available products to the design intent, aiming at environmental impact minimization.  

The solution is written in Python and for the ease of use wrapped into a Grasshopper nodes available for Rhino users. 
 
The implementation includes greedy algorithms, bipartite graphs, and mixed integer linear programming.

Test data contains simulated sets of building elements. The aim of test cases is to compare score - reduction of embodied emissions, and speed of the calculation. 

The algorithms are also explained in this Colab notebook:
https://colab.research.google.com/drive/1BRBxFhya6xCnV-flAPq-Ro8zkxuSnnhC?usp=sharing

# How to run
First, run the console and install the required packages from the list using:

```
pip install -r requirements.txt
```

Then, import the relevant packages and define the input data:

```
# import relevant packages
import pandas as pd
import sys
sys.path.append('./Matching')
import helper_methods as hm
from matching import run_matching
import LCA as lca

# Create two datasets with two elements in each - demand D and supply S:
demand = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Gwp_factor'])
demand.loc['D1'] = {'Length': 4.00, 'Area': 0.06, 'Inertia_moment':0.00030, 'Height': 0.30, 'Gwp_factor':lca.TIMBER_GWP}
demand.loc['D2'] = {'Length': 5.00, 'Area': 0.04, 'Inertia_moment':0.00010, 'Height': 0.20, 'Gwp_factor':lca.TIMBER_GWP}
supply = pd.DataFrame(columns = ['Length', 'Area', 'Inertia_moment', 'Height', 'Gwp_factor'])
supply.loc['S1'] = {'Length': 5.50, 'Area': 0.045, 'Inertia_moment':0.00010, 'Height': 0.20, 'Gwp_factor':lca.TIMBER_REUSE_GWP}
supply.loc['S2'] = {'Length': 4.50, 'Area': 0.065, 'Inertia_moment':0.00035, 'Height': 0.35, 'Gwp_factor':lca.TIMBER_REUSE_GWP}

# create constraint dictionary
constraint_dict = {'Area' : '>=', 'Inertia_moment' : '>=', 'Length' : '>='}

# create optimization formula
score_function_string = "@lca.calculate_lca(length=Length, area=Area, gwp_factor=Gwp_factor, include_transportation=False)"
```

Finally, run the matching with desired methods and display the resultant pairs of elements:
```
# run the matching
result_simple = run_matching(demand, supply, score_function_string=score_function_string, constraints = constraint_dict, add_new = True, sci_milp=False, milp=False, greedy_single=True, bipartite=True)

# display results - matching table
print(hm.extract_pairs_df(result_simple))
# display results - the score
print(hm.extract_results_df(result_simple))
```

# Read more

You can read our journal paper called 'Matching algorithms to assist in designing with reclaimed building elements' here: https://iopscience.iop.org/article/10.1088/2634-4505/acf341
