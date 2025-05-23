## stability-of-duality-pairings
This reprository Contains everything to reproduce the results of the experiments shown in my Bachelor Thesis: "Stable $L^2$-Pairings for Discrete Differential Forms". The implementations were done in the python interface of [NGSolve](https://ngsolve.org/).
In the Notebook folder is a jupyter notebook that performs the experiments and plots their Results. The Results shown in the paper were done with the python files in the folders "Primal Mesh Exp" and "Dual Mesh Exp", on [EULER](https://scicomp.ethz.ch/wiki/Main_Page), ETH Zürichs Super Computer. The data that was calculated is still available in the "data" folders, and the corresponding plots which were shown in the paper in the "plots" folders. 

# Installation
Python and pip are the only prerequisites. 

After cloning the reprository, install the requirements, preferably in a [virtual environment](https://docs.python.org/3/library/venv.html), like this:
```bash
    pip install -r requirements.txt
```

# Running The Jupyter Notebook
To run the Jupyter Notebook in the browser type
```bash
    jupyter notebook
```
in the terminal. 
Then open "main.ipynb", which is located in the notebook directory. 

# Usage Notes
Especially the experiments in 3D for small meshwidths take quite long (~2-3 days), so I recommend running them remotely if the option is available.
If the number of meshwidths or higher polynomial orders is too large, an out of memory error will occur, for the 3D experiments the highest that was possible for me was 5 meshwidths with 3 higher orders.