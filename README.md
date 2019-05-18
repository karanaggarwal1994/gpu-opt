# LiFE-GPU-opt: Optimizing the Linear Fascicle Evaluation Algorithm for Multi-Core Systems

# About
This software is an optimized implementation of the compute-intensive matrix operations of the LiFE algorithm running on GPUs.

# LiFE Code 
The original LiFE[1,2] code can be found using the Github [link](https://github.com/brain-life/encode).

# License
Copyright (2019), Karan Aggarwal, [karan@iisc.ac.in](karan@iisc.ac.in), Uday Bondhugula, [udayb@iisc.ac.in](udayb@iisc.ac.in)

# Funding 
This work was supported in part by a grant (EMR/2016/008015) from the Science and Engineering Research Board (SERB), India through its Extramural Research funding program.

# Dependencies
* [LiFE](https://github.com/brain-life/encode)
* [MatLab](http://www.mathworks.com/products/matlab/)
* [vistasoft](https://github.com/vistalab/vistasoft)
* [Matlab Brain Anatomy (MBA)](https://github.com/francopestilli/mba)
* [CUDA](https://developer.nvidia.com/cuda-downloads)

# Installation
1. Download LiFE software 

	``` git clone https://github.com/brain-life/encode```
	
2. Change directory

	``` cd encode ```
	
3. Download vistasoft software

	``` git clone https://github.com/vistalab/vistasoft```
	
4. Download MBA software

	``` git clone https://github.com/francopestilli/mba```
	
5. Download and Install CUDA

	``` https://developer.nvidia.com/cuda-downloads ```

6. Download demo datasets from the repository [doi:10.5967/K8X63JTX](https://scholarworks.iu.edu/cgi-bin/mdssRequest.pl?file=2022/20995/Demo_Data_for_Multidimensional_Encoding_of_Brain_Connectomes.tar.gz)
	
	``` https://scholarworks.iu.edu/cgi-bin/mdssRequest.pl?file=2022/2099/Demo_Data_for_Multidimensional_Encoding_of_Brain_Connectomes.tar.gz```
7. Unzip the downloaded .tar.gz file 
	``` tar -xvzf Demo_Data_for_Multidimensional_Encoding_of_Brain_Connectomes.tar.gz ``` 
8. Download GPU-opt software

	``` git clone https://github.com/karanaggarwal1994/gpu-opt```

# Running the GPU-opt code
1. Run MATLAB
2. Add the encode folder path to MATLAB search path

	```>>> addpath(genpath('/my/path/to/the/encode/folder/'))```
	
3. Run the script

	```>>> gpu_opt_demo```

# How to cite the paper
Aggarwal, K., Bondhugula U. (2019, June) "Optimizing the Linear Fascicle Evaluation Algorithm for Multi-Core Systems" Accepted to ICS 2019 (to appear).

# References
[1] Pestilli, Franco, Jason D. Yeatman, Ariel Rokem, Kendrick N. Kay, and Brian A. Wandell. Evaluation and statistical inference for human connectomes. Nature methods 11, no. 10 (2014): 1058-1063.

[2] Caiafa, C. and Pestilli, F. Multidimensional encoding of brain connectome. Nature Scientific Reports 7, Article number: 11491 (2017)

