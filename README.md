# RNA-aggregation
Code to accompany Kimchi et al. 2022

computational_enumeration.py exactly calculates the partition functions for a given value of F_b using the dynamic programming algorithm detailed in the paper. Because of file upload size limits, the results themselves cannot be uploaded directly but they can be regenerated by running this code, or can be shared directly if you reach out. For a given repeat, the code takes approximately 15 min to run. The repeat is defined by several parameters, most notably len_repeat_bp (the number of nucleotides comprising each complementary section of the repeat), len_linker (the number of nts comprising each non-complementary section) and FE_from_one_bp (the equivalent of F_b in the main text of the paper).

figures_for_paper_github.py loads the calculated partition functions and makes the plots shown in the paper.

ways_to_connect_m/ gives stored results necessary to correct the partition functions to calculate only connected clusters

The code was developed in Python version 3.5, and tested in 3.9.

Any questions or correspondence can be addressed to okimchi@princeton.edu.
