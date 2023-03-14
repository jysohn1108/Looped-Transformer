# Looped_TF


This is an official implementation of the transformer (TF) architecture suggested in [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196). We first provide the implementation of a looped TF for running subleq commands, and then use this architecture for running three programs (`factorial`, `reverse_list`, and `linear_search`) written in subleq. 

## 1. SUBLEQ

### Description 

The SUBLEQ command implemented in this repository is the one presented in our arXiv paper. For more details about the difference between this and the initially proposed SUBLEQ, see Section 5.1.
To implement a program using SUBLEQ commands one need to write a sequence of commands of the form 
`SUBLEQ a,b,c` where `a` and `b` are the pointers to the memory and `c` is a pointer to a command. Additionally, the user need to specify the initial memory content `[mem[0], mem[1], ..., mem[m-1]]` used for running the SUBLEQ commands. 

After designing any algorithm with SUBLEQ commands, the user may add the command lines in `inputs/subleq_cmds.csv`, and the initial memory content in `inputs/subleq_init_mem.csv`. The supported range of the input parameters `a`,`b`,`c` can be found in the arXiv paper.

Afterwards, the user can run `test_subleq.ipynb` to build the TF architecture implementing the given SUBLEQ commands, and to check whether our TF architecture provides the correct program output.



### Main files \& results

`subleq.py`: This file contains all the building blocks for implementing SUBLEQ, which are 
- read instruction
- read memory
- subtract memory
- write memory
- conditional branching
- error correction

Each building block is implemented with the TF architecture, by defining the weight matrices (Query, Key, Value for attention and Weight/Bias for feed-forward network), following the constructions presented in the paper.  


`test_subleq.ipynb`: This jupyter notebook runs an execution of SUBLEQ commands, without some specific routine. 
This notebook also shows how our TF architecture looks like, for example:


<img src="jupyter_output_figs/read_instr_dpi_600.gif" alt="drawing" width="500"/>
<img src="jupyter_output_figs/read_mem_dpi_600.gif" alt="drawing" width="500"/>





## 2. Programs with SUBLEQ

Currently, we implemented the following three programs that are run interactively. 

- `test_factorial.ipynb`: We compute the factorial ($n!$) of a given positive integer $n$. 

- `test_reverse_list.ipynb`: Given a list $[a_1, a_2, ..., a_n]$, we compute the reverse of the list, i.e., $[a_n, ..., a_2, a_1]$.

- `test_line_search.ipynb`: Given a list $[a_1, a_2, ..., a_n]$ and a target number $T$, we search whether the target number exists in the list, and returns the index in which the target number first appears in the list.


### Converting a program to SUBLEQ commands

Recall that our Looped TF building block implements each SUBLEQ comamnd. Thus, any program written in SUBLEQ commands can be implemented in our Looped TF. Here we describe how we convert a program into SUBLEQ commands.  

Suppose we want to implement a program multiplying two positive numbers, for example $7$ and $9$, stored in two memory positions, say `a` and `b`. Since SUBLEQ implements subtraction and then jumps to another command, the program for multiplication consists of the following steps:

* *We first initialize five memory components*  `[a,b,temp,zero,one] = [7,9,0,0,1]`.

* Command 1: *Save `-a` to a temporary memory `temp`*  with the command **(SUBLEQ a,temp,2)**, where `2` signifies the next command in line, command 2. 

* Command 2: *Subtract 1 from `b`* with the command **(SUBLEQ one,b,EOF)**, where `EOF` is the End Of File command which we reach once `b=0`; otherwise we continue with command 3.

* Command 3: *Subtract the value of `temp` from `a` with the command **(SUBLEQ temp,a,4)**. 

* Command 4: *Form the loop* with the command **(SUBLEQ zero,zero,2)**.

In a similar manner, we implement `factorial`; the only difference is that we use a second counter that we decrease each time to form a partial result of the factorial. Specifically, we first create `n(n-1)` by setting the counter to be `(n-1)`. Once the counter is zeroed, we reset it with a value decreased by one, which is `(n-2)`, until the counter is reseted to zero. In this case, the program terminates and the factorial has been computed and saved in the memory. 

Now for `reverse_list` and `linear_search`, the implementation requires to iterate through the points of the list. Note that this version of SUBLEQ does not allow for this operation. The reason is that we are not able to change a command itself and set the pointer from `a` to `a+1`, but neither to implement `mem[mem[a]]`. Thus, the size of the program grows linearly with the size of the list. On the contrary, the original SUBLEQ command can alter the instructions themselves, since commands and memory are not distinct, containing the size of the program to be constant. However, we follow the implementation that separates commands and memory for simplicity.


