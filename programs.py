import numpy as np
import pandas as pd
import csv
import pdb
import os
from subleq import read_inst, read_mem, subtract_mem, write_mem, conditional_branching, error_correction
from utils import save_np_to_csv, get_nrows_subleq, get_row_idx_list_subleq, init_input, run_manual_subleq, save_subleq_results, compare_TF_and_manual



def subleq_TF(opt): # NOTE: the address index (row, col) starts from 0, which is different from the notation in the paper

    # initialize the size of each sub-block (s,m,n-s-m) and the precision (N)
    mylist = [int(item) for item in opt.subleq_smnN.split(',')]
    (s,m,n,N) = mylist
    #print(s,m,n,N)

    logn = int(np.log2(n))
    # Specify the number of rows of X = 8*logn + 3*N + max(3*logn, 2*N) + 1 (Here, 1 is for indicator of the scratchpad)
    nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer = get_nrows_subleq(logn, N) 
    num_rows_X = nrows_cmds + nrows_memory + nrows_scratchpad + nrows_pc + nrows_pos_enc + nrows_buffer + 1 
    nrows_list = [nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer, 1]
    row_idx_list = get_row_idx_list_subleq(nrows_list)

    # initialize the commands "cmds" & input "X"
    cmds = load_cmd_triplets(s,m,n,opt)
    X, mem = init_input(s,m,n,logn,N,num_rows_X,cmds,nrows_list,row_idx_list,opt=opt,mem_given=None)     # initialize the input X
    #print('shape of input X: ', X.shape)
    #print('X: \n',X)
    assert(X.shape[0] == num_rows_X)
    assert(X.shape[1] == n)
    save_np_to_csv(X,'R1_X0.csv',opt)


    # 1. run SUBLEQ manually
    subleq_results = run_manual_subleq(cmds, mem, s, m, n, N, opt=opt, num_loops=opt.num_loops)

    # 2. run SUBLEQ using our TF blocks
    our_subleq_results = []
    for i in range(opt.num_loops):
        #print('we are in the loop ', i+1)
        X1, _, TF_read_inst = read_inst(X,s,m,n,logn,num_rows_X,N,i+1,nrows_list,row_idx_list,opt) # Step 1. read instruction & check a, b, c 
        X2, _, TF_mem = read_mem(X1,n,logn,num_rows_X,N,i+1,nrows_list,row_idx_list,opt) # Step 2. read memory & check mem[a], mem[b]
        X3, _, *TF_subtract_mem = subtract_mem(X2,n,logn,num_rows_X,N,i+1,nrows_list,row_idx_list,opt) # Step 3. subtract memory & check mem[b]-mem[a] 
        X4, _, TF_write_mem = write_mem(X3,n,logn,num_rows_X,N,i+1,nrows_list,row_idx_list,opt) # Step 4. write memory 
        X5, _, *TF_cond_branch = conditional_branching(X4,n,logn,num_rows_X,N,i+1,nrows_list,row_idx_list,opt) # Step 5. conditional branching & check flag, p_{next}
        X6, _, TF_err_corr = error_correction(X5,n,logn,num_rows_X,N,i+1,nrows_list,row_idx_list,opt) # Step 6. error correction        

        X = X6 # go to the next loop
        our_subleq_results.append(tuple(opt.our_curr_result))


    save_subleq_results(our_subleq_results, 'TF_subleq_results.csv', N, opt)


    # 3. compare the results
    compare_TF_and_manual(subleq_results, our_subleq_results)






def load_cmd_triplets(s,m,n,opt): 
    cmds = []
    for i in range(n-s-m-1): # last command (EOF) is omitted
        mem_add = np.arange(s+2, s+m) # s, s+1 is not allowed to change (for EOF)
        a,b = np.random.choice(mem_add, 2, replace=False)
        cmd_add = np.arange(s+m, n) 
        c = np.random.choice(cmd_add, 1)[0]
        cmd = (a,b,c)
        cmds.append(cmd) #cmds = (*cmds, cmd)
    # last command (EOF)
    a, b, c = s, s+1, n-1
    cmd = (a,b,c)
    cmds.append(cmd) #cmds = (*cmds, cmd)

    print('cmd_triplets: ', cmds)
    save_cmd_triplets(cmds, opt)
    return cmds

def save_cmd_triplets(cmds, opt):
    filename = 'cmds.csv'
    list_path = [opt.output_folder, opt.mode, filename]
    path = os.path.join(*list_path)
    with open(path,'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['a','b','c'])
        csv_out.writerows(cmds)
    return


