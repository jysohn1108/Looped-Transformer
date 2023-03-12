import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pdb
import csv

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_np_to_csv(np_data, csv_filename, opt):
    list_path = [opt.output_folder, opt.mode, csv_filename]
    path = os.path.join(*list_path)
    pd.DataFrame(np_data).to_csv(path, header=None, index=None)

def get_nrows_subleq(logn, N):
    nrows_cmds = 3*logn # commands
    nrows_memory = N # memory
    nrows_scratchpad = 3*logn + 2*N # for writing current command line & memory
    nrows_pc = logn # program counter
    nrows_pos_enc = logn # positional encoding
    nrows_buffer = np.maximum(3*logn, 2*N) # for writing intermediate result (step 1 size is 3*logn for storing cmd, step 2 size is 2*N for storing mem[a], mem[b])

    nrows_list = [nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer]

    return nrows_list

def get_row_idx_list_subleq(nrows_list):
    nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer = nrows_list[:-1] # load num_rows except 1
    idx_memory = nrows_cmds
    idx_scratchpad = nrows_cmds + nrows_memory
    idx_pc = -nrows_pc-nrows_pos_enc-nrows_buffer-1
    idx_pos_enc = -nrows_pos_enc-nrows_buffer-1 # row index starting positional encoding
    idx_buff = -nrows_buffer-1
    row_idx_list = [idx_memory, idx_scratchpad, idx_pc, idx_pos_enc, idx_buff]

    return row_idx_list


def numpy_softmax(arr, lam=1, dim=1):
    if dim == 0:
        raise NotImplementedError

    exp_arr = np.exp(lam*arr)
    num_rows = exp_arr.shape[0]
    sum_exp_arr = (np.sum(exp_arr, axis=0, keepdims=True) * np.ones((num_rows,1)))
    
    return exp_arr/sum_exp_arr
    #return nn.Softmax(dim=dim)(torch.from_numpy(arr)).numpy()
            
def numpy_relu(arr):
    return nn.ReLU()(torch.from_numpy(arr)).numpy()


def get_int(x):
    # input x should only contain +1, -1
    if 0 in x:
        raise ValueError
    x = [(1+int(i))/2 for i in np.around(x)]
    n = len(x)
    int_x = sum(x * 2**(np.arange(n-1,-1,-1)))
    
    return int(int_x)


def get_signed_int(x):
    #print(x)
    # input x should only contain +1, -1
    # input x should have the format [b_n, ..., b_1] where b_n is the MSB
    # check the representation for memory M for more details
    # output is signed integer
    if 0 in x:
        raise ValueError
    x = [(1+int(i))/2 for i in np.around(x)]
    n = len(x)
    int_x = 0
    for i in np.arange(1, n):
        int_x += 2**(i-1) * x[n-i]
    if x[0] == 1:
        int_x += -2**(n-1)
    #print(int(int_x))
    return int(int_x)

def save_subleq_results(subleq_results, filename, N, opt):
    list_path = [opt.output_folder, opt.mode, filename]
    path = os.path.join(*list_path)
    with open(path,'w') as out:
        csv_out=csv.writer(out)
        header = ['mem[0]']
        for i in range(N-2):
            header.append('...')
        header = np.append(header, ['mem[N-1]', 'a', 'b', 'c', 'mem[a]', 'mem[b]', 'mem[b]-mem[a]', 'flag', 'p-next'])
        csv_out.writerow(header) 
        #csv_out.writerow(['mem[0]', 'mem[1]', 'mem[2]', 'mem[3]', 'a', 'b', 'c', 'mem[a]', 'mem[b]', 'mem[b]-mem[a]', 'flag', 'p-next']) 
        csv_out.writerows(subleq_results)
    return



def run_manual_subleq(cmds, mem, s, m,  n, N, opt=None, num_loops=None):
    subleq_results = []         # store mem[0], ..., mem[N-1], a, b, c, mem[a], mem[b], mem[b]-mem[a], flag, p_{next} by reading each command
    next_cmd_idx = s+m          # the column number in X, within the range of (s+m):n
    for i in range(num_loops):
        cmd = cmds[next_cmd_idx-s-m] # load command
        curr_result = np.append(mem, np.asarray(cmd))  # mem[0], ..., mem[N-1], a, b, c
        mem_a, mem_b = mem[cmd[0]-s], mem[cmd[1]-s]
        mem_sub = mem_b - mem_a # compute mem[b]-mem[a]
        if mem_sub >= 2**(N-1):
            mem_sub -= 2**N
        elif mem_sub < -2**(N-1):
            mem_sub += 2**N 
        mem[cmd[1]-s] = mem_sub # update mem[b]

        flag = int(mem_sub <= 0)
        if flag:
            next_cmd_idx = cmd[-1] # go to c
        else:
            next_cmd_idx = next_cmd_idx+1 # increase the pointer by 1

        if next_cmd_idx == n: # terminate condition
            next_cmd_idx = 0

        result_others = np.array([mem_a, mem_b, mem_sub, flag, next_cmd_idx]) # mem[a], mem[b], mem[b]-mem[a], flag, p_{next}
        curr_result = np.append(curr_result, result_others)  
        if opt is not None:
            print(curr_result)
        subleq_results.append(tuple(curr_result))

        if next_cmd_idx == 0: 
            break 

    if opt is not None:
        save_subleq_results(subleq_results, 'manual_subleq_results.csv', N, opt)
    
    return subleq_results


def compare_TF_and_manual(subleq_results, our_subleq_results):

    assert(len(subleq_results) == len(our_subleq_results))
    for i in range(len(subleq_results)):
        #print('checking round ', i)
        assert(subleq_results[i] == our_subleq_results[i])
        # if subleq_results[i] != our_subleq_results[i]:
        #     print('error found')
        #     pdb.set_trace()

    print('TF SUBLEQ is working properly!')


def init_input(s,m,n,logn,N,num_rows_X,cmds,nrows_list,row_idx_list,opt=None,mem_given=None):
    
    nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer = nrows_list[:-1] # load num_rows except 1
    idx_memory, idx_scratchpad, idx_pc, idx_pos_enc, idx_buff = row_idx_list
    X = np.zeros((num_rows_X, n))

    # define the last row (indicator of the scratchpad)
    X[-1, :s] = np.ones((1,s))

    # define the 2nd last row block (read buffer)
    # >> This is all zero (nrows_buffer x n)

    # define the 3rd last row block (positional encoding part)
    for i in range(1, n):
        X[idx_pos_enc:idx_buff, i] = get_posit_enc(logn, i)

    # define the scratchpad columns, i.e., initialize the program counter
    curr_pc = X[idx_pos_enc:idx_buff, s+m] # p_{s+m+1}
    X[idx_pc:idx_pos_enc, 0] = curr_pc
    
    # define the memory columns
    ## randomly generate $m$ signed integers, each with precision $N$.
    ## Here, each integer is in the range of [-2^{N-1}+1, 2^{N-1}-1]
    #mem_range = np.arange(-2**(N-1)+1, 2**(N-1)-1) 
    if mem_given is None:
        mem_range = np.arange(-2**(N-2), 2**(N-2)-1) 
        mem_elem_digit = np.random.choice(mem_range, m) # integer stored in the memory
    else:
        mem_elem_digit = mem_given

    ## use our binary representing of the integer 
    mem_elem = np.zeros((N, m)) # each column is [b_n, ..., b_1] representing each integer
    for i in range(m):
        #pdb.set_trace()
        if mem_elem_digit[i] == 0:
            mem_elem[0, i] = -1
        else:
            mem_elem[0, i] = -np.sign(mem_elem_digit[i]) # b_n
        if mem_elem_digit[i] >= 0: # when memory element is positive             
            bin_str = get_bin(mem_elem_digit[i], N-1) # string format '0001'
        else:
            bin_str = get_bin(mem_elem_digit[i] + 2**(N-1), N-1) # string format '0001'
        bin_list = [2*int(d)-1 for d in bin_str] # list format [-1, -1, -1, 1]
        mem_elem[1:, i] = np.array(bin_list) #.reshape(-1, 1)

    if mem_given is None:
        ## re-define 1st and 2nd column of memory, such that EOF command is valid (See Step.7 of Sec.5.1)
        mem_elem[:, 0] = np.ones(N,) * -1   # mem[s+1] = 0 in the paper
        mem_elem[:, 1] = np.ones(N,)        # mem[s+2] = -1 in the paper
        mem_elem_digit[0] = 0
        mem_elem_digit[1] = -1

        save_np_to_csv(mem_elem_digit, 'memory_digit.csv', opt)
        save_np_to_csv(mem_elem, 'memory_bin.csv', opt)

    #print('mem_elem_digit: ', mem_elem_digit)
    #print('mem_elem: ', mem_elem)
    #pdb.set_trace()
    assert(0 not in mem_elem)

    ## store it in X memory
    X[idx_memory:idx_scratchpad, s:s+m] = mem_elem

    # define the commands columns (including the EOF command)
    col_idx = 0
    for cmd in cmds:
        a,b,c = cmd
        # load positional encodings (p_a, p_b, p_c) and put [p_a, p_b, p_c] in the command columns
        p_a = X[idx_pos_enc:idx_buff, a]
        p_b = X[idx_pos_enc:idx_buff, b]
        p_c = X[idx_pos_enc:idx_buff, c]
        cmd_concat = np.concatenate((p_a, p_b), axis=0)
        cmd_concat = np.concatenate((cmd_concat, p_c), axis=0)
        X[:idx_memory, s+m+col_idx] = cmd_concat        
        col_idx += 1   

    return X, mem_elem_digit


def get_posit_enc(length, col_idx):

    posit_enc = np.zeros((length,1)) # positional encoding vector
    bin_str = get_bin(col_idx, length) # string format '0001'
    bin_list = [2*int(d)-1 for d in bin_str] # list format [-1, -1, -1, 1]
    posit_enc = np.array(bin_list) #.reshape(-1, 1)

    assert 0 not in posit_enc
    return posit_enc

def get_bin(x, n=0):
    """
    Get the binary representation of x.
    Parameters
    ----------
    x : int
    n : int
        Minimum number of digits. If x needs less digits in binary, the rest
        is filled with zeros.
    Returns
    -------
    str
    """
    return format(x, 'b').zfill(n)