
import numpy as np
import torch
import torch.nn as nn
import pdb
import os
#from option import opt
from utils import save_np_to_csv, numpy_softmax, numpy_relu, get_int, get_signed_int
import matplotlib.pyplot as plt

def read_inst(X, s, m, n, logn, num_rows_X, N, round_idx, nrows_list,row_idx_list, opt, our_curr_result=None, lam=10): # read the instruction of subleq

    # load the variables explaining the structure of X
    nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer = nrows_list[:-1] # load num_rows except 1
    idx_memory, idx_scratchpad, idx_pc, idx_pos_enc, idx_buff = row_idx_list
    size_inst = nrows_cmds # size of instruction (commands) = 3*logn
    size_pos_enc = idx_buff - idx_pos_enc # size of positional encoder = logn
    size_mem = idx_scratchpad - idx_memory # size of memory = N    
    idx_scratch_cmd = idx_scratchpad+2*size_mem

    # initialize opt.our_curr_result
    init_memory = X[idx_memory:idx_memory+size_mem, s:s+m]
    init_memory_int = [get_signed_int(init_memory[:, i]) for i in range(init_memory.shape[1]) ]
    if opt is not None:
        opt.our_curr_result = init_memory_int # store current memory 
    else:
        our_curr_result = init_memory_int # store current memory 

    # define TF network f for the read operation  
    f = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=4*size_inst, num_heads=1, opt=opt, lam=lam) # num_rows_W = 12*logn

    # specify the parameter Q,K,V for attention
    f.Q[:, idx_pc:idx_pos_enc] = np.eye(size_pos_enc) # rows for the p_c (program counter)
    f.Q[:, idx_pos_enc:idx_buff] = np.eye(size_pos_enc) # rows for the positional encoding
    f.K = f.Q
    f.V[idx_buff:idx_buff+size_inst, :nrows_cmds] = np.eye(size_inst) # read "commands" to the buffer
    f.V[idx_buff:idx_buff+size_inst, idx_scratch_cmd:idx_scratch_cmd+size_inst] = np.eye(size_inst) # read "current command" to the buffer 
    #print(f.V @ X)

    # specify the parameters W1,W2,b1,b2 for the feed-forward network
    large_const = 100 # a large constant (used for the feed-forward network)
    ## W1 (first 3*logn rows)
    f.W1[:size_inst, idx_scratch_cmd:idx_scratch_cmd+size_inst] = -2 * np.eye(size_inst)  # matrix for v_orig^(1) = current command
    f.W1[:size_inst, idx_buff:idx_buff+size_inst] = 2 * np.eye(size_inst)        # matrix for v_new = buffer
    f.W1[:size_inst, -1] = large_const * np.ones(size_inst,)      # matrix for b = indicator for scratchpad   
    ## W1 (second 3*logn rows)
    f.W1[size_inst:2*size_inst, idx_scratch_cmd:idx_scratch_cmd+size_inst] = 2 * np.eye(size_inst)  # matrix for v_orig^(1)
    f.W1[size_inst:2*size_inst, idx_buff:idx_buff+size_inst] = -2 * np.eye(size_inst)            # matrix for v_new
    f.W1[size_inst:2*size_inst, -1] = large_const * np.ones(size_inst,)          # matrix for b    
    ## W1 (third 3*logn rows)
    f.W1[2*size_inst:3*size_inst, idx_buff:idx_buff+size_inst] = np.eye(size_inst)        # matrix for v_new
    ## W1 (fourth 3*logn rows)
    f.W1[3*size_inst:, idx_buff:idx_buff+size_inst] = -1 * np.eye(size_inst)        # matrix for v_new
    ## b1
    f.b1[:2*size_inst] = -1 * large_const * np.ones((2*size_inst,n))
    ## W2
    f.W2[idx_scratch_cmd:idx_scratch_cmd+size_inst, :size_inst] = np.eye(size_inst)
    f.W2[idx_scratch_cmd:idx_scratch_cmd+size_inst, size_inst:2*size_inst] = -1 * np.eye(size_inst)
    f.W2[idx_buff:idx_buff+size_inst, 2*size_inst:3*size_inst] = -1 * np.eye(size_inst)
    f.W2[idx_buff:idx_buff+size_inst, 3*size_inst:] = np.eye(size_inst)
    ## b2 = 0

    X1 = f.forward(X, heatmap_title='R{}_Read Instruction'.format(round_idx))
    # check the difference btw X1 & X
    #print('X1 - X: \n', X1 - X)
    if opt is not None and opt.save_all:
        save_np_to_csv(X1-X,'R{}_X1-X0.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X1-X),'R{}_X1-X0_around.csv'.format(round_idx),opt)
        save_np_to_csv(X1,'R{}_X1.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X1),'R{}_X1_around.csv'.format(round_idx),opt)


    # check a,b,c and store it in opt.our_curr_result
    inst = X1[idx_scratch_cmd:idx_scratch_cmd+size_inst, 0]
    cmd_a = inst[:size_pos_enc]
    cmd_b = inst[size_pos_enc:2*size_pos_enc]
    cmd_c = inst[2*size_pos_enc:3*size_pos_enc]
    #print(cmd_a, cmd_b, cmd_c)
    cmd_a_int = get_int(cmd_a)
    cmd_b_int = get_int(cmd_b)
    cmd_c_int = get_int(cmd_c)
    #print(cmd_a_int, cmd_b_int, cmd_c_int)

    extracted_cmd = [cmd_a_int, cmd_b_int, cmd_c_int]
    if opt is not None:
        opt.our_curr_result = np.append(opt.our_curr_result, extracted_cmd)
    else:
        our_curr_result = np.append(our_curr_result, extracted_cmd)


    return X1, our_curr_result, f




def read_mem(X, n, logn, num_rows_X, N, round_idx, nrows_list,row_idx_list, opt, our_curr_result=None, lam=10):

    # load the variables explaining the structure of X
    nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer = nrows_list[:-1] # load num_rows except 1
    idx_memory, idx_scratchpad, idx_pc, idx_pos_enc, idx_buff = row_idx_list
    size_inst = nrows_cmds # size of instruction (commands) = 3*logn
    size_pos_enc = idx_buff - idx_pos_enc # size of positional encoder = logn
    size_mem = idx_scratchpad - idx_memory # size of memory = N    
    idx_scratch_cmd = idx_scratchpad+2*size_mem # 3*logn + N + 2N

    # 1. read mem[a] and write it in the scratchpad (head 1)
    ## source pointer -- [idx_scratch_cmd : idx_scratch_cmd+size_pos_enc, 0]
    ## source data location -- rows containing the memory [idx_memory : idx_memory+size_mem]
    ## target data location -- [idx_scratchpad:idx_scratchpad+size_mem]
    ## buffer location (only the part we use) -- [idx_buff:idx_buff+size_mem]

    ## define TF network f for the read operation 
    f = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=8*size_mem, num_heads=2, opt=opt, lam=lam) 

    ## specify the parameter Q,K,V for attention
    f.Q1[:, idx_scratch_cmd:idx_scratch_cmd+size_pos_enc] = np.eye(size_pos_enc) # rows for p_a
    f.Q1[:, idx_pos_enc:idx_pos_enc+size_pos_enc] = np.eye(size_pos_enc) # rows for all positional encoding
    f.K1 = f.Q1    
    f.V1[idx_buff:idx_buff+size_mem, idx_memory:idx_memory+size_mem] = np.eye(size_mem) # read "memory" to the buffer, for mem[a]
    f.V1[idx_buff:idx_buff+size_mem, idx_scratchpad:idx_scratchpad+size_mem] = np.eye(size_mem) # read "current memory" to the buffer, for mem[a]

    f.Q2[:, idx_scratch_cmd+size_pos_enc:idx_scratch_cmd+2*size_pos_enc] = np.eye(size_pos_enc) # rows for p_b
    f.Q2[:, idx_pos_enc:idx_pos_enc+size_pos_enc] = np.eye(size_pos_enc) # rows for all positional encoding
    f.K2 = f.Q2    
    f.V2[idx_buff+size_mem:idx_buff+2*size_mem, idx_memory:idx_memory+size_mem] = np.eye(size_mem) # read "memory" to the buffer, for mem[b]
    f.V2[idx_buff+size_mem:idx_buff+2*size_mem, idx_scratchpad+size_mem:idx_scratchpad+2*size_mem] = np.eye(size_mem) # read "current memory" to the buffer, for mem[b]

    ## specify the parameters W1,W2,b1,b2 for the feed-forward network
    large_const = 100 # a large constant (used for the feed-forward network)
    ### W1 (first size_mem rows for mem[a])
    f.W1[:size_mem, idx_scratchpad:idx_scratchpad+size_mem] = -2 * np.eye(size_mem)  # matrix for current memory for mem[a]
    f.W1[:size_mem, idx_buff:idx_buff+size_mem] = 2 * np.eye(size_mem)        # matrix for buffer for mem[a]
    f.W1[:size_mem, -1] = large_const * np.ones(size_mem,)      # matrix for b = indicator for scratchpad   
    ### W1 (second size_mem rows for mem[a])
    f.W1[size_mem:2*size_mem, idx_scratchpad:idx_scratchpad+size_mem] = 2 * np.eye(size_mem)  # matrix for current memory for mem[a]
    f.W1[size_mem:2*size_mem, idx_buff:idx_buff+size_mem] = -2 * np.eye(size_mem)            # matrix for buffer for mem[a]
    f.W1[size_mem:2*size_mem, -1] = large_const * np.ones(size_mem,)          # matrix for b    
    ### W1 (third size_mem rows for mem[a])
    f.W1[2*size_mem:3*size_mem, idx_buff:idx_buff+size_mem] = np.eye(size_mem)        # matrix for buffer for mem[a]
    ### W1 (fourth size_mem rows for mem[a])
    f.W1[3*size_mem:4*size_mem, idx_buff:idx_buff+size_mem] = -1 * np.eye(size_mem)        # matrix for buffer for mem[a]
    ### W1 (first size_mem rows for mem[b])
    f.W1[4*size_mem:5*size_mem, idx_scratchpad+size_mem:idx_scratchpad+2*size_mem] = -2 * np.eye(size_mem)  # matrix for current memory for mem[b]
    f.W1[4*size_mem:5*size_mem, idx_buff+size_mem:idx_buff+2*size_mem] = 2 * np.eye(size_mem)        # matrix for buffer for mem[b]
    f.W1[4*size_mem:5*size_mem, -1] = large_const * np.ones(size_mem,)      # matrix for b = indicator for scratchpad   
    ### W1 (second size_mem rows for mem[b])
    f.W1[5*size_mem:6*size_mem, idx_scratchpad+size_mem:idx_scratchpad+2*size_mem] = 2 * np.eye(size_mem)  # matrix for current memory for mem[b]
    f.W1[5*size_mem:6*size_mem, idx_buff+size_mem:idx_buff+2*size_mem] = -2 * np.eye(size_mem)            # matrix for buffer for mem[b]
    f.W1[5*size_mem:6*size_mem, -1] = large_const * np.ones(size_mem,)          # matrix for b    
    ### W1 (third size_mem rows for mem[b])
    f.W1[6*size_mem:7*size_mem, idx_buff+size_mem:idx_buff+2*size_mem] = np.eye(size_mem)        # matrix for buffer for mem[b]
    ### W1 (fourth size_mem rows for mem[b])
    f.W1[7*size_mem:, idx_buff+size_mem:idx_buff+2*size_mem] = -1 * np.eye(size_mem)        # matrix for buffer for mem[b]
    ### b1
    f.b1[:2*size_mem] = -1 * large_const * np.ones((2*size_mem,n)) # for mem[a]
    f.b1[4*size_mem:6*size_mem] = -1 * large_const * np.ones((2*size_mem,n)) # for mem[b]

    ### W2 (for mem[a])
    f.W2[idx_scratchpad:idx_scratchpad+size_mem, :size_mem] = np.eye(size_mem)
    f.W2[idx_scratchpad:idx_scratchpad+size_mem, size_mem:2*size_mem] = -1 * np.eye(size_mem)
    f.W2[idx_buff:idx_buff+size_mem, 2*size_mem:3*size_mem] = -1 * np.eye(size_mem)
    f.W2[idx_buff:idx_buff+size_mem, 3*size_mem:4*size_mem] = np.eye(size_mem)
    ### W2 (for mem[b])
    f.W2[idx_scratchpad+size_mem:idx_scratchpad+2*size_mem, 4*size_mem:5*size_mem] = np.eye(size_mem)
    f.W2[idx_scratchpad+size_mem:idx_scratchpad+2*size_mem, 5*size_mem:6*size_mem] = -1 * np.eye(size_mem)
    f.W2[idx_buff+size_mem:idx_buff+2*size_mem, 6*size_mem:7*size_mem] = -1 * np.eye(size_mem)
    f.W2[idx_buff+size_mem:idx_buff+2*size_mem, 7*size_mem:] = np.eye(size_mem)
    ### b2 = 0

    X2 = f.forward(X, heatmap_title='R{}_Read Memory'.format(round_idx))
    if opt is not None and opt.save_all:    
        save_np_to_csv(X2,'R{}_X2.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X2-X),'R{}_X2-X1_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X2),'R{}_X2_around.csv'.format(round_idx),opt)


    # check mem[a], mem[b] and store it in opt.our_curr_result
    mem_a_b = X2[idx_scratchpad:idx_scratchpad+2*size_mem, 0]
    mem_a = mem_a_b[:size_mem]
    mem_b = mem_a_b[size_mem:2*size_mem]
    #print(mem_a, mem_b)
    mem_a_int = get_signed_int(mem_a)
    mem_b_int = get_signed_int(mem_b)
    #print(mem_a_int, mem_b_int)

    extracted_mem = [mem_a_int, mem_b_int]
    if opt is not None:
        opt.our_curr_result = np.append(opt.our_curr_result, extracted_mem)
    else:
        our_curr_result = np.append(our_curr_result, extracted_mem)

    return X2, our_curr_result, f


def subtract_mem(X,n,logn,num_rows_X,N,round_idx,nrows_list,row_idx_list,opt, our_curr_result=None, lam=10):

    # load the variables explaining the structure of X
    nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer = nrows_list[:-1] # load num_rows except 1
    idx_memory, idx_scratchpad, idx_pc, idx_pos_enc, idx_buff = row_idx_list
    size_inst = nrows_cmds # size of instruction (commands) = 3*logn
    size_pos_enc = idx_buff - idx_pos_enc # size of positional encoder = logn
    size_mem = idx_scratchpad - idx_memory # size of memory = N    
    idx_scratch_cmd = idx_scratchpad+2*size_mem # 3*logn + N + 2N

    # 1. save the flipped b_r in the buffer
    f1 = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=size_mem, num_heads=1, opt=opt, lam=lam) # NOTE: num_rows_Q does not matter, since Q=K=V=0
    f1.W1[:, idx_scratchpad:idx_scratchpad+size_mem] = -1 * np.eye(size_mem) # extract -b_r
    f1.W2[idx_buff:idx_buff+size_mem, :] = 2 * np.eye(size_mem) # put 2 ReLU(-b_r) in the buffer
    f1.b2[idx_buff:idx_buff+size_mem, 0] = -1 * np.ones(size_mem,) # add -1 only for the first column
    X1 = f1.forward(X)
    #print('X3a: ', X1)  # 1st buffer contains b_r^{flipped}
    if opt is not None and opt.save_all:
        save_np_to_csv(X1,'R{}_X3a.csv'.format(round_idx),opt)
        save_np_to_csv(X1-X,'R{}_X3a-X2.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X1),'R{}_X3a_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X1-X),'R{}_X3a-X2_around.csv'.format(round_idx),opt)



    # 2. add 1 to the buffer elements, getting b_{-r}
    f2 = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=6*size_mem, num_heads=1, opt=opt, lam=lam) # NOTE: num_rows_Q does not matter, since Q=K=V=0
    assert(size_mem == N)
    ref = idx_buff
    for i in np.arange(1, N+1): # for all components of mem[a]
        for j in np.arange(1, i+1):
            elem = (2**(j-1))/2
            f2.W1[6*(N-i), ref+N-j] = elem 
            f2.W1[6*(N-i)+1, ref+N-j] = elem 
            f2.W1[6*(N-i)+2, ref+N-j] = -elem 
            f2.W1[6*(N-i)+3, ref+N-j] = -elem 
            f2.W1[6*(N-i)+4, ref+N-j] = elem 
            f2.W1[6*(N-i)+5, ref+N-j] = elem 

        # bias
        s_bias = (2**(i) - 1)/2 + 1 # bias to get the sum s, and +1 is for getting b_{-r}
        f2.b1[6*(N-i), 0] = (s_bias - 2**(i-1) + 1) #* np.ones(n,)
        f2.b1[6*(N-i)+1, 0] = (s_bias - 2**(i-1)) #* np.ones(n,)
        f2.b1[6*(N-i)+2, 0] = (-s_bias + 2**(i)) #* np.ones(n,)
        f2.b1[6*(N-i)+3, 0] = (-s_bias + 2**(i) - 1) #* np.ones(n,)
        f2.b1[6*(N-i)+4, 0] = (s_bias - 3*(2**(i-1)) + 1) #* np.ones(n,)
        f2.b1[6*(N-i)+5, 0] = (s_bias - 3*(2**(i-1))) #* np.ones(n,)

        f2.W2[idx_buff+size_mem+N-i, 6*(N-i):6*(N-i+1)] = 2*np.array([1, -1, 1, -1, 1, -1])
    f2.b2[idx_buff+size_mem:idx_buff+2*size_mem, 0] = -3 * np.ones(N,)  # 2 c_i - 1 (to convert to +1, -1 notation), where c_i contains (\sum (ReLU) -1)

    X2 = f2.forward(X1)
    #print('X3b: ', X2) # 2nd buffer contains b_{-r}
    if opt is not None and opt.save_all:
        save_np_to_csv(X2,'R{}_X3b.csv'.format(round_idx),opt)
        save_np_to_csv(X2-X1,'R{}_X3b-X3a.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X2),'R{}_X3b_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X2-X1),'R{}_X3b-X3a_around.csv'.format(round_idx),opt)


    # 3. add b_{-r} to b_r & b_s
    f3 = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=14*size_mem, num_heads=1, opt=opt, lam=lam) # NOTE: num_rows_Q does not matter, since Q=K=V=0
    
    assert(size_mem == N)
    ref = idx_buff+size_mem # b_{-r}
    ref1 = idx_scratchpad # b_r
    ref2 = idx_scratchpad+size_mem # b_s
    
    # b_{s-r} = b_s + b_{-r}
    for i in np.arange(1, N+1): # for all components of mem[b]
        for j in np.arange(1, i+1):
            elem = (2**(j-1))/2
            f3.W1[6*(N-i), ref+N-j] = elem 
            f3.W1[6*(N-i)+1, ref+N-j] = elem 
            f3.W1[6*(N-i)+2, ref+N-j] = -elem 
            f3.W1[6*(N-i)+3, ref+N-j] = -elem 
            f3.W1[6*(N-i)+4, ref+N-j] = elem 
            f3.W1[6*(N-i)+5, ref+N-j] = elem 
            
            f3.W1[6*(N-i), ref2+N-j] = elem 
            f3.W1[6*(N-i)+1, ref2+N-j] = elem 
            f3.W1[6*(N-i)+2, ref2+N-j] = -elem 
            f3.W1[6*(N-i)+3, ref2+N-j] = -elem 
            f3.W1[6*(N-i)+4, ref2+N-j] = elem 
            f3.W1[6*(N-i)+5, ref2+N-j] = elem 
        # bias
        s_bias = (2**(i) - 1) # bias to get the sum s
        f3.b1[6*(N-i), 0] = s_bias - 2**(i-1) + 1
        f3.b1[6*(N-i)+1, 0] = s_bias - 2**(i-1)
        f3.b1[6*(N-i)+2, 0] = -s_bias + 2**(i)
        f3.b1[6*(N-i)+3, 0] = -s_bias + 2**(i) - 1
        f3.b1[6*(N-i)+4, 0] = s_bias - 3*(2**(i-1)) + 1
        f3.b1[6*(N-i)+5, 0] = s_bias - 3*(2**(i-1))

        f3.W2[ref2+N-i, 6*(N-i):6*(N-i+1)] = 2*np.array([1, -1, 1, -1, 1, -1])
    f3.b2[ref2:ref2+size_mem, 0] = -3 * np.ones(N,)  # 2 c_i - 1 (to convert to +1, -1 notation), where c_i contains (\sum (ReLU) -1)

    # zero-out buffer (b_r^{flipped}, b_{-r})
    f3.W1[6*N:7*N, idx_buff:idx_buff+size_mem] = np.eye(size_mem)
    f3.W1[7*N:8*N, idx_buff:idx_buff+size_mem] = -1 * np.eye(size_mem)
    f3.W1[8*N:9*N, idx_buff+size_mem:idx_buff+2*size_mem] = np.eye(size_mem)
    f3.W1[9*N:10*N, idx_buff+size_mem:idx_buff+2*size_mem] = -1 * np.eye(size_mem)
    f3.W2[idx_buff:idx_buff+size_mem, 6*N:7*N] = -1 *  np.eye(size_mem)    
    f3.W2[idx_buff:idx_buff+size_mem, 7*N:8*N] = +1 *  np.eye(size_mem)    
    f3.W2[idx_buff+size_mem:idx_buff+2*size_mem, 8*N:9*N] = -1 *  np.eye(size_mem)    
    f3.W2[idx_buff+size_mem:idx_buff+2*size_mem, 9*N:10*N] = +1 *  np.eye(size_mem)    

    # zero-out existing scratchpad contents for mem[a], mem[b]
    f3.W1[10*N:11*N, ref1:ref1+size_mem] = np.eye(size_mem)
    f3.W1[11*N:12*N, ref1:ref1+size_mem] = -1 * np.eye(size_mem)
    f3.W1[12*N:13*N, ref2:ref2+size_mem] = np.eye(size_mem)
    f3.W1[13*N:14*N, ref2:ref2+size_mem] = -1 * np.eye(size_mem)
    f3.W2[ref1:ref1+size_mem, 10*N:11*N] = -1 *  np.eye(size_mem)    
    f3.W2[ref1:ref1+size_mem, 11*N:12*N] = +1 *  np.eye(size_mem)    
    f3.W2[ref2:ref2+size_mem, 12*N:13*N] = -1 *  np.eye(size_mem)    
    f3.W2[ref2:ref2+size_mem, 13*N:14*N] = +1 *  np.eye(size_mem)    


    X3 = f3.forward(X2)#, 'Subtract Memory')
    #print('X3c: ', X3) # b_{s-r}
    if opt is not None and opt.save_all:
        save_np_to_csv(X3,'R{}_X3c.csv'.format(round_idx),opt)
        save_np_to_csv(X3-X2,'R{}_X3c-X3b.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X3),'R{}_X3c_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X3-X2),'R{}_X3c-X3b_around.csv'.format(round_idx),opt)    
        # X3 finally saved
        save_np_to_csv(np.around(X3),'R{}_X3_around.csv'.format(round_idx),opt)

    # check mem[b]-mem[a] and store it in opt.our_curr_result
    mem_sub = X3[idx_scratchpad+size_mem:idx_scratchpad+2*size_mem, 0]
    #print(mem_sub)
    mem_sub_int = get_signed_int(mem_sub)
    #print(mem_sub_int)
    if opt is not None: 
        opt.our_curr_result = np.append(opt.our_curr_result, [mem_sub_int])
    else:
        our_curr_result = np.append(our_curr_result, [mem_sub_int])
    #print(opt.our_curr_result)

    return X3, our_curr_result, f1, f2, f3


# write "mem[b]-mem[a]" (in the scratchpad) to mem[b]
def write_mem(X, n, logn, num_rows_X, N, round_idx, nrows_list,row_idx_list, opt, our_curr_result=None, lam=10): 
    # load the variables explaining the structure of X
    nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer = nrows_list[:-1] # load num_rows except 1
    idx_memory, idx_scratchpad, idx_pc, idx_pos_enc, idx_buff = row_idx_list
    size_inst = nrows_cmds # size of instruction (commands) = 3*logn
    size_pos_enc = idx_buff - idx_pos_enc # size of positional encoder = logn
    size_mem = idx_scratchpad - idx_memory # size of memory = N    
    idx_scratch_cmd = idx_scratchpad+2*size_mem # 3*logn + N + 2N

    ## write pointer (p_b) -- [idx_scratch_cmd+size_pos_enc : idx_scratch_cmd+2*size_pos_enc, 0]
    ## read location (where mem[b]-mem[a] is located) -- [idx_scratchpad+size_mem:idx_scratchpad+2*size_mem]
    ## write location (where we write mem[b]) -- [idx_memory : idx_memory+size_mem]
    ## buffer location (only the part we use) -- [idx_buff:idx_buff+size_mem]

    ## define TF network f for the read operation 
    f = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=4*size_mem, num_heads=1, opt=opt, lam=lam) 
    
    ## specify the parameter Q,K,V for attention
    f.Q[:, idx_scratch_cmd+size_pos_enc:idx_scratch_cmd+2*size_pos_enc] = np.eye(size_pos_enc) # rows for p_b
    f.Q[:, idx_pos_enc:idx_pos_enc+size_pos_enc] = np.eye(size_pos_enc) # rows for all positional encoding
    f.K = f.Q    
    f.V[idx_buff:idx_buff+size_mem, idx_memory:idx_memory+size_mem] = np.eye(size_mem) # read "current memory" to the buffer, for mem[b]
    f.V[idx_buff:idx_buff+size_mem, idx_scratchpad+size_mem:idx_scratchpad+2*size_mem] = np.eye(size_mem) # read "updated memory" to the buffer, for mem[b] 

    #pdb.set_trace()
    ## specify the parameters W1,W2,b1,b2 for the feed-forward network
    large_const = 100 # a large constant (used for the feed-forward network)
    # ### W1 (third size_mem rows for mem[a])
    # f.W1[2*size_mem:3*size_mem, idx_buff:idx_buff+size_mem] = np.eye(size_mem)        # matrix for buffer for mem[a]
    # ### W1 (fourth size_mem rows for mem[a])
    # f.W1[3*size_mem:4*size_mem, idx_buff:idx_buff+size_mem] = -1 * np.eye(size_mem)        # matrix for buffer for mem[a]
    
    ### W1 (first size_mem rows for mem[b])
    f.W1[:size_mem, idx_memory:idx_memory+size_mem] = -2 * np.eye(size_mem)  # matrix for current memory for mem[b]
    f.W1[:size_mem, idx_buff:idx_buff+size_mem] = 2 * np.eye(size_mem)        # matrix for buffer for mem[b]
    f.W1[:size_mem, -1] = -1 * large_const * np.ones(size_mem,)      # matrix for b = indicator for scratchpad   
    ### W1 (second size_mem rows for mem[b])
    f.W1[size_mem:2*size_mem, idx_memory:idx_memory+size_mem] = 2 * np.eye(size_mem)  # matrix for current memory for mem[b]
    f.W1[size_mem:2*size_mem, idx_buff:idx_buff+size_mem] = -2 * np.eye(size_mem)            # matrix for buffer for mem[b]
    f.W1[size_mem:2*size_mem, -1] = -1 * large_const * np.ones(size_mem,)          # matrix for b    
    ### W1 (third size_mem rows for mem[b])
    f.W1[2*size_mem:3*size_mem, idx_buff:idx_buff+size_mem] = np.eye(size_mem)        # matrix for buffer for mem[b]
    ### W1 (fourth size_mem rows for mem[b])
    f.W1[3*size_mem:, idx_buff:idx_buff+size_mem] = -1 * np.eye(size_mem)        # matrix for buffer for mem[b]
    ### b1 = 0

    ### W2 (for mem[b])
    f.W2[idx_memory:idx_memory+size_mem, :size_mem] = np.eye(size_mem)
    f.W2[idx_memory:idx_memory+size_mem, size_mem:2*size_mem] = -1 * np.eye(size_mem)
    f.W2[idx_buff:idx_buff+size_mem, 2*size_mem:3*size_mem] = -1 * np.eye(size_mem)
    f.W2[idx_buff:idx_buff+size_mem, 3*size_mem:4*size_mem] = np.eye(size_mem)
    ### b2 = 0

    X4 = f.forward(X, heatmap_title='R{}_Write Memory'.format(round_idx))
    if opt is not None and opt.save_all:
        save_np_to_csv(X4,'R{}_X4.csv'.format(round_idx),opt)
        save_np_to_csv((X4-X),'R{}_X4-X3.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X4-X),'R{}_X4-X3_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X4),'R{}_X4_around.csv'.format(round_idx),opt)


    return X4, our_curr_result, f



def conditional_branching(X, n, logn, num_rows_X, N, round_idx, nrows_list,row_idx_list, opt, our_curr_result=None, lam=10):

    # load the variables explaining the structure of X
    nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer = nrows_list[:-1] # load num_rows except 1
    idx_memory, idx_scratchpad, idx_pc, idx_pos_enc, idx_buff = row_idx_list
    size_inst = nrows_cmds # size of instruction (commands) = 3*logn
    size_pos_enc = idx_buff - idx_pos_enc # size of positional encoder = logn
    size_mem = idx_scratchpad - idx_memory # size of memory = N    
    idx_scratch_cmd = idx_scratchpad+2*size_mem # 3*logn + N + 2N

    # 1. store the flag (set flag=1 iff mem[b]-mem[a]<=0)
    f1 = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=2+2*size_mem, num_heads=1, opt=opt, lam=lam) 
    ref1 = idx_scratchpad + size_mem # where mem[b]-mem[a] is stored
    ref2 = idx_scratchpad # where flag will be stored
    assert(size_mem == N)
     
    f1.W1[0, ref1] = 1 # extract b_N
    f1.W1[1, ref1:ref1+size_mem] = -1 * np.ones(size_mem,) # extract -\sum_{i=1}^N b_i
    f1.W1[2:2+size_mem, ref1:ref1+size_mem] = +1 * np.eye(size_mem) # [b_N, ..., b_1]
    f1.W1[2+size_mem:2+2*size_mem, ref1:ref1+size_mem] = -1 * np.eye(size_mem) # -[b_N, ..., b_1]
    f1.b1[1, 0] = -N+1 
    f1.W2[ref2, 0:2] = np.ones(2,) # put flag = ReLU(b_N) + ReLU(-\sum_{i=1}^N b_i - N + 1)
    f1.W2[ref1:ref1+size_mem, 2:2+size_mem] = -1 * np.eye(size_mem)
    f1.W2[ref1:ref1+size_mem, 2+size_mem:2+2*size_mem] = +1 * np.eye(size_mem)
    
    X5a = f1.forward(X)
    #print('X5a: ', X5a)  
    if opt is not None and opt.save_all:
        save_np_to_csv(X5a,'R{}_X5a.csv'.format(round_idx),opt)
        save_np_to_csv(X5a-X,'R{}_X5a-X4.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X5a),'R{}_X5a_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X5a-X),'R{}_X5a-X4_around.csv'.format(round_idx),opt)

    
    # 2. define p_{PC+1} -> save it to p_{PC} #? or in a buffer?
    unit = size_pos_enc # basic unit
    ref = idx_pc # program counter (PC) ## source/dest rows = ref:ref+unit
    ref2 = idx_pc - unit # p_c (pointer when flag=1)
    ref3 = idx_scratchpad # where flag is stored
    f2 = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=8*unit, num_heads=1, opt=opt, lam=lam)

    ## 6*N neurons for computing p_{PC+1}
    for i in np.arange(1, unit+1): # for all components of p_{PC}
        for j in np.arange(1, i+1):
            elem = (2**(j-1))/2
            f2.W1[6*(unit-i), ref+unit-j] = elem 
            f2.W1[6*(unit-i)+1, ref+unit-j] = elem 
            f2.W1[6*(unit-i)+2, ref+unit-j] = -elem 
            f2.W1[6*(unit-i)+3, ref+unit-j] = -elem 
            f2.W1[6*(unit-i)+4, ref+unit-j] = elem 
            f2.W1[6*(unit-i)+5, ref+unit-j] = elem 

        # bias
        s_bias = (2**(i) - 1)/2 + 1 # bias to get the sum s, and +1 is for getting b_{-r}
        f2.b1[6*(unit-i), 0] = (s_bias - 2**(i-1) + 1) 
        f2.b1[6*(unit-i)+1, 0] = (s_bias - 2**(i-1)) 
        f2.b1[6*(unit-i)+2, 0] = (-s_bias + 2**(i)) 
        f2.b1[6*(unit-i)+3, 0] = (-s_bias + 2**(i) - 1) 
        f2.b1[6*(unit-i)+4, 0] = (s_bias - 3*(2**(i-1)) + 1) 
        f2.b1[6*(unit-i)+5, 0] = (s_bias - 3*(2**(i-1))) 

        f2.W2[ref+unit-i, 6*(unit-i):6*(unit-i+1)] = 2*np.array([1, -1, 1, -1, 1, -1])
    f2.b2[ref:ref+unit, 0] = -3 * np.ones(unit,)  # 2 c_i - 1 (to convert to +1, -1 notation), where c_i contains (\sum (ReLU) -1)

    ## 2*N neurons for computing -p_{PC} = - ReLU(p_{PC}) + ReLU(-p_{PC})
    f2.W1[6*unit:7*unit, ref:ref+unit] = +1 * np.eye(unit) # p_{PC}
    f2.W1[7*unit:8*unit, ref:ref+unit] = -1 * np.eye(unit) # -p_{PC}
    f2.W2[ref:ref+unit, 6*unit:7*unit] = -1 * np.eye(unit)
    f2.W2[ref:ref+unit, 7*unit:8*unit] = +1 * np.eye(unit)

    X5b = f2.forward(X5a)

    #print('X5b: ', X5b)  
    if opt is not None and opt.save_all:
        save_np_to_csv(X5b,'R{}_X5b.csv'.format(round_idx),opt)
        save_np_to_csv(X5b-X5a,'R{}_X5b-X5a.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X5b),'R{}_X5b_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X5b-X5a),'R{}_X5b-X5a_around.csv'.format(round_idx),opt)


    # 3. change the program counter pointer (p_{PC}) to p_{next} 
    f3 = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=6*unit+2, num_heads=1, opt=opt, lam=lam)

    ## 4*N neurons for defining p_{next} depending on the flag value
    ### ReLU (p_c - np.ones(unit,) * (1-flag))
    f3.W1[:unit, ref2:ref2+unit] = +1 * np.eye(unit) 
    f3.W1[:unit, ref3] = +1 * np.ones(unit,) 
    f3.b1[:unit, 0] = -1 * np.ones(unit,)
    ### ReLU (-p_c - np.ones(unit,) * (1-flag))
    f3.W1[unit:2*unit, ref2:ref2+unit] = -1 * np.eye(unit) 
    f3.W1[unit:2*unit, ref3] = +1 * np.ones(unit,) 
    f3.b1[unit:2*unit, 0] = -1 * np.ones(unit,)
    ### ReLU (p_{PC+1} - np.ones(unit,) * flag)
    f3.W1[2*unit:3*unit, ref:ref+unit] = +1 * np.eye(unit) 
    f3.W1[2*unit:3*unit, ref3] = -1 * np.ones(unit,) 
    ### ReLU (-p_{PC+1} - np.ones(unit,) * flag)
    f3.W1[3*unit:4*unit, ref:ref+unit] = -1 * np.eye(unit) 
    f3.W1[3*unit:4*unit, ref3] = -1 * np.ones(unit,) 

    ## 2*N neurons for computing -p_{PC+1} = - ReLU(p_{PC+1}) + ReLU(-p_{PC+1})
    f3.W1[4*unit:5*unit, ref:ref+unit] = +1 * np.eye(unit) # p_{PC+1}
    f3.W1[5*unit:6*unit, ref:ref+unit] = -1 * np.eye(unit) # -p_{PC+1}


    ## save p_next in p_{PC+1} location
    f3.W2[ref:ref+unit, :unit] = +1 * np.eye(unit)
    f3.W2[ref:ref+unit, unit:2*unit] = -1 * np.eye(unit)
    f3.W2[ref:ref+unit, 2*unit:3*unit] = +1 * np.eye(unit)
    f3.W2[ref:ref+unit, 3*unit:4*unit] = -1 * np.eye(unit)
    f3.W2[ref:ref+unit, 4*unit:5*unit] = -1 * np.eye(unit)
    f3.W2[ref:ref+unit, 5*unit:6*unit] = +1 * np.eye(unit)

    ## re-initialize the flag row: -ReLU(flag) + ReLU(-flag)
    f3.W1[6*unit, ref3] = +1 # flag
    f3.W1[6*unit+1, ref3] = -1 # -flag
    f3.W2[ref3, 6*unit] = -1
    f3.W2[ref3, 6*unit+1] = +1

    X5c = f3.forward(X5b)# , heatmap_title='R{}_Conditional Branching'.format(round_idx))
    # final output
    X5 = X5c
    #print('X5c: ', X5c)  
    if opt is not None and opt.save_all:
        save_np_to_csv(X5c,'R{}_X5c.csv'.format(round_idx),opt)
        save_np_to_csv(X5c-X5b,'R{}_X5c-X5b.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X5c),'R{}_X5c_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X5c-X5b),'R{}_X5c-X5b_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X5),'R{}_X5_around.csv'.format(round_idx),opt)

    # save_np_to_csv(X5,'R{}_X5.csv'.format(round_idx),opt)
    # save_np_to_csv(X5-X,'R{}_X5-X4.csv'.format(round_idx),opt)
    # save_np_to_csv(np.around(X5-X),'R{}_X5-X4_around.csv'.format(round_idx),opt)

    # check flag & p_{next} and store it in opt.our_curr_result
    flag = X5b[idx_scratchpad, 0]
    p_next = X5[idx_pc:idx_pc+size_pos_enc, 0]
    #print(flag, p_next)
    flag_int = int(np.around(flag))
    p_next_int = get_int(p_next)
    #print(flag_int, p_next_int)
    if opt is not None:
        opt.our_curr_result = np.append(opt.our_curr_result, [flag_int, p_next_int])
        print(opt.our_curr_result)
    else:
        our_curr_result = np.append(our_curr_result, [flag_int, p_next_int])

    return X5, our_curr_result, f1, f2, f3



def error_correction(X,n,logn,num_rows_X,N,round_idx,nrows_list,row_idx_list,opt, our_curr_result=None, lam=10):

    # what parts do we need to correct error? 
    # (1) rows having memory M with size "Nxn", 
    # (2) scratchpad (p_a, p_b, p_c, p_{PC}) with row size "4logn"

    # load the variables explaining the structure of X
    nrows_cmds, nrows_memory, nrows_scratchpad, nrows_pc, nrows_pos_enc, nrows_buffer = nrows_list[:-1] # load num_rows except 1
    idx_memory, idx_scratchpad, idx_pc, idx_pos_enc, idx_buff = row_idx_list
    size_inst = nrows_cmds # size of instruction (commands) = 3*logn
    size_pos_enc = idx_buff - idx_pos_enc # size of positional encoder = logn
    size_mem = idx_scratchpad - idx_memory # size of memory = N    
    idx_scratch_cmd = idx_scratchpad+2*size_mem # 3*logn + N + 2N    


    assert(size_mem == N)
    ref1 = idx_memory # where memory is stored
    ref2 = idx_pc # where program counter is stored
    offset = 6*N
    unit = size_pos_enc 

    ref3 = idx_pc - unit # p_c
    ref4 = idx_pc - 2*unit # p_b
    ref5 = idx_pc - 3*unit # p_a

    # compute epsilon
    eps = np.amax(np.minimum(np.minimum(np.abs(X-1), np.abs(X)), np.abs(X+1)))
    #eps = 0.1
    #print('maximum error eps: ', eps)

    # define network
    f = TF(num_rows_X=num_rows_X, num_cols_X=n, num_rows_Q=logn, num_rows_W=6*(size_mem+4*size_pos_enc), num_heads=1, opt=opt, lam=lam) 

    ## memory part

    f.W1[:N, ref1:ref1+N] = np.eye(N)
    f.W1[N:2*N, ref1:ref1+N] = np.eye(N)
    f.W1[2*N:3*N, ref1:ref1+N] = np.eye(N)
    f.W1[3*N:4*N, ref1:ref1+N] = np.eye(N)
    f.W1[4*N:5*N, ref1:ref1+N] = -1 * np.eye(N)
    f.W1[5*N:6*N, ref1:ref1+N] = np.eye(N)
    f.b1[:N, :] = (1-eps) * np.ones((N, n))
    f.b1[N:2*N, :] = eps * np.ones((N, n))
    f.b1[2*N:3*N, :] = -eps * np.ones((N, n))
    f.b1[3*N:4*N, :] = (-1+eps) * np.ones((N, n))
    f.W2[ref1:ref1+N, :N] = (1/(1-2*eps)) * np.eye(N)
    f.W2[ref1:ref1+N, N:2*N] = -(1/(1-2*eps)) * np.eye(N)
    f.W2[ref1:ref1+N, 2*N:3*N] = (1/(1-2*eps)) * np.eye(N)
    f.W2[ref1:ref1+N, 3*N:4*N] = -(1/(1-2*eps)) * np.eye(N)
    f.W2[ref1:ref1+N, 4*N:5*N] = np.eye(N)
    f.W2[ref1:ref1+N, 5*N:6*N] = -1 * np.eye(N)
    f.b2[ref1:ref1+N, :] = -1 * np.ones((N, n))


    ## positional embedding part (p_{PC}, p_c, p_b, p_a)

    f.W1[offset:offset+unit, ref2:ref2+unit] = np.eye(unit)
    f.W1[offset+unit:offset+2*unit, ref2:ref2+unit] = np.eye(unit)
    f.W1[offset+2*unit:offset+3*unit, ref2:ref2+unit] = np.eye(unit)
    f.W1[offset+3*unit:offset+4*unit, ref2:ref2+unit] = np.eye(unit)
    f.W1[offset+4*unit:offset+5*unit, ref2:ref2+unit] = -1 * np.eye(unit)
    f.W1[offset+5*unit:offset+6*unit, ref2:ref2+unit] = np.eye(unit)

    offset3 = offset+6*unit
    f.W1[offset3:offset3+unit, ref3:ref3+unit] = np.eye(unit)
    f.W1[offset3+unit:offset3+2*unit, ref3:ref3+unit] = np.eye(unit)
    f.W1[offset3+2*unit:offset3+3*unit, ref3:ref3+unit] = np.eye(unit)
    f.W1[offset3+3*unit:offset3+4*unit, ref3:ref3+unit] = np.eye(unit)
    f.W1[offset3+4*unit:offset3+5*unit, ref3:ref3+unit] = -1 * np.eye(unit)
    f.W1[offset3+5*unit:offset3+6*unit, ref3:ref3+unit] = np.eye(unit)

    offset4 = offset3+6*unit
    f.W1[offset4:offset4+unit, ref4:ref4+unit] = np.eye(unit)
    f.W1[offset4+unit:offset4+2*unit, ref4:ref4+unit] = np.eye(unit)
    f.W1[offset4+2*unit:offset4+3*unit, ref4:ref4+unit] = np.eye(unit)
    f.W1[offset4+3*unit:offset4+4*unit, ref4:ref4+unit] = np.eye(unit)
    f.W1[offset4+4*unit:offset4+5*unit, ref4:ref4+unit] = -1 * np.eye(unit)
    f.W1[offset4+5*unit:offset4+6*unit, ref4:ref4+unit] = np.eye(unit)

    offset5 = offset4+6*unit
    f.W1[offset5:offset5+unit, ref5:ref5+unit] = np.eye(unit)
    f.W1[offset5+unit:offset5+2*unit, ref5:ref5+unit] = np.eye(unit)
    f.W1[offset5+2*unit:offset5+3*unit, ref5:ref5+unit] = np.eye(unit)
    f.W1[offset5+3*unit:offset5+4*unit, ref5:ref5+unit] = np.eye(unit)
    f.W1[offset5+4*unit:offset5+5*unit, ref5:ref5+unit] = -1 * np.eye(unit)
    f.W1[offset5+5*unit:offset5+6*unit, ref5:ref5+unit] = np.eye(unit)

    f.b1[offset:offset+unit, :] = (1-eps) * np.ones((unit, n))
    f.b1[offset+unit:offset+2*unit, :] = eps * np.ones((unit, n))
    f.b1[offset+2*unit:offset+3*unit, :] = -eps * np.ones((unit, n))
    f.b1[offset+3*unit:offset+4*unit, :] = (-1+eps) * np.ones((unit, n))

    f.b1[offset3:offset3+unit, :] = (1-eps) * np.ones((unit, n))
    f.b1[offset3+unit:offset3+2*unit, :] = eps * np.ones((unit, n))
    f.b1[offset3+2*unit:offset3+3*unit, :] = -eps * np.ones((unit, n))
    f.b1[offset3+3*unit:offset3+4*unit, :] = (-1+eps) * np.ones((unit, n))

    f.b1[offset4:offset4+unit, :] = (1-eps) * np.ones((unit, n))
    f.b1[offset4+unit:offset4+2*unit, :] = eps * np.ones((unit, n))
    f.b1[offset4+2*unit:offset4+3*unit, :] = -eps * np.ones((unit, n))
    f.b1[offset4+3*unit:offset4+4*unit, :] = (-1+eps) * np.ones((unit, n))

    f.b1[offset5:offset5+unit, :] = (1-eps) * np.ones((unit, n))
    f.b1[offset5+unit:offset5+2*unit, :] = eps * np.ones((unit, n))
    f.b1[offset5+2*unit:offset5+3*unit, :] = -eps * np.ones((unit, n))
    f.b1[offset5+3*unit:offset5+4*unit, :] = (-1+eps) * np.ones((unit, n))


    f.W2[ref2:ref2+unit, offset:offset+unit] = (1/(1-2*eps)) * np.eye(unit)
    f.W2[ref2:ref2+unit, offset+unit:offset+2*unit] = -(1/(1-2*eps)) * np.eye(unit)
    f.W2[ref2:ref2+unit, offset+2*unit:offset+3*unit] = (1/(1-2*eps)) * np.eye(unit)
    f.W2[ref2:ref2+unit, offset+3*unit:offset+4*unit] = -(1/(1-2*eps)) * np.eye(unit)
    f.W2[ref2:ref2+unit, offset+4*unit:offset+5*unit] = np.eye(unit)
    f.W2[ref2:ref2+unit, offset+5*unit:offset+6*unit] = -np.eye(unit)

    f.W2[ref3:ref3+unit, offset3:offset3+unit] = (1/(1-2*eps)) * np.eye(unit)
    f.W2[ref3:ref3+unit, offset3+unit:offset3+2*unit] = -(1/(1-2*eps)) * np.eye(unit)
    f.W2[ref3:ref3+unit, offset3+2*unit:offset3+3*unit] = (1/(1-2*eps)) * np.eye(unit)
    f.W2[ref3:ref3+unit, offset3+3*unit:offset3+4*unit] = -(1/(1-2*eps)) * np.eye(unit)
    f.W2[ref3:ref3+unit, offset3+4*unit:offset3+5*unit] = np.eye(unit)
    f.W2[ref3:ref3+unit, offset3+5*unit:offset3+6*unit] = -np.eye(unit)

    f.W2[ref4:ref4+unit, offset4:offset4+unit] = (1/(1-2*eps)) * np.eye(unit)
    f.W2[ref4:ref4+unit, offset4+unit:offset4+2*unit] = -(1/(1-2*eps)) * np.eye(unit)
    f.W2[ref4:ref4+unit, offset4+2*unit:offset4+3*unit] = (1/(1-2*eps)) * np.eye(unit)
    f.W2[ref4:ref4+unit, offset4+3*unit:offset4+4*unit] = -(1/(1-2*eps)) * np.eye(unit)
    f.W2[ref4:ref4+unit, offset4+4*unit:offset4+5*unit] = np.eye(unit)
    f.W2[ref4:ref4+unit, offset4+5*unit:offset4+6*unit] = -np.eye(unit)

    f.W2[ref5:ref5+unit, offset5:offset5+unit] = (1/(1-2*eps)) * np.eye(unit)
    f.W2[ref5:ref5+unit, offset5+unit:offset5+2*unit] = -(1/(1-2*eps)) * np.eye(unit)
    f.W2[ref5:ref5+unit, offset5+2*unit:offset5+3*unit] = (1/(1-2*eps)) * np.eye(unit)
    f.W2[ref5:ref5+unit, offset5+3*unit:offset5+4*unit] = -(1/(1-2*eps)) * np.eye(unit)
    f.W2[ref5:ref5+unit, offset5+4*unit:offset5+5*unit] = np.eye(unit)
    f.W2[ref5:ref5+unit, offset5+5*unit:offset5+6*unit] = -np.eye(unit)

    f.b2[ref2:ref2+unit, :] = -1 * np.ones((unit, n))
    f.b2[ref3:ref3+unit, :] = -1 * np.ones((unit, n))
    f.b2[ref4:ref4+unit, :] = -1 * np.ones((unit, n))
    f.b2[ref5:ref5+unit, :] = -1 * np.ones((unit, n))

    X6 = f.forward(X)#, 'Error Correction')
    if opt is not None and opt.save_all:
        save_np_to_csv(X6-X,'R{}_X6-X5.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X6),'R{}_X6_around.csv'.format(round_idx),opt)
        save_np_to_csv(np.around(X6-X),'R{}_X6-X5_around.csv'.format(round_idx),opt)
        save_np_to_csv(X6,'R{}_X6.csv'.format(round_idx),opt)

    if opt is not None and round_idx == opt.num_loops - 1:    
        save_np_to_csv(X6,'R{}_X6.csv'.format(round_idx),opt)

    #assert((np.around(X) == np.around(X6)).all())
    #print('locations where error correction is not proper: ', np.where (np.around(X) != X6))

    return X6, our_curr_result, f
 

class TF: # Transformer architecture

    def __init__(self, num_rows_X, num_cols_X, num_rows_Q, num_rows_W, num_heads, opt, lam=10): # initialize the parameters in TF Encoder layer        
        self.num_heads = num_heads
        if self.num_heads == 1:
            self.Q = np.zeros((num_rows_Q,num_rows_X))  # query
            self.K = np.zeros((num_rows_Q,num_rows_X))  # key
            self.V = np.zeros((num_rows_X,num_rows_X))  # value
        elif self.num_heads == 2:
            self.Q1 = np.zeros((num_rows_Q,num_rows_X))  # query
            self.K1 = np.zeros((num_rows_Q,num_rows_X))  # key
            self.V1 = np.zeros((num_rows_X,num_rows_X))  # value
            self.Q2 = np.zeros((num_rows_Q,num_rows_X))  # query
            self.K2 = np.zeros((num_rows_Q,num_rows_X))  # key
            self.V2 = np.zeros((num_rows_X,num_rows_X))  # value
        else:
            raise NotImplementedError
        self.W1 = np.zeros((num_rows_W,num_rows_X)) # FF1 weight
        self.b1 = np.zeros((num_rows_W,num_cols_X)) # FF1 bias
        self.W2 = np.zeros((num_rows_X,num_rows_W)) # FF2 weight
        self.b2 = np.zeros((num_rows_X,num_cols_X)) # FF2 bias

        self.opt = opt
        if self.opt is not None:
            self.lam = self.opt.lam
        else:
            self.lam = lam

        return

    def forward(self, X, heatmap_title=None): # return the output of TF Encoder layer (in eq.1b)
        # X: input matrix with size (num_rows_X, num_cols_X)
        
        np.set_printoptions(precision=1, suppress=True)
        if self.num_heads == 1:
            self.softmax_output = numpy_softmax(X.T @ self.K.T @ self.Q @ X, lam=self.lam, dim=1) 
            self.attn = X + self.V @ X @ self.softmax_output
        elif self.num_heads == 2:
            self.softmax_output1 = numpy_softmax(X.T @ self.K1.T @ self.Q1 @ X, lam=self.lam, dim=1) 
            self.softmax_output2 = numpy_softmax(X.T @ self.K2.T @ self.Q2 @ X, lam=self.lam, dim=1) 
            self.attn = X + self.V1 @ X @ self.softmax_output1 + self.V2 @ X @ self.softmax_output2
        else:
            raise NotImplementedError
        if heatmap_title is not None and self.opt is not None:
            self.save_heatmap(heatmap_title)
        

        ff1_output = numpy_relu(self.W1 @ self.attn + self.b1)
        output = self.attn + self.W2 @ ff1_output + self.b2

        return output

    def save_heatmap(self, heatmap_title):
        if self.b2.shape[-1] == 32:
            if self.num_heads == 1:
                fig, ax = plt.subplots()
                im = ax.imshow(self.softmax_output)
                labels_name = ['scratchpad[0]', '...', 'scratchpad[7]', 
                'memory[0]', '...', 'memory[7]', 'command[0]', '...', 'command[15]']
                # Show all ticks and label them with the respective list entries
                ax.set_xticks(np.array([0, 3.5, 6, 8, 11.5, 14, 16, 23.5, 31]))
                ax.set_xticklabels(labels_name)
                ax.set_yticks(np.array([0, 3.5, 6, 8, 11.5, 14, 16, 23.5, 31]))
                ax.set_yticklabels(labels_name)
                
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=75, ha="right",
                        rotation_mode="anchor")

                plt.title(heatmap_title)
                fig.colorbar(im, ax=ax)
                if self.opt is not None:
                    list_path = [self.opt.output_folder, self.opt.mode, heatmap_title]
                    path = os.path.join(*list_path)
                    fig.savefig(path)
                #plt.subplots_adjust(bottom=0.3, top=0.93, left=-0.0, right=0.75)                                
                fig.tight_layout(rect=[0, 0, 0.85, 1.0])
                plt.close()

                return fig

            elif self.num_heads == 2:

                fig, ax = plt.subplots()
                im = ax.imshow(self.softmax_output1)
                labels_name = ['scratchpad[0]', '...', 'scratchpad[7]', 
                'memory[0]', '...', 'memory[7]', 'command[0]', '...', 'command[15]']
                # Show all ticks and label them with the respective list entries
                ax.set_xticks(np.array([0, 3.5, 6, 8, 11.5, 14, 16, 23.5, 31]))
                ax.set_xticklabels(labels_name)
                ax.set_yticks(np.array([0, 3.5, 6, 8, 11.5, 14, 16, 23.5, 31]))
                ax.set_yticklabels(labels_name)
                
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=75, ha="right",
                        rotation_mode="anchor")

                plt.title(heatmap_title + ' (head 1)')
                fig.colorbar(im, ax=ax)
                if self.opt is not None:                
                    list_path = [self.opt.output_folder, self.opt.mode, heatmap_title+'_head_1']
                    path = os.path.join(*list_path)
                    fig.savefig(path)
                #plt.subplots_adjust(bottom=0.3, top=0.93, left=-0.0, right=0.75)                
                fig.tight_layout(rect=[0, 0, 0.85, 1.0])
                plt.close()

                fig2, ax2 = plt.subplots()
                im = ax2.imshow(self.softmax_output2)
                labels_name = ['scratchpad[0]', '...', 'scratchpad[7]', 
                'memory[0]', '...', 'memory[7]', 'command[0]', '...', 'command[15]']
                # Show all ticks and label them with the respective list entries
                ax2.set_xticks(np.array([0, 3.5, 6, 8, 11.5, 14, 16, 23.5, 31]))
                ax2.set_xticklabels(labels_name)
                ax2.set_yticks(np.array([0, 3.5, 6, 8, 11.5, 14, 16, 23.5, 31]))
                ax2.set_yticklabels(labels_name)
                
                # Rotate the tick labels and set their alignment.
                plt.setp(ax2.get_xticklabels(), rotation=75, ha="right",
                        rotation_mode="anchor")

                plt.title(heatmap_title + ' (head 2)')
                fig2.colorbar(im, ax=ax2)
                if self.opt is not None:                
                    list_path = [self.opt.output_folder, self.opt.mode, heatmap_title+'_head_2']
                    path = os.path.join(*list_path)
                    fig2.savefig(path)
                #plt.subplots_adjust(bottom=0.3, top=0.93, left=-0.0, right=0.75)                
                # fig2.tight_layout()
                fig2.tight_layout(rect=[0, 0, 0.85, 1.0])
                plt.close()


                return fig, fig2