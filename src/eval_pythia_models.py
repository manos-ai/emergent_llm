# -*- coding: utf-8 -*-
"""
test scaling of pythia models on arithmetic tasks
"""

#%% imports

import tqdm
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# torch
import torch

# HF
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM


#%% functions

def check_answer_batch(num1, num2, op, responses):
    '''
    check an LLMs answer on numeric tasks
    Input:
        num1, num2: int
            the two input numbers
        op: str
            the operation to perform: either +, -, * or /
        responses: list of str
            the LLMs responses; may contain additional text
    Output: 
        num_correct: int
            number of correct LLM answers
    '''

    # calculate the desired response
    if op == '+':
        res = num1 + num2
    elif op == '-':
        res = num1 - num2
    elif op == '*':
        res = num1 * num2
    elif op == '/':
        res = num1 // num2
    else:
        return None # op undefined
    
    # formulate correct answer
    corr_ans = str(num1) + ' ' + op + ' ' + str(num2) + ' = ' + str(res)

    # check if correct answer is inside the LLms responses
    num_correct = 0
    for response in responses:
        if corr_ans in response:
            num_correct += 1
    
    return num_correct


def eval_model(A, B, model, tokenizer, op = '+', n_succ = 1, n_max = 1e5, batch_size = 100):
    p_sum = 0
    m = len(A)

    for i in tqdm.tqdm( range(m) ):
        num1 = A[i]
        num2 = B[i]
        #op = '+'

        pi = eval_model_until(num1, num2, op, model, tokenizer, n_succ, n_max, batch_size)
        p_sum += pi
    # end for

    acc = p_sum / m
    return acc


def eval_model_until(num1, num2, op, model, tokenizer, n_succ = 1, n_max = 1e4, batch_size = 1e2):
    '''
    evaluates a model on the arithmetic task: 'How much is num1 op num2? Answer: '
    num1, num2 are two integers, op is the arithemtic operation (+, -, *, /)
    we use the function check_answer to check the model's response
    we estimate the probability that the model answers correctly by running multiple times (with temperature) until we achieve n_succ
    successes, or we reach the max number of iterations, n_max
    '''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    n_max = int(n_max)
    batch_size = int(batch_size)

    # build the prompt
    input_text_batch = ['How much is {} {} {}? Answer: '.format(num1, op, num2) for _ in range(int(batch_size))]
    # Tokenize the input text
    tokenizer.padding_side = 'left' 
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    encoding = tokenizer(input_text_batch, padding = True, return_tensors = 'pt').to(device)
    #input_ids = tokenizer.encode(input_text_batch, return_tensors = 'pt')

    # query model multiple times
    succ_cnt = 0
    n_iters = n_max
    #for i in tqdm.tqdm( range(1, int(n_max) + 1) ):
    for i in range(0, n_max // batch_size):
        #output = model.generate(input_ids, do_sample = True)
        # Decode the response
        #ans = tokenizer.decode(output[:, 0], skip_special_tokens = True)
        #print(ans)
        
        with torch.no_grad():
            # supress warning: Setting `pad_token_id` to `eos_token_id`:0 for open-end generation
            # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
            generated_ids = model.generate(**encoding, do_sample = True, pad_token_id = tokenizer.eos_token_id)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)

        # check ans correctness - if True, success counter increases
        ans = generated_texts
        succ_cnt += check_answer_batch(num1, num2, op, ans)

        # if we reached the req number of successes
        if succ_cnt >= n_succ:
            n_iters = (i + 1) * batch_size
    
    # calc probability
    prob = succ_cnt / n_iters
    return prob


def load_pythia_model(size, cache_dir = '../HF'):
    '''
    load the corresponding pythia model and tokenizer, based on the size
    size: str
        values: 14m, 70m, 160m, 410m, 1.4b, 2.8b, 6.9b, 12b
    '''
    
    # get the model's name
    model_name = 'EleutherAI/pythia-{}'.format(size)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir)
    
    return tokenizer, model


#%% main

# create m random arithmetic tasks
m = 1
op = '+'
A = np.random.randint(0, 100, m)
B = np.random.randint(0, 100, m)

# eval params
n_succ = 1
n_max = 1e4

# pythia number of non-embedding params for each model size
non_emb_params = {'70m': 18.91e6, '160m': 85.06e6, '410m': 302.31e6, '1.4b': 1208.602e6,
                  '2.8b': 2517.652e6, '6.9b': 6444.163e6, '12b': 11327.027e6}

# list of model sizes to evaluate on
model_sizes = ['160m', '410m', '1.4b', '2.8b', '6.9b']
batch_sizes = [1e3,    1e3,    1e2,    1e2,    1e1]

# testing
#model_sizes = ['1.4b']
#batch_sizes = [1e2]


# run
trainable_params = [non_emb_params[x] for x in model_sizes]
accs = []
for i in tqdm.tqdm( range(len(model_sizes)) ):
    # load model and tokenizer
    size = model_sizes[i]
    batch_size = batch_sizes[i]
    tokenizer_i, model_i = load_pythia_model(size, cache_dir = '../HF')

    #acc = eval_model(A, B, model_i, tokenizer_i, n_succ, n_max)
    acc = eval_model(A, B, model_i, tokenizer_i, op, n_succ, n_max, batch_size)
    accs.append(acc)
    
    # clear mem
    del model_i, tokenizer_i
print(accs)


# save results in csv
# create a dictionary from the lists
data = {'model_sizes': model_sizes, 'trainable_params': trainable_params, 'accs': accs}
# convert the dictionary to a DataFrame
df = pd.DataFrame(data)
# save the DataFrame to a CSV file
df.to_csv('../results/acc_vs_N_results.csv', index = False)


        


