from collections import defaultdict
import random
import pickle
from preprocess import generic_labels, task_splitter
from model import NERModel
import random
from mpi4py import MPI
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import time
start_time = time.time()
comm= MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
device = f"cuda:{rank}"
random.seed(42)
test_df = pickle.load(open("data/test.pkl", "rb"))
train_df = pickle.load(open("data/train.pkl", "rb"))
val_df = pickle.load(open("data/val.pkl", "rb"))

train_labels = generic_labels(list(train_df["Label"].to_numpy()))
test_labels = generic_labels(list(test_df["Label"].to_numpy()))
val_labels = generic_labels(list(val_df["Label"].to_numpy()))

train_words = list(train_df["Word"].to_numpy())
test_words = list(test_df["Word"].to_numpy())
val_words = list(val_df["Word"].to_numpy())

label_to_int = {"O": 0}
index = 1
for labels in train_labels:
    for label in labels:
        if label not in label_to_int:
            label_to_int[label] = index
            index += 1

k = 25
#num_tasks=5

# need each task to be split into a support and query set
task_words, task_labels = task_splitter(train_words, train_labels, num_tasks=4)# default num_tasks is 5, might have to reduce if not enough GPUs
tasks_sq = [] 
for t in range(len(task_words)):
    support = []
    query = []
    entity_counter=defaultdict(lambda: 0)
    for s in range(len(task_words[t])):
        non_o = [x for x in task_labels[t][s] if x!="O"] # everything that's not o
        if len(non_o)==1: #if there's exactly one entity, we're cooking
            #this is also a limitation, if we have multiple entities in one
         # sentence, we don't get to use that sentence. this could be avoidable
         #future work.
            if entity_counter[non_o[0]]<k: # if we don't have enough examples
                entity_counter[non_o[0]]+=1
                new_task_labels_t_s = [label_to_int[x] for x in task_labels[t][s]]#text to int label
                support.append((task_words[t][s], new_task_labels_t_s))
            else:
                new_task_labels_t_s = [label_to_int[x] for x in task_labels[t][s]]#same here
                query.append((task_words[t][s], new_task_labels_t_s[:]))
    tasks_sq.append((support, query))

#now we're cooking
epochs = 1
batch_size=25
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",use_fast=True)
support, query=tasks_sq[rank]
support_words = [x[0] for x in support]
support_labels = [x[1] for x in support]
query_words = [x[0] for x in query]
query_labels = [x[1] for x in query]



result = tokenizer(support_words, is_split_into_words=True, padding=True, truncation=True, max_length=768)#i just picked a random number for length lol
model = NERModel()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=.01)
input_ids = result.input_ids
attention_mask = result.attention_mask
new_labels = []
assert(len(result.encodings)==len(input_ids))
for i in range(len(input_ids)):
    word_ids = result.encodings[i].word_ids
    newnew_labels = []
    for label_idx in word_ids:
        if label_idx is not None:
            newnew_labels.append(support_labels[i][label_idx])
        else:
            newnew_labels.append(-100)
    new_labels.append(newnew_labels)
support_labels = new_labels

support_dataset = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(support_labels))
support_dataloader = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)

result = tokenizer(query_words, is_split_into_words=True,padding=True, max_length=72, truncation=True)
input_ids = result.input_ids
attention_mask = result.attention_mask
new_labels = []

assert(len(result.encodings)==len(input_ids))
for i in range(len(input_ids)):
    word_ids = result.encodings[i].word_ids
    newnew_labels = []
    for label_idx in word_ids:
        if label_idx is not None:
            newnew_labels.append(query_labels[i][label_idx])
        else:
            newnew_labels.append(-100)
    new_labels.append(newnew_labels)
query_labels = new_labels
query_dataset = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(query_labels))
query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)
loss_func = CrossEntropyLoss(ignore_index=-100)

for i in tqdm(range(epochs)):
    for j in tqdm(range(2)):
        for batch_input_ids, batch_attention_masks, batch_labels in support_dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_masks = batch_attention_masks.to(device)
            batch_labels=batch_labels.view(-1)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids=batch_input_ids, attention_mask = batch_attention_masks)
            loss = loss_func(logits.view(-1, logits.size(-1)), batch_labels)
            loss.backward()
            optimizer.step()
    params = list(model.parameters())
    params_np = [p.detach().cpu().numpy() for p in params]
    g_params = comm.gather(params_np, root=0)
    average_params = None
    if rank==0:
        total_params = [np.zeros_like(p.detach().cpu()) for p in params]
        for p in g_params:
            for idx, param in enumerate(p):
                total_params[idx] += param
        average_params = [p/size for p in total_params]
        model = NERModel()
        new_params = [torch.tensor(p, dtype=torch.float32) for p in average_params]
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param
        model.to(device)
        for batch_input_ids, batch_attention_masks, batch_labels in query_dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_masks = batch_attention_masks.to(device)
            batch_labels=batch_labels.view(-1)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids=batch_input_ids, attention_mask = batch_attention_masks)
            loss = loss_func(logits.view(-1, logits.size(-1)), batch_labels)
            loss.backward()
            optimizer.step()
        params = list(model.parameters())
        params_np = [p.detach().cpu().numpy() for p in params]
    average_params = comm.bcast(average_params, root=0)
    model = NERModel()
    new_params = [torch.tensor(p, dtype=torch.float32) for p in average_params]
    for param, new_param in zip(model.parameters(), new_params):
        param.data = new_param
    model.to(device)

end_time = time.time()

runtime = end_time-start_time
print(f"{runtime} seconds")

##Not worried about inference, accuracy is going to suck anyway and it's now 3 am, we just care about SPEEEEEEEED and "can we do it?"



