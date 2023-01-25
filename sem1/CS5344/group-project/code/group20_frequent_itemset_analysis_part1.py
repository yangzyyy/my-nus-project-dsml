import pandas as pd
from random import sample
def partitions(entities, res):
    if entities:
        res.add(tuple(entities))
        for i in range(1,len(entities)+1):
            temp = entities[:i-1]+entities[i:]
            partitions(temp, res)
res = set()
overall_occurrences = {}

#divide file into chunks
chunk_size = 10000
source_list = [i for i in range(chunk_size)]
sample_portion = 0.3
s_sup = 0.0002
overall_size = 0
w = 0
for chunk in pd.read_csv('processed.csv', chunksize=chunk_size):
    w+=1
    print(w)
    raw_entities = chunk['Annotated_Entities'].to_list()
    source_list = [i for i in range(len(raw_entities))]
    sample_size = int(sample_portion*len(raw_entities))
    sampled_indices = sample(source_list, sample_size)
    sampled_raw_entities = [raw_entities[i] for i in sampled_indices]
    overall_size += sample_size
    entities = set()
    occurrences = {}
    for e in sampled_raw_entities:
        tmp = e[1:-1]
        curr_line_entities = []
        tmp = tmp.split(', ')
        tmp = list(set(tmp))
        if len(tmp) > 10:
            continue
        for i in tmp:
            curr_line_entities.append(i[1:-1])
        
        sorted_entities = sorted(curr_line_entities)
        set_entities = set()
        partitions(sorted_entities, set_entities)
        for i in set_entities:
            if i in occurrences:
                occurrences[i] += 1
            else:
                occurrences[i] = 1
        
    for e, count in occurrences.items():
        if count > s_sup*sample_size and e not in res:
            res.add(e)
        if e not in overall_occurrences:
            overall_occurrences[e] = occurrences[e]
        else:
            overall_occurrences[e] += occurrences[e]

    

##overall:
frequent_items = []
for x in res:
    if overall_occurrences[x] < s_sup*overall_size:
        continue
    frequent_items.append((x, overall_occurrences[x]))

frequent_items.sort(key=lambda x: x[1], reverse=True)
top_itemsets = frequent_items[:20]

with open('frequent_items.txt', 'w') as f:
    for itemset in top_itemsets:
        f.write(f"{itemset[0]}: support = {itemset[1]}\n")
