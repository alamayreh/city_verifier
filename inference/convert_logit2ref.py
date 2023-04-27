import numpy as np
import csv
from scipy.special import softmax
import torch


target_cvsw = 'closedset_valid10k_score2.csv'
with open(target_cvsw, 'w') as cvsFilew:
    writer = csv.writer(cvsFilew)
    writer.writerow(['Amsterdam', 'Barcelona', 'Berlin', 'London', 'LosAngeles', 'Milan', 'NewYork', 'Paris', 'Rome', 'Tokyo', 'GT', 'pred'])
        
    target_cvs = 'closedset_valid10k_score.csv'
    
    csvFile = open(target_cvs, "r")
    reader = csv.reader(csvFile)
    
    label = 0
    count = 0
    for row in reader:
        if reader.line_num == 1:
            continue
            
        count += 1
        rows = [float(item) for item in row]
        #print(row)
        
        rows = torch.Tensor(rows)
        labels = torch.Tensor([label])
        
        pred_idx = rows.argmax(dim=0).item()
        #print(pred_idx)
        
        if pred_idx == label:
            row += [label, 1]
        else:
            row += [label, 0]
        
        #print(row)
        if count == 1000:
            label += 1
            count = 0
            
        writer.writerow(row)

        