import os
import re
import csv
import pandas as pd

def sentence_to_dict(sentence, prefix):
    result = {}
    for i, line in enumerate(sentence.split("\n")[:-1]):
        elements = line.split(", ")
        if i !=2: continue
        key = elements[2]
        value = float(elements[3])
        result[f"{prefix}_{key}"] = value
    return result

def sentence_to_dict2(sentence, prefix):
    headers = sentence.split("\n")[0].split(" ")[:19]
    data = sentence.split("\n")[1:]
    data_dict = {}
    for row in data[:-1]:
        row_data = re.split(r'\s+', row)
        for i, header in enumerate(headers[3:]):
            if i == 3:
                data_dict[f"{prefix}_{row_data[0]}_{header}"] = float(row_data[i+1].strip())
            if i == 5:
                data_dict[f"{prefix}_{row_data[0]}_{header}"] = float(row_data[i+1].strip()) * 20
        
    return data_dict

def file_to_dict(path, prefix):
    with open(path, "r") as file:
        contents = file.read()
    
    result = {}
    sentence = contents[contents.find("# Measure Cortex, NumVert,"):contents.find("# NTableCols 10")]
    result.update(sentence_to_dict(sentence, prefix))
    sentence2 = contents[contents.find("# ColHeaders"):]
    result.update(sentence_to_dict2(sentence2, prefix))
    
    return result

if __name__ == "__main__":
    data_path = "/NFS/Users/kimyw/data/sample"

    flag = True
    with open(os.path.join(data_path, "feats_local.csv"), "w") as f:
        w = csv.writer(f)
        for subject in os.listdir(data_path):
            lstat_path = os.path.join(data_path, subject, "stats/lh.aparc.DKTatlas.mapped.stats")
            rstat_path = os.path.join(data_path, subject, "stats/rh.aparc.DKTatlas.mapped.stats")
            if not os.path.exists(lstat_path): continue
            result = {"subject": subject}
            result.update(file_to_dict(lstat_path, "l"))
            result.update(file_to_dict(rstat_path, "r"))
            if flag:
                w.writerow(result.keys())
                flag = False
            w.writerow(result.values())

    df = pd.read_csv(os.path.join(data_path, "feats_local.csv"), index_col="subject")
    normalization_df = (df - df.mean()) / df.std()
    normalization_df.to_csv(os.path.join(data_path, "nfeats_local.csv"))