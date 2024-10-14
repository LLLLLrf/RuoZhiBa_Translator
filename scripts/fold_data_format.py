import os
import json

fold_path = "data/"
train_path = fold_path+"train/"
val_path = fold_path+"val/"
test_path = fold_path+"test/"

data = []
for i in range(1, 11):
    file_path = fold_path+"val_fold_"+str(i)+".json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data.append(json.load(f))

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

for i in range(10):
    train = []
    for j in range(8):
        train += data[(i+j) % 10]
    with open(train_path+"fold_"+str(i)+".json", 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False)
    val = data[(i+8) % 10]
    test = data[(i+9) % 10]
    with open(val_path+"fold_"+str(i)+".json", 'w', encoding='utf-8') as f:
        json.dump(val, f, ensure_ascii=False)
    with open(test_path+"fold_"+str(i)+".json", 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False)
