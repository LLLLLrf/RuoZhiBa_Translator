import os
import json

folder_path = "data/"
text_path = "data/sentence.txt"

text = []

for file_path in os.listdir(folder_path):
    if file_path[-4:] != "json":
        continue
    with open(folder_path+file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            for key in sample.keys():
                if key == "data_id":
                    continue
                text.append(sample[key])
        f.close()

cnt = 0
limit = 100

for fpathe, dirs, fs in os.walk('data/wiki_zh'):
    for f in fs:
        fp = os.path.join(fpathe, f)
        print(fp, end='\r')
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                for sentence in data["text"].split('\n'):
                    if sentence != "" and len(sentence) >= 20:
                        # text.append(sentence)
                        cnt += 1
                        if cnt == limit:
                            cnt = 0
                            text.append(sentence)
            f.close()

with open(text_path, 'w', encoding='utf-8') as f:
    for line in text:
        f.write(line+'\n')
    f.close()
