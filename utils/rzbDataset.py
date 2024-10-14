import json
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class rzbDataset(Dataset):

    """
    Initialize Dataset.

    Parameters:
        folder_path (str): Data folder path, like "./data/"
        k (int): K-Fold Cross-Validation parameter, means "k-th fold as test"
        mode (str): Dataset type, like "train", "val", "test"

    Returns:
        Object: Dataset
    """

    def __init__(self, folder_path, k, mode=["train", "val", "test"]):
        super().__init__()
        self.k = k
        self.mode = mode
        self.folder_path = folder_path + \
            "/" if folder_path[-1] != "/" else folder_path
        self.data = self.__load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dic = self.data.loc[index].to_dict()
        return dic


    def __load_data(self):
        data = pd.DataFrame(columns=["original", "annotated"])
        if self.mode == "test":
            train_file_path = self.folder_path + \
                "train_fold_" + str(self.k) + ".json"
            val_file_path = self.folder_path + \
                "val_fold_" + str(self.k) + ".json"
            data = pd.concat([self.__pd_from_json(train_file_path), self.__pd_from_json(
                val_file_path)], ignore_index=True)
        else:
            for i in range(1, 11):
                if i == self.k:
                    continue
                file_path = self.folder_path + \
                    self.mode+"_fold_" + str(i) + ".json"
                data = pd.concat(
                    [data, self.__pd_from_json(file_path)], ignore_index=True)

        return data

    def __pd_from_json(self, path):
        # print(path)
        data_dict = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for sample in data:
                for result in sample.keys():
                    if result[0] != "a":
                        continue
                    col = [sample["original_data"],
                           sample[result]]
                    data_dict.append(col)
            f.close()
        return pd.DataFrame(data_dict, columns=["original", "annotated"])
