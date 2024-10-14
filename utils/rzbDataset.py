import json
import pandas as pd
from torch.utils.data import Dataset
import tqdm


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

    def __init__(self, folder_path, k=-1, mode=["train", "val", "test"], method=lambda x: x):
        super().__init__()
        assert mode == "test" or k >= 0
        self.k = k
        self.mode = mode
        self.method = method
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
        file_path = self.folder_path + \
            self.mode+"/fold_" + str(self.k) + ".json"
        if self.mode == "test":
            file_path = self.folder_path + \
                self.mode+"/test.json"
        print("[INFO] Load '"+file_path+"' dataset...")
        data = pd.concat(
            [data, self.__pd_from_json(file_path)], ignore_index=True)

        return data

    def __pd_from_json(self, path):
        data_dict = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for idx in tqdm.tqdm(range(len(data))):
                sample = data[idx]
                for result in sample.keys():
                    if result[0] != "a":
                        continue
                    col = [sample["original_data"],
                           sample[result]]
                    data_dict.append(self.__pre_process(col))
            f.close()
        return pd.DataFrame(data_dict, columns=["original", "annotated"])

    def __pre_process(self, col: list):
        for i in range(len(col)):
            col[i] = self.method(col[i])
        return col
