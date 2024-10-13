from utils import rzbDataset
from torch.utils.data import DataLoader


def main():
    # k-fold dataset
    for k in range(1, 11):
        train_dataset = rzbDataset("data", k, mode="train")
        val_dataset = rzbDataset("data", k, mode="val")
        test_dataset = rzbDataset("data", k, mode="test")

        train_dataloader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(
            val_dataset, batch_size=16, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(
            test_dataset, batch_size=16, shuffle=True, num_workers=4)


if __name__ == "__main__":
    main()
