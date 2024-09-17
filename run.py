from datasets import load_dataset
from torch.utils.data import DataLoader

from model.transformer import transformer



if __name__ == "__main__":
    dataset = load_dataset("wmt/wmt14", "de-en")
    trainset, validateset, testset = dataset["train"], dataset["validation"], dataset["test"]
    trainLoader = DataLoader(trainset, batch_size=32, shuffle=True)
    validateLoader = DataLoader(validateset, batch_size=32, shuffle=True)
    testLoader = DataLoader(testset, batch_size=32, shuffle=True)
    model = transformer()

    for data in trainLoader:
        model.train()
        x, y = data["translation"]["de"], data["translation"]["en"]
        out = model(x, y)

