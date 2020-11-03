import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm

from colorhandpose3d.model.HandNet import HandNet
from torchvision.models import vgg11


class HandDataset(Dataset):
    def __init__(self, root_dir="./data/handnet/"):
        data = pd.read_csv(root_dir + "data_df.csv", sep=",")
        self.root_dir = root_dir
        self.x = list(data.loc[:, "name"])
        self.y = list(data.loc[:, "target"])

        self.preprocess = transforms.Compose(
            [
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.641, 0.619, 0.612], std=[0.254, 0.262, 0.258]),
            ]
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.open(self.root_dir + "imgs/" + x)
        x = self.preprocess(x)
        y = self.y[idx]
        return x, y


def main():
    loss_fn = torch.nn.BCELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 50

    best_val_acc = 0
    patience = 0
    for e in range(epoch):
        print("##### Epoch {} #####".format(e))
        train_loop(loss_fn, optimizer)
        val_acc = valid_loop(loss_fn)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./saved/handnet.pth.tar")

        else:
            patience += 1
            if patience >= epoch * 0.2:
                print("End of training")
                break


def train_loop(loss_fn, optimizer):
    model.fc.train()
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out.squeeze(), y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("Train loss: {:.5f}".format(loss.item()))


def valid_loop(loss_fn):
    model.eval()
    val_loss = 0
    val_acc = 0
    num = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_loader):
            num += x.shape[0]
            x = x.to(device)
            y = y.to(device).float()

            out = model(x)
            loss = loss_fn(out.squeeze(), y)
            val_loss += loss.item() * x.shape[0]

            pred = torch.round(out).squeeze()
            correct = torch.sum(torch.eq(pred, y)).item()
            val_acc += correct

    print("Validation loss: {:.5f}, Validation acc: {}".format(val_loss, val_acc / num))
    return val_acc


if __name__ == "__main__":

    # prepare dataloader
    dataset = HandDataset()
    train_set, valid_set = torch.utils.data.random_split(
        dataset, [int(0.9 * dataset.__len__()), dataset.__len__() - int(0.9 * dataset.__len__())]
    )
    # is it imbalanced?
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False)

    # prepare model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    model = HandNet(vgg11(pretrained=True))
    model.to(device)
    main()
