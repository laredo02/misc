
import torch

import load_data as ld
import model as m

def train_model(dataloader, model, lossfn, optim):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        pred = model(X)
        loss = lossfn(pred, y)
        loss.backward()
        optim.step()

        if batch%100 == 0:
            print(f"{loss.item()}, {batch}")

    print()

def test_model():
    nop

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

train_loader, test_loader = ld.getDataLoaders()

model = m.getModel(device)
lossfn = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), 0.001)

train_model(train_loader, model, lossfn, optim)

