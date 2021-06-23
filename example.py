import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

import h5m


# the model we'll be training
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32, 32)

    def forward(self, x):
        # some target distribution...
        return nn.Softmax(dim=-1)(self.fc1(x))


# fake feature that simulate loading an example and a label from a source
class Example(h5m.Feature):

    # set some of the h5py.Dataset properties for each sub feature
    __ds_kwargs__ = dict(
        x=dict(chunks=(32,), compression="lzf"),
        label=dict(chunks=(1,))
    )

    def load(self, source):
        return dict(
            x=np.random.randn(32).astype(np.float32),
            label=np.random.randint(0, 32, (1,))
        )


# Our Dataset Set
class Dataset(h5m.Database, torch.utils.data.Dataset):
    data = Example()

    def __getitem__(self, item):
        # returns dict : {"x": ..., "label": ...}
        return self.data.get(self.data.ids[item])

    def __len__(self):
        # the number of sources we loaded :
        return len(self.data.ids)


# While consuming `train`, we will write to `logs` :
class ExerimentData(h5m.Database):
    # those are initialized from 1st add(...) statement
    loss = h5m.Feature()
    acc = h5m.Feature()
    # passing a state_dict initializes the configs of
    # the children datasets
    ckpts = h5m.TensorDict(Net().state_dict())

    def report(self):
        """do something with the collected data"""
        import matplotlib.pyplot as plt

        # print the .h5 structure
        self.info()

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(self.loss[()])
        ax1.set_title("loss")
        ax2.plot(self.acc[()])
        ax2.set_title("accuracy")

        # because of the ".", we have to use get_feat() - or getattr(self.ckpts, "fc1.weight")
        w = self.get_feat("ckpts/fc1.weight")
        epochs = w.ids
        f, axs = plt.subplots(1, len(epochs))
        for i, ep in enumerate(epochs):
            axs[i].imshow(w.get(ep))
            axs[i].set_title(f"epoch - {ep}")

        plt.show()


# now the script

MAX_EPOCHS = 30

# build the dataset from sources (no split for simplicity...)
sources = [str(i) for i in range(200)]
train = Dataset.create("train.h5", sources, "w", keep_open=True)
train.info()

net = Net()
if torch.cuda.is_available():
    net = net.to("cuda")
opt = torch.optim.Adam(net.parameters())
loader = torch.utils.data.DataLoader(train, num_workers=4,
                                     shuffle=True,
                                     batch_size=16, pin_memory=True)
n_batches = len(loader)

# create the logs
logs = ExerimentData("logs.h5", mode="w", keep_open=True)

logs.add("train", {"loss": np.zeros(MAX_EPOCHS * n_batches),
                   "acc": np.zeros(MAX_EPOCHS * n_batches)})

# here we go!
for epoch in tqdm(range(MAX_EPOCHS)):
    for i, batch in enumerate(loader):
        x, labels = batch["x"], batch["label"]
        if torch.cuda.is_available():
            x = x.to("cuda")
            labels = labels.to("cuda")
        opt.zero_grad()
        out = net(x)
        L = nn.NLLLoss()(out, labels.squeeze())
        L.backward()
        opt.step()
        acc = (out.max(dim=-1).indices == labels.squeeze()).sum() / labels.size(0)
        # set indices within the region "train"
        logs.loss.iset("train", epoch * n_batches + i, L.detach().item())
        logs.acc.iset("train", epoch * n_batches + i, acc.detach().item())

    if (epoch + 1) % 10 == 0:
        # log checkpoints
        logs.add(str(epoch), {"ckpts": logs.ckpts.format(net.state_dict())})
        # flush every 10 epochs...
        logs.flush()

logs.report()

# teardown
train.close()
logs.close()
os.remove("train.h5")
os.remove("logs.h5")
