from dataset import MNIST, collater
from model import Model
import os
import torch
from torch.utils.data import DataLoader
import numpy

reg_model = Model(num_classes=2, num_instances=100).cuda('0')
reg_model.train()
mnist = MNIST()
data_loader = DataLoader(mnist, batch_size=8, collate_fn=collater, pin_memory=True)

params = [p for p in reg_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-3)

cal_loss = torch.nn.L1Loss()

num_epoch = 0
max_epoch = 100
iter = 0

for num_epoch in range(max_epoch):
    for data in data_loader:
        img, gt = data
        img = img.cuda('0')
        gt = gt.cuda('0')

        optimizer.zero_grad()
        y_logits = reg_model(img).view(-1)

        loss = cal_loss(y_logits, gt)
        loss.backward()
        optimizer.step()

torch.save(reg_model, "final_model.pth")
