import os
import numpy
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import StepLR
import DataLoader as D
import Net0_CNN as N0
import Net1_MSCNN as N1
import Net2_Unet as N2
import Net3_ViT as N3
import DiceLoss

threshold = 0.5

dataset = D.readAll('./train_data/temp', 10, 128, 128)
dataloader = dataset.getLoader(10)

GPU: bool = torch.cuda.is_available()
# model = N0.CNN()
# model = N1.MSCNN()
model = N2.UNet()
# model = N3.ViTAE()
optimizer = torch.optim.Adam(model.parameters())
scheduler = StepLR(optimizer, step_size=1, gamma=0.995)
cost = nn.MSELoss()

if GPU:
    model.cuda()
    cost.cuda()

# for epoch in range(1, 1001):
#     model.train()
#     meanLoss = []
#     for (imgO, imgT) in dataloader:
#         if GPU:
#             (imgO, imgT) = (imgO.cuda(), imgT.cuda())
#         imgReconT = model(imgO)
#         # loss = cost(imgT, imgReconT)
#         loss = DiceLoss.calc(imgT, imgReconT)
#         # loss = model.loss(imgReconT, imgT, cost)
#         meanLoss.append(loss.item())
#         loss.backward()
#         optimizer.step()
#         model.zero_grad()
#     scheduler.step()
#     meanLoss = numpy.mean(meanLoss)
#     if epoch % 1 == 0:
#         print('Epoch {} completed. Mean Loss: {:.3f}'.format(epoch, meanLoss))
#     if epoch % 1000 == 0:
#         folder = './result/'
#         name = 'net3-{}-mse.pkl'.format(epoch)
#         if not os.path.exists(folder):
#             os.makedirs(folder)
#         torch.save(model.state_dict(), folder + name)

# model.load_state_dict(torch.load('./pretrain/2017/net3-100-mse.pkl'))
model.load_state_dict(torch.load('./result/net2-1000-mse.pkl'))

model.eval()

# testImgs, h_i, w_i = D.executeSingle('./dataset/2017/o/15.bmp', size=128, step=128)
testImgs, h_i, w_i = D.executeSingle('./train_data/o/79.jpg', size=128, step=128)

outputImg = torch.zeros((1, h_i*128, w_i*128))
for index, img in enumerate(testImgs):
    transform = torchvision.transforms.ToTensor()
    img = transform(img).cuda().unsqueeze(dim=0)
    imgT = model(img).squeeze(dim=0).detach().cpu()
    imgT = torch.where(imgT > threshold, torch.tensor(1.0), torch.tensor(0.0))

    i, j = index // w_i, index % w_i
    outputImg[:, i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = imgT
torchvision.utils.save_image(outputImg, './result/res.jpg')
