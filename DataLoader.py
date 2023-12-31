import PIL.Image as Image
import torch.utils.data as data
import torchvision
import cv2


def executeSingle(path: str, size=128, step=128, mode='RGB'):
    listPatch = []
    img = Image.open(path).convert(mode)
    width, height = img.size
    i = height // 128
    j = width // 128
    for x in range(size, height, step):
        for y in range(size, width, step):
            patch = img.crop((y - size, x - size, y, x))
            listPatch.append(patch)
    return listPatch, i , j


def readAll(path: str, num: int, size=128, step=128):
    listO = []
    listT = []
    for index in range(1, num + 1):
        listO += executeSingle(path + '/o/{}.jpg'.format(index), size=size, step=step)[0]
        listT += executeSingle(path + '/t/{}_gt.jpg'.format(index), size=size, step=step, mode='L')[0]
    return Dataset(listO, listT)


class Dataset(data.Dataset):
    def __init__(self, listO, listT):
        transform = torchvision.transforms.ToTensor()
        self.listO = [transform(o) for o in listO]
        self.listT = [transform(t) for t in listT]

    def __len__(self):
        return len(self.listO)

    def __getitem__(self, index):
        return self.listO[index], self.listT[index]

    def getLoader(self, batchSize) -> data.DataLoader:
        return data.DataLoader(
            dataset=self,
            batch_size=batchSize,
            shuffle=True
        )
