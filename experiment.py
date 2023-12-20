import os
import numpy as np
import cv2

def duplicate_images(path):
    extensions = ["jpg", "bmp", "png"]
    for dirname, dirnames, filenames in os.walk(os.path.abspath(path)):
        for filename in filenames:
            abspath = os.path.abspath(dirname + "\\" + filename)
            name, _, extension = abspath.rpartition('.')
            if extension in extensions:
                img = np.array(cv2.imread(abspath, 1))
                img = np.concatenate([img, img], axis=1)
                img = np.concatenate([img, img], axis=0)
                img = np.concatenate([img, img], axis=0)
                img = np.concatenate([img, img], axis=0)
                cv2.imwrite(abspath, img)

def cut_images(path):
    extensions = ["jpg", "bmp", "png"]
    for dirname, dirnames, filenames in os.walk(os.path.abspath(path)):
        for filename in filenames:
            abspath = os.path.abspath(dirname + "\\" + filename)
            name, _, extension = abspath.rpartition('.')
            if extension in extensions:
                img = np.array(cv2.imread(abspath, 1))
                h = img.shape[0]
                w = img.shape[1]
                cv2.imwrite(abspath, img[:h // 8, :w // 2])

def rename_images(path):
    extensions = ["jpg", "bmp", "png"]
    for dirname, dirnames, filenames in os.walk(os.path.abspath(path)):
        for filename in filenames:
            abspath = os.path.abspath(dirname + "\\" + filename)
            name, _, extension = abspath.rpartition('.')
            if extension in extensions:
                os.rename(abspath, name + "_gt.jpg")

def test_vit():
    import torch
    from vit_pytorch import ViT

    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)  # (1, 1000)
    print(preds.shape)

    from vit_pytorch import SimpleViT

    v = SimpleViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)  # (1, 1000)
    print(preds.shape)

if __name__ == "__main__":
    # cut_images("./train_data")
    # rename_images("train_data/temp/t")
    test_vit()