from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np



def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for i in range(0, nc):
        idx = image == i
        r[idx] = label_colors[i, 0]
        g[idx] = label_colors[i, 1]
        b[idx] = label_colors[i, 2]


    rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(net, path, show_orig=True, dev='cuda'):
    img = Image.open(path)

    if show_orig:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    trf = T.Compose([T.Resize(640), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()


# Model 1
# fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
# img = './Images/horse.jpeg'
# segment(fcn, img)

# Model 2
dlab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
img = './Images/dog-human.jpg'
segment(dlab, img)






# fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
# img = Image.open('Images/bird.jpg')
# plt.imshow(img)
# plt.show()
# trf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
# inp = trf(img).unsqueeze(0)
# out = fcn(inp)['out']
# print(out.shape)
# om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
# print(om.shape)
# print(np.unique(om))
# rgb = decode_segmap(om)
# plt.imshow(rgb)
# plt.show()
