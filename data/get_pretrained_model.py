import torch
import torchvision

img_tsr = torch.rand(1, 3, 1000, 600)
model = torchvision.models.vgg16(True)
model.eval()
traced = torch.jit.trace(model, img_tsr)
traced.save('vgg16.pt')