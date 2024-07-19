
import torch
from torchvision import transforms
from torchvision import models
from torchvision.models import AlexNet_Weights
import urllib
import matplotlib.pyplot as plt
from PIL import Image

print(torch.__version__)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#url = "https://pytorch.tips/coffee"
url = "https://cdn.britannica.com/69/155469-131-14083F59/airplane-flight.jpg"

fpath = "image.jpg"

urllib.request.urlretrieve(url, fpath)
image = Image.open("image.jpg")
plt.imshow(image)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.445, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image)
batch = image_tensor.unsqueeze(0)
print(image_tensor.device, image_tensor.type, image_tensor.shape)
print(batch.shape)

model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
print(model)
model.eval()
model.to(device)
y = model(batch.to(device))
print(y.shape)
max, index = torch.max(y, 1)
print(index, max)
url = "https://pytorch.tips/imagenet-labels"
fpath = 'imagenet_class_labels.txt'
urllib.request.urlretrieve(url, fpath)
with open("imagenet_class_labels.txt") as f:
      classes = [line.strip() for line in f.readlines()]

prob = torch.nn.functional.softmax(y, dim=1)[0]*100
_, indices = torch.sort(y, descending=True)
for idx in indices[0][:5]:
    print(classes[idx], prob[idx].item())



