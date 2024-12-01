
from torchvision import models
from torchvision.models import VGG16_Weights

model = models.vgg16(weights=VGG16_Weights)
print(model)

