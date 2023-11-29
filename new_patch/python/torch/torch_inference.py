import torch
import heapq

resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
resnet50.eval()

import os
pic_dir = "../../pics/"
file_to_predict = [pic_dir + f for f in os.listdir(pic_dir) if os.path.isfile(pic_dir + f) ]

from PIL import Image
from torchvision import transforms

for filename in file_to_predict:
  input_image = Image.open(filename)
  preprocess = transforms.Compose([
      transforms.Resize(224),
      #transforms.CenterCrop(224),
      transforms.ToTensor(),
      #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  output = resnet50(input_batch)

  # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
  res = list(output[0].detach().numpy())
  index = heapq.nlargest(5, range(len(res)), res.__getitem__)

  print("\npredict picture: " + filename)
  # Read the categories
  with open("../imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(5):
      print("         top " + str(i+1) + ": " + categories[index[i]])
