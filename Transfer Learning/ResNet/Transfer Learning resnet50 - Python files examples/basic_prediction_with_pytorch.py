from PIL import Image
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms


classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
model = torch.load('model_tl_pytorch2.pth')

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def prediction(img_path, transformer):
    image = Image.open(img_path)

    image_tensor = transformer(image).float()

    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    input = Variable(image_tensor)

    output = model(input)

    index = output.data.numpy().argmax()

    pred = classes[index]

    return pred


image_path = "tulip.jpg"
result1 = prediction(image_path, transformer)

print(result1)
