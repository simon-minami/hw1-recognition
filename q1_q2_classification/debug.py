from train_q2 import ResNet
import torch
model = ResNet(num_classes=20)
model.eval()
dummy = torch.rand(1,3,224,224)
with torch.inference_mode():
    # for name, module in model.named_modules():
    #     print(name, module)
    # print(model.resnet.fc)
    # result = model(dummy)
    # for param in model.resnet.fc.parameters():
    #     print(param)

    for tag, value in model.resnet.named_parameters():
        if tag == 'layer1.1.conv1.weight' or tag == 'layer4.0.bn2.bias':
            print(tag)