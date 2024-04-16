import torch
from .models.resnet import createModel
import torchvision.transforms as transforms
from PIL import Image
import pickle as p


def getTestData(image_file, normalized=False):
    resized_image = image_file.resize((32, 32))
    resized_image.convert('RGB')
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                        std=[0.2675, 0.2565, 0.2761])
    common_trans = [transforms.ToTensor()]
    if normalized:
        common_trans.append(normalize)
    test_compose = transforms.Compose(common_trans)
    input_im = test_compose(resized_image)
    
    return input_im.unsqueeze(0)


def test(test_data, model, topk=5):
    model.eval()
    with torch.no_grad():
        test_data = test_data
        outputs = model(test_data).data
        conf, pred = outputs.topk(topk, 1, True, True)
    return torch.nn.functional.softmax(conf.squeeze(), dim=0), pred.squeeze()

def Predict(checkpoint_path, depth, test_file, device, topk):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    model = createModel(**vars(args))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    input = getTestData(test_file).to(device)
    return test(input, model, topk=topk)

def load_labels_name(filename):
    with open(filename, 'rb') as f:
        obj = p.load(f)
    return obj
