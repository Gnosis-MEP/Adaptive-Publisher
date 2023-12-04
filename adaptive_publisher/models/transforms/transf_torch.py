
from torchvision import transforms
from torchvision.transforms.transforms import InterpolationMode


def get_transforms_torch(mode='CLS'):
    transform = None
    if mode.upper() == 'OBJ':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(640, interpolation=InterpolationMode.BILINEAR),
        ])
    if mode.upper() == 'CLS':
        transform = transforms.Compose([
            transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform

get_transforms = get_transforms_torch