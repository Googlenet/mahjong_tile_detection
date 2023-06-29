from torchvision import transforms


# transform that resizes to size (length, width)
def size_transform_tensor(length, width):
    return transforms.Compose([
        transforms.Resize((length, width)),
        transforms.ToTensor()
    ])


def size_transform(length, width):
    return transforms.Compose([
        transforms.Resize((length, width)),
    ])


def trivial_transform_tensor(length, width):
    return transforms.Compose([
        transforms.Resize((length, width)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])


def rot_transform_tensor(length, width):
    return transforms.Compose([
        transforms.Resize((length, width)),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor()
    ])


def gray_transform_tensor(length, width):
    return transforms.Compose([
        transforms.Resize((length, width)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
