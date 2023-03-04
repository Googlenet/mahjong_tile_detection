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

def trivial_transform():
    trivial_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
    ])

rot_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor()
])

gray_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(3),
    transforms.ToTensor()
])