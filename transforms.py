from torchvision import transforms

# transform that resizes to size (length, width)
def simple_transform(length, width):
    transforms.Compose([
    transforms.Resize((length, width)),
    transforms.ToTensor()
    ])

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