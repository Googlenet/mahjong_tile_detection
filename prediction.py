import torch
import torchvision
from tile_cnn import tile_cnn
import matplotlib.pyplot as plt
from typing import List
from torchvision import datasets
import transforms
from pathlib import Path

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image and plots the image with its prediction."""

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0))  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()


test_dir = 'tiles/test'
test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform=transforms.size_transform(128, 128))

loaded_model = tile_cnn(input_shape=3,  # number of color channels (3 for RGB)
                  hidden_units=3,
                  output_shape=len(test_data_simple.classes)).to(device)


# Load in the saved state_dict()
MODEL_PATH = Path("models")
MODEL_NAME = "ver10_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


custom_image_path = 'tiles/test/char/IMG_20230226_175100856.jpg'
class_names = test_data_simple.classes

# Pred on our custom image
pred_and_plot_image(model=loaded_model,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=transforms.size_transform(128, 128),
                    device=device)
