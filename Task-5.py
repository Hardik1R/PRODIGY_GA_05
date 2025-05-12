import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image and resize smaller to reduce memory usage
def load_image(path, max_size=256):
    image = Image.open(path).convert('RGB')
    size = max_size
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

# Display tensor image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (b * c * h * w)

content_path = r"C:/Users/AMAN/OneDrive/Desktop/Python Project/Prodigy Infotech/suii.jpg"
style_path = r"C:\Users\AMAN\OneDrive\Desktop\Python Project\Prodigy Infotech\Ronaldo.jpg"

content = load_image(content_path)
style = load_image(style_path)

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

content_layers = ['21']  # conv4_2
style_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

content_features = get_features(content, vgg, content_layers)
style_features = get_features(style, vgg, style_layers)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

input_img = content.clone().requires_grad_(True)

style_weight = 1e6
content_weight = 1

# Use Adam optimizer for stability
optimizer = optim.Adam([input_img], lr=0.01)

max_steps = 300
for step in range(max_steps):
    optimizer.zero_grad()
    input_features = get_features(input_img, vgg, content_layers + style_layers)

    content_loss = torch.mean((input_features[content_layers[0]] - content_features[content_layers[0]]) ** 2)

    style_loss = 0
    for layer in style_layers:
        input_gram = gram_matrix(input_features[layer])
        style_loss += torch.mean((input_gram - style_grams[layer]) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward(retain_graph=True)  # Retain the graph for the next iteration
    optimizer.step()

    if step % 50 == 0 or step == max_steps - 1:
        print(f"Step {step+1}/{max_steps} - Total Loss: {total_loss.item():.4f}")

input_img.data.clamp_(0, 1)
imshow(input_img, "Neural Style Transfer Output")
