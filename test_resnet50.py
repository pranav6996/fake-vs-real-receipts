import timm, torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=2)
ckpt = torch.load('/content/drive/MyDrive/fake-vs-real-receipts/efficientnet/epoch_models/EfficientNet_B3_epoch_04.pth', map_location='cuda')
model.load_state_dict(ckpt['model_state'])
model.eval().to('cuda')

target_layers = [model.conv_head]
cam = GradCAM(model=model, target_layers=target_layers)

transform = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

def check(image_path, true_label):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((448, 448))
    rgb_img = np.array(img_resized).astype(np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to('cuda')

    output = model(input_tensor)
    pred = output.argmax(dim=1).item()
    conf = torch.softmax(output, dim=1)[0][pred].item()

    targets = [ClassifierOutputTarget(pred)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(rgb_img); axes[0].set_title(f"True: {true_label}"); axes[0].axis('off')
    axes[1].imshow(visualization)
    axes[1].set_title(f"Pred: {['Real','AI'][pred]} ({conf*100:.1f}%)")
    axes[1].axis('off')
    plt.show()

check("/content/drive/MyDrive/download3.jpeg", "Real")
check("/content/drive/MyDrive/download.png", "Ai")