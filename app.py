import gradio as gr
import timm
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Prediction function
def predict(image):
    if image is None:
        return "❌ Please upload an image"

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    classes = ["Fake", "Real"]
    predicted = classes[probs.argmax().item()]
    confidence = probs.max().item() * 100

    if predicted == "Fake":
        return f"🚨 FAKE ({confidence:.1f}%)"
    else:
        return f"✅ REAL ({confidence:.1f}%)"

# Gradio UI
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"]),
    outputs=gr.Text(label="🔍 Result"),
    title="🛡️ Deepfake Detector",
    description="📸 Upload or use Webcam to detect Real or Fake face!",
    theme=gr.themes.Soft()
)

app.launch()