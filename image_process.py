import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt

# Load the MTCNN and Inception Resnet V1 models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adjust MTCNN parameters
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.6, 0.7, 0.7])
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load model weights
mtcnn.load_state_dict(torch.load('models/mtcnn_state_dict.pt', map_location=device))
resnet.load_state_dict(torch.load('models/inception_resnet_v1_state_dict.pt', map_location=device))

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Increased size for better face detection
        transforms.ToTensor()
    ])
    image = transform(image)
    return image

# Visualize the image
def visualize_image(image_path):
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.show()

# Extract facial features
def extract_features(image_path):
    image = preprocess_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        print(f"Detected boxes: {boxes}")  # Debugging line
        if boxes is not None and len(boxes) > 0:
            faces = [image[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])] for box in boxes]
            faces = torch.stack(faces).to(device)
            embeddings = resnet(faces)
            return embeddings.mean(dim=0).cpu().numpy()
        else:
            return None

# Dummy classifier (replace this with your actual classifier)
def classify_features(features):
    # This is a placeholder classifier. Replace with your trained classifier.
    threshold = 0.5
    score = np.random.rand()  # Replace with actual classification logic
    return 'AI-generated' if score > threshold else 'real'

# Main function to classify image
def classify_image(image_path):
    visualize_image(image_path)  # Show the image for debugging
    features = extract_features(image_path)
    if features is not None:
        result = classify_features(features)
        return result
    else:
        return 'No face detected'

# Example usage
image_paths = ['438811170_17926612643856628_1930376114489558432_n.jpg', '452533163_17936997224856628_7150830372252250643_n.jpg']
for path in image_paths:
    path='test_images/'+path
    result = classify_image(path)
    print(f'The image {path} is {result}.')
