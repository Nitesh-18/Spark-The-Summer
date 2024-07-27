import torch
import cv2
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

class DetectionPipeline:
    
    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):

        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
    
    def __call__(self, filename):

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        frames = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                frames.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    faces.extend(self.detector(frames))
                    frames = []

        v_cap.release()

        return faces    

def process_faces(faces, resnet, device):
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    faces = torch.cat(faces).to(device)

    # Generate facial feature vectors using a pretrained model
    embeddings = resnet(faces)

    # Calculate centroid for video and distance of each face's feature vector from centroid
    centroid = embeddings.mean(dim=0)
    x = (embeddings - centroid).norm(dim=1).cpu().numpy()
    
    return x

def load_models_and_predict(video_path, mtcnn_path, resnet_path, bias, weight):
    # Device configuration
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load MTCNN model
    mtcnn = MTCNN(margin=14, keep_all=True, factor=0.5, device=device)
    mtcnn.load_state_dict(torch.load(mtcnn_path, map_location=device))
    mtcnn.eval()

    # Load Inception Resnet V1 model
    resnet = InceptionResnetV1(pretrained=None, device=device)
    resnet.load_state_dict(torch.load(resnet_path, map_location=device), strict=False)
    resnet.eval()

    # Define detection pipeline
    detection_pipeline = DetectionPipeline(detector=mtcnn, batch_size=60, resize=0.25)

    # Process video
    with torch.no_grad():
        faces = detection_pipeline(video_path)
        if faces:
            embeddings = process_faces(faces, resnet, device)
            prob = 1 / (1 + np.exp(-(bias + (weight * embeddings).mean())))
            print(prob)
            return 'real' if prob < 0.5 else 'fake'
        else:
            return 'undetermined'

# Example usage
def get_video_prediction(video_path):
    # video_path = r'test_video.mp4'
    
    mtcnn_path = r'models\mtcnn_state_dict.pt'
    resnet_path = r'models\inception_resnet_v1_state_dict.pt'
    bias = -0.2942
    weight = 0.68235746
    print("Processing....")
    result = load_models_and_predict(video_path, mtcnn_path, resnet_path, bias, weight)
    print(f'The video is {result}.')
