import os
from werkzeug.utils import secure_filename
from datetime import datetime

UPLOAD_FOLDER = "static/uploads"

def save_post(content, image, video):
    if not content:
        raise ValueError("Content is required")

    image_filename = None
    video_filename = None

    if image and image.filename:
        image_filename = secure_filename(image.filename)
        image.save(os.path.join(UPLOAD_FOLDER, image_filename))

    if video and video.filename:
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(UPLOAD_FOLDER, video_filename))

    # Print the post content to the terminal
    print("Post content:", content)
    print("Image filename:", image_filename)
    print("Video filename:", video_filename)

    return {
        "content": content,
        "image": image_filename,
        "video": video_filename,
        "timestamp": datetime.now()
    }
