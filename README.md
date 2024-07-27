# Deepfake AI Detection for Social Media

## Overview

This repository contains the User Interface (UI) and the Artificial Intelligence (AI) model developed to identify and block AI-generated deepfake images on social media platforms. The primary goal is to ensure that AI-generated images are detected and prevented from spreading on social media, thereby maintaining the integrity and authenticity of visual content.

## Technology Stack

- **HTML**: For structuring the web pages.
- **CSS**: For styling the web pages.
- **JavaScript**: For adding interactivity to the web pages.
- **Flask**: A micro web framework for Python to handle backend operations and API integration.
- **Python**: Used for developing the AI model for deepfake detection.
- **Python Libraries**:
  - TensorFlow/PyTorch: For building and training the deepfake detection model.
  - OpenCV: For image processing.
  - NumPy: For numerical operations.

## Repository Structure

```
deepfake-detection/
├── README.md
├── app/
│ ├── static/
│ │ ├── css/
│ │ │ └── styles.css
│ │ └── js/
│ │ └── scripts.js
│ ├── templates/
│ │ └── index.html
│ ├── app.py
│ └── model/
│ ├── model.py
│ └── utils.py
├── requirements.txt
└── .gitignore
```


## Getting Started

### Prerequisites

- Python 3.7 or above
- pip (Python package installer)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/deepfake-detection.git
    cd deepfake-detection
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Navigate to the `app` directory:
    ```bash
    cd app
    ```

2. Start the Flask server:
    ```bash
    python app.py
    ```

3. Open your browser and go to `http://127.0.0.1:5000` to view the application.

## Contributing

We welcome contributions to improve this project. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.


