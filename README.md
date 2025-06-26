# Sign Language Recognition (MediaPipe + KNN)

A real-time sign language recognition system using your webcam, [MediaPipe](https://google.github.io/mediapipe/), and a simple KNN classifier. No pre-trained model files required! You collect your own samples for each sign, making it easy to adapt to your own hand and environment.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features
- Real-time hand landmark detection using MediaPipe
- User-driven training: collect your own samples for each sign
- KNN classifier for sign recognition (default: A, B, C)
- Tkinter GUI with webcam feed and recognized sign display
- Speech output for recognized sign (pyttsx3)
- Easily extensible to more signs

## Setup
1. **Clone this repository:**
   ```bash
   git clone https://github.com/HarshitAgarwal05/Sign-Language-Recognition.git
   cd Sign-Language-Recognition
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the program:**
   ```bash
   python sign_language_recognizer.py
   ```
2. **Collect samples:**
   - Hold your hand in the shape of sign 'A' and press the `a` key repeatedly (or hold it down) until 30 samples are collected.
   - Repeat for 'B' (`b` key), and 'C' (`c` key).
   - You can add more signs by editing the `SIGNS` list in the code.
3. **Train the classifier:**
   - Press `t` to train the classifier after collecting samples for each sign.
4. **Recognition:**
   - Show a sign to the camera. The program will display and speak the recognized sign.
   - Press `q` or close the window to quit.

## Customization
- To recognize more signs, add them to the `SIGNS` list in `sign_language_recognizer.py` and collect samples for each.
- You can adjust `SAMPLES_PER_SIGN` for more or fewer training samples.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE) 
