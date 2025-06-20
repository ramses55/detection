# detection
# Command-Line Python Tool

![Python Version](https://img.shields.io/badge/python-3.11.2-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A cross-platform Python script for people detection in video using Faster R-CNN.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ramses55/detection
   cd detection
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage


Basic command structre
```bash
python3 main.py [-h] -f FILENAME [-o OUTPUT] [-t THRESHOLD]
options:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Input video file to process
  -o OUTPUT, --output OUTPUT
                        Processed video name, default is "out.mp4"
  -t THRESHOLD, --threshold THRESHOLD
                        Confidenece threshold, default is 0.5
```
## Example 
```bash
python3 main.py -f crowd.mp4 -o result.mp4

