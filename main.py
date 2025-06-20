import argparse
import sys
import cv2
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from pathlib import Path



def transform_numpy(image_np):
    ''' Transforms from numpy and applies transforamtions for network'''

    image_pil = Image.fromarray(image_np)
    return weights.transforms()(image_pil)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument( '-f', '--filename', type=str, required=True,
                    help='Input video file to process')


parser.add_argument('-o', '--output', type=str, default='out.mp4',
                    help='Processed video name, default is "out.mp4"')


parser.add_argument('-t', '--threshold', type=float, default=0.5,
                    help='Confidenece threshold, default is 0.5')

args=vars(parser.parse_args())

VIDEO_PATH = Path(args['filename']).resolve()
OUTPUT_VIDEO_PATH = Path(args['output']).resolve()
CONFIDENCE_THRESHOLD = 0.5

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video source.")
    sys.exit()



length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

#pretrained weights are used
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
model.to(device)

class_names = weights.meta["categories"]

i=0
print("Number of processed frames:")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    i+=1
#    if i>200:
#        break

    if i % 100 == 0:
        print(f"{i}/{length}")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_tensor = transform_numpy(rgb_frame)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(input_batch)[0]

    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score > CONFIDENCE_THRESHOLD and class_names[label]=='person':
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[label]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {score:.2f}',
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 1)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
