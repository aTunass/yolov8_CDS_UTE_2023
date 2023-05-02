import socket       
import sys      
import time
import cv2
import numpy as np
import json 
import base64
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
from unet import UNet
import torch.nn.functional as F
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from numpy import random
from controller_test_v80 import Controller
# from duphong2_v8_v80 import Controller
# from duphong_v8_v80 import Controller
from utils_segment.data_loading import BasicDataset
# from Controller_Uit_45 import Controller
import logging
from ultralytics import YOLO
# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
# from ultralytics import YOLO

# model = YOLO("best.pt")
# Define the port on which you want to connect 
port = 54321                
pre_t = time.time()
current_speed = 0
current_angle = 0
# connect to the server on local computer  
s.connect(('127.0.0.1', port)) 
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoint_epoch300.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()
def predict_img(net,
                full_img,
                device,
                scale_factor,
                out_threshold):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()
def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)
device = 'cuda'
torch.cuda.is_available()
""" Load the checkpoint """
args = get_args()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

net = UNet(n_channels=3, n_classes=3, bilinear=args.bilinear)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Loading model {args.model}')
logging.info(f'Using device {device}')

net.to(device=device)
state_dict = torch.load(args.model, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)

logging.info('Model loaded!')

sendBack_angle = 0
sendBack_speed = 0
"""OD"""
model_OD = YOLO("best_v8_new.pt")
device = 'cuda'
model_OD.to(device)
def Control(angle, speed):
    global sendBack_angle, sendBack_speed
    sendBack_angle = angle
    sendBack_speed = speed
def PID(err, Kp, Ki, Kd):
    global pre_t
    err_arr[1:] = err_arr[0:-1]
    err_arr[0] = err
    delta_t = time.time() - pre_t
    pre_t = time.time()
    P = Kp*err
    D = Kd*(err - err_arr[1])/delta_t
    I = Ki*np.sum(err_arr)*delta_t
    angle = P + I + D
    return int(angle)
def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove
check = 50
check_err=0
line = 40
angle = 0
speed = 100
right=0
left=0
straight=0
no_turn_left=0
no_turn_right=0
lefttest = 0
righttest = 0
count_check = 0
err_arr = np.zeros(5)
if __name__ == "__main__":
    try:
        """
            python client.py  
            - Chương trình đưa cho bạn 3 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [0, 150]
            """
        while True:
            # Send data để điều khiển xe
            message = bytes(f"{angle } {speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)
            try:
                data_recv = json.loads(data)
            except:
                continue
            try:
                current_angle = data_recv["Angle"]
                current_speed = data_recv["Speed"]
                # print('current_speed', current_speed)

                
            except Exception as er:
                print(er)
                pass
            try:
                start = time.time()
                jpg_original = base64.b64decode(data_recv["Img"])
                jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
                imgage = cv2.imdecode(jpg_as_np, flags=1)
                img_OD = imgage
                image_name = "img_OD.jpg"
                cv2.imwrite(image_name, img_OD)
                image = imgage[200:,:]
                image_resize = cv2.resize(image, (160, 80))
                image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
                # image_test = cv2.cvtColor(cv2.imread('img_OD_trai.jpg'), cv2.COLOR_RGB2BGR)
                # cv2.imwrite('fix.jpg', image)
                img = Image.fromarray(image_resize)
                # print(image_resize.shape)
                """DETECT OD YOLOV7"""
                """ 0: No
                    1: turn_right
                    2: straight
                    3: no turn left
                    4: no turn right
                    5: no straight
                    6: car
                    7: unknown
                    8: turn left"""
                torch.cuda.is_available()
                # with torch.no_grad():
                #     xmin, ymin, xmax, ymax, conf_OD, cls_OD = detect('img_OD.jpg', device, img_size=320, iou_thres=0.25, conf_thres=0.25, net=model_od)
                try:
                    results = model_OD.predict(source="img_OD.jpg")
                    for result in results:
                    # detection
                        result = result.to("cpu")
                        result = result.numpy()
                        xmin, ymin, xmax, ymax = result.boxes.xyxy[0,0], result.boxes.xyxy[0,1], result.boxes.xyxy[0,2], result.boxes.xyxy[0,3]   # box with xyxy format, (N, 4)
                        conf_OD = result.boxes.conf[0]   # confidence score, (N, 1)
                        cls_OD = result.boxes.cls[0]    # cls, (N, 1)
                    S_OD = (xmax - xmin)*(ymax - ymin)  # S>1000, nga ba arrmax = 160
                except Exception as er:
                    xmax=0
                    xmin=0
                    cls_OD=0
                    conf_OD=0
                    S_OD=0
                    pass
                """DETECT LANE"""
                # mask = predict_img(net=net,
                #            full_img=image_resize,
                #            device=device,
                #            scale_factor=args.scale,
                #            out_threshold=args.mask_threshold,
                #            )
                # result = mask_to_image(mask)
                # out = np.array(result)
                mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
                result = mask_to_image(mask, mask_values)
                result = result.convert('L')
                out = np.array(result)
                # img_cv = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                img_remove = remove_small_contours(out)
                edges = img_remove
                # cv2.imwrite('fixed.jpg', img_remove)
                '''Controller'''
                angle, speed, check_err, right, left, straight, no_turn_left, no_turn_right, lefttest, righttest, count_check = Controller(edges=edges, PID=PID, current_speed=current_speed, current_angle=current_angle, 
                                                     check_err=check_err, xmax=xmax, xmin=xmin, conf=conf_OD, cls=cls_OD, right=right,
                                                      left=left, straight=straight, S=S_OD, notleft=no_turn_left, notright=no_turn_right, 
                                                      lefttest=lefttest, righttest=righttest, count_check=count_check)
                end = time.time()
                fps = 1 / (end - start)
                # if (check==50):
                #     print('------------------+SPK_SANDBOX+--------------------')
                #     check=0
                # check = check + 1
                print(S_OD, conf_OD, cls_OD)
                print(fps)
                # cv2.imshow("IMG", image_test)
                # cv2.imshow("IMG", edges)
                key = cv2.waitKey(1)
            except Exception as er:
                print(er)
                speed = -45
                pass
    finally:
        print('closing socket')
        s.close()
