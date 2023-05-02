# yolov8_CDS_UTE_2023
## Introduces
This project is used to participate in the Autonomous Vehicle Racing Competition at the University of Technical Education in 2023
## Descriptions
- Use yolov8 to recognize signs and vehicles
- Use Unet for lane detection
- max speed: 80
## Advantages and disadvantages
``Advantages``
- Use the linear function to get the speed according to the deflection of the road
- Use the function to determine the delay when turning

``Disadvantages``
- Using yolo to identify obstacles is not as good as using Unet
- Functions are not optimized
## Setting
``pip install -r requirements.txt``
## Inference
``python client.py``
## Result
[Map_demo](https://www.youtube.com/watch?v=bjkq4dZFzao)
