import cv2
from SimpleHRNet import SimpleHRNet

model = SimpleHRNet(48, 4, "/home/maihaoxiang/other_person/Fetal-head-segmentation-and-circumference-measurement-from-ultrasound-images-master/save_weights/model-999.pth")
image = cv2.imread("/home/maihaoxiang/dataset/HC18/training_set/001_HC.png", cv2.IMREAD_COLOR)

joints = model.predict(image)
print(joints)