from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.autograd import Variable 
import glob
import cv2
import numpy as np
import torch


useCuda = True
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=128, margin=0)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

if useCuda:
    resnet = resnet.cuda()

# from PIL import Image


# img = Image.open("./user_faces/b07901033/face_0.jpg")



def ReadImage(pathname):
    img = cv2.imread(pathname)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255.0
    # print(np.min(img), np.max(img))
    # print(np.sum(img[0]), np.sum(img[1]), np.sum(img[2]))
    I_ = torch.from_numpy(img).unsqueeze(0)
    if useCuda:
        I_ = I_.cuda()
    return I_
# Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path=<optional save path>)

user_faces_path = "./user_faces"

jpg_faces_files = sorted(glob.glob(user_faces_path + "/**/*.jpg", recursive = True))

studentids = [x.split('/')[2] for x in jpg_faces_files]

np.save("studentids.npy",studentids)
print("studentids.npy saved")

imgs = []
for img_path in jpg_faces_files:
    imgs.append(ReadImage(img_path))


# img_cropped = mtcnn(img)
# print(img_cropped.shape)
# exit()
# Calculate embedding (unsqueeze to add batch dimension)
I_ = torch.cat(imgs, 0)
I_ = Variable(I_, requires_grad=False)
# I_ = I_.cuda()
# print(I_.shape)
# exit()
# img_embedding = resnet(img_cropped.unsqueeze(0))
img_embedding = resnet(I_)
print(img_embedding.shape)
torch.save(img_embedding,"userfaces.pth")
print("userfaces.pth saved")
exit()
# # Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))




# print(jpg_faces_files)
# print(studentids)
# exit()
# img_paths = [	\
#     'example_faces/b07901033/face_0.jpg',	\
#     'example_faces/b07901033/face_1.jpg',	\
#     'example_faces/b07901033/face_2.jpg',	
#     # '/home/polphit/Downloads/face_images/lennon-2.jpg_aligned.png',	\
#     # '/home/polphit/Downloads/face_images/clapton-1.jpg_aligned.png',	\
#     # '/home/polphit/Downloads/face_images/clapton-2.jpg_aligned.png',	\
# ]


