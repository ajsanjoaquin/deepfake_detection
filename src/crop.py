from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from mtcnn.mtcnn import MTCNN
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Finds and crops the faces in a set of images in a folder ')
parser.add_argument('--i','--input',type=str, default='.', help='input_path')
parser.add_argument('--o','--output',type=str, default='../cropped_images',help='output_folder')


args = parser.parse_args()
 
 #Parameters
detector = MTCNN(min_face_size=180)
tobe_filtered=args.i
if not os.path.isdir(tobe_filtered):
    os.mkdir(tobe_filtered)
if not os.path.isdir(args.o):
    os.mkdir(args.o)

j=0
for file in tqdm(os.listdir(tobe_filtered)): 
  if os.path.isfile(os.path.join(tobe_filtered,file)) and file.endswith(".png"):
    j+=1
    path=os.path.join(tobe_filtered,file)
    img=cv2.imread(path)
    if type(img) == type(None):
      pass
    else:
      pixels = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
      # create the detector, using default weights
      # detect faces in the image
      faces = detector.detect_faces(pixels)
      # display faces on the original image
      # save each face as an image
      for i in (range(len(faces))):
        # get coordinates
        x1, y1, width, height = faces[i]['box']
        x2, y2 = x1 + width, y1 + height
        fig=pyplot.figure(figsize=[6.4, 4.8])
        ax = pyplot.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        # plot face
        try: 
          ax.imshow(pixels[y1:y2, x1:x2])
          pyplot.savefig(os.path.join(args.o,'cropped_{}'.format(j)),\
                        bbox_inches = 'tight', pad_inches = 0, dpi=100)
        except ValueError:
          pass
        pyplot.close()