#(1)IMPORT MODULES 
from tqdm import tqdm 
import cv2
import os 
import random 
import matplotlib.pylab as plt
#from matplotlib import gridspec
from glob import glob
import pandas as pd
import numpy as np



#(2)LISTING FILE DIRECTIORIES 
PATH = os.path.abspath(os.path.join("..", "C:\Program Files (x86)\kmh\Lib\site-packages\chestxray\input"))
SOURCE_FILE = os.path.join(PATH, "C:\Program Files (x86)\kmh\Lib\site-packages\chestxray\input\samples")
images = glob(os.path.join(SOURCE_FILE, "*.png"))              

#(3)READ DATASET FILE
labels = pd.read_csv("sample_labels.csv")

#Display some samples from the dataset 
print(images[0:5])
print(labels.head(10))



#Show random images 
r = random.sample(images, 4)
plt.figure(figsize =(16,16))

plt.subplot(141)
plt.imshow(cv2.imread(r[0]))

plt.subplot(142)
plt.imshow(cv2.imread(r[1]))

plt.subplot(143)
plt.imshow(cv2.imread(r[2]));

plt.subplot(144)
plt.imshow(cv2.imread(r[3]));
plt.show()                    #Essential line to diplay graph and images using matplotlib library



#(4)RESIZE IMAGE DIMENSIONS AND LABELLING 
def proc_images():
	disease = "Infiltration"

	#arrays to hold feature vectors(samples) and class labels 
	a = []       #samples(images) as arrays
	b = [] 	     #labels Infiltration or Not Infiltration
	
	WIDTH = 224
	HEIGHT = 224

	#for img in images:
	for img in tqdm(images):
		base = os.path.basename(img)
		finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]

	#Read and Resize images 
		full_size_image = cv2.imread(img)
		a.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
	
	#Labelling
		if disease in finding:
			finding = 1
			b.append(finding)

		else:
			finding = 0
			b.append(finding)

	return a,b 

a,b = proc_images()	




#(5)SET UP DATAFRAME(2 DIMENSIONAL DATA IN A TABULAR FASHION IN ROWS AND COLUMNS) 
df = pd.DataFrame()
#tqdm.pandas(desc="Processing...")
#df.progress_apply(proc_images)
df["labels"] = b
df["images"] = a 
print(len(df), df.images[0].shape)           #print length of image index and dimensions of dataframe #df.shape #len(df)
print(type(b))			
print(df.head(5))


#print(labels.head(5))


#(6)SAVE THE DATASET AS NUMPY ARRAYS
np.savez("a_images_arrays", a)
np.savez("b_labels", b)


print(os.listdir("C:\\Program Files (x86)\\kmh\\Lib\\site-packages\\chestxray\\input"))
