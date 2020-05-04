# robotic_grasping-
Flexible image-based robotic grasping solution. 


# Bibliothèques et Pré-requis

Pour pouvoir être exploité, le jeu de données doit d'abord être sur la machine virtuelle Colab. Ce code est là pour ça. Si vous n'utilisez pas votre machine pendant  un certains temps, elle est recyclée: les fichiers téléchargés ne sont pas gardés. IL faut alors tout recompiler.

!pip install wget
!pip install tqdm 

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os #pour créer des dossiers
import glob #pour manipuler les noms de fichiers
import wget #pour télécharger des fichiers sur internet
import tarfile, zipfile #pour décompresser des fichiers
import shutil #used for copying files
from tqdm import tqdm #progress bar
from google.colab import files #for downloading/uploading purposes

# Obtention du jeu de données

Le code suivant permet d'extraire les images des grasps labelisés et les stocker.

def process_graspbox(name):
    """Permet d'extraire les points du fichier de labels"""
    f=open(name, 'r')
    graspbox = list(map(
        lambda coordinate: float(coordinate), f.read().strip().split()))    
    return graspbox

def process_rectangles(graspbox):
    """Rassembler chaque rectangle composé de 4 points"""
    rectangles_grasp=[]
    indices_min=[]
    for i in range(0,len(graspbox)):
        if i%8==0:
            NaN=0
            rect=graspbox[i:i+8]
            points=[]
            ind_mini=0
            mini=rect[0]
            for j in range(0,len(rect),2):
                points.append(rect[j:j+2])
                if rect[j]<mini:
                    mini=rect[j]
                    ind_mini=j
            for k in rect:
                if np.isnan(k)==True:
                    NaN=1
            if NaN==0:
                rectangles_grasp.append(points)
                indices_min.append(ind_mini)
    return rectangles_grasp, indices_min

def rotMat2D(angle):
    return [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]

def pointsInAARect(imPoints,rectPoints):
    inRectX=np.not_equal(np.sign(imPoints[:,0] - rectPoints[0][0]),np.sign(imPoints[:,0]-rectPoints[2][0]))
    inRectY=np.not_equal(np.sign(imPoints[:,1] - rectPoints[0][1]),np.sign(imPoints[:,1]-rectPoints[2][1]))
    return np.logical_and(inRectX,inRectY)

def extract_grasp(graspbox,im):
    """ 
    Extract a grasp from an image 'im'. Enter the graspbox(array with the coordinates
    of the four vertices of the gras). The function outputs
    an (n,m,3) array that can then be saved as an image of the grasp.
    This code was in part adapted from a MATLAB code written by
    Ian Lenz and Kevin Lai's from Cornell's Robotics Lab. For practical purposes
    we chose to work with MATLAB's column-major rule to parse arrays. This is why 
    you will see many 'F' arguments.
    """
    rectPoints=graspbox
    im=np.array(im)
    im=im[:,:,:3]
    
    #build an index for each pixel in the image
    t1,t2=np.shape(im)[:2]
    [imX,imY]=np.meshgrid(np.arange(0,t2),np.arange(0,t1))
    imPoints=np.transpose([imX.flatten('F'),imY.flatten('F')])
    
    #find rotation angle and call rotation matrix
    ang=np.arctan2(rectPoints[0][1]-rectPoints[1][1],rectPoints[0][0]-rectPoints[1][0])
    rotMat=rotMat2D(ang)
    
    #rotated graspbox and image index:
    rectPointsRot = np.array(rectPoints) @ np.array(rotMat)
    imPointsRot = np.array(imPoints)@ np.array(rotMat)
    
    #determine if a pixel is inside the graspbox
    #and build a truth matrix to identify those inside
    inRect = pointsInAARect(imPointsRot,rectPointsRot)
    
    #keep only relevant pixels
    newPoints=imPointsRot[inRect]
    
    #turn the rotated image index of the pixels inside
    #the matrix into a coordinate system
    newPoints=newPoints-[newPoints[:,0].min(),newPoints[:,1].min()]
    
    #create an array the size of the grasp
    I2=np.zeros([int(np.around(newPoints[:,1].max())),int(np.around(newPoints[:,0].max())),3])
    
    newPoints=np.uint32(np.around(newPoints))
    
    #for each color in r,g,b:
    for i in range(3):
        #extract the image layer
        channel =im[:,:,i]
        #create a new channel for the output grasp
        newChannel=np.zeros((np.shape(I2)[0],np.shape(I2)[1]))
        
        inRect=np.reshape(inRect,np.shape(im)[:2],'F')
        height,length= np.shape(im)[:2]
        
        #fill the output grasp layer with the correct pixels
        l=0
        k=0
        for j in range(len(newPoints)):
            while inRect[k][l] ==False and l<=length:
                k=k+1
                if k==height:
                    k=0
                    l=l+1
        
            newChannel[newPoints[j][1]-1,newPoints[j][0]-1]= int(channel[k][l])
            k=k+1
            if k==height:
                k=0
                l=l+1
                
        I2[:,:,i]=newChannel
        
    I2=I2.astype(np.uint8)
    grasp_image = Image.fromarray(I2)
    return grasp_image

def sub2ind(array_shape, rows, cols):
    return rows+(cols)*array_shape[0]

def extract_grasp2(graspbox,im):
    """ 
    This is an alternative to the previous extract_grasp function. It performs the same
    task. I have not done any benchmarking at this point to determine which is the best,
    but I think this one might be easier to understand.
    
    Extract a grasp from an image 'im'. Enter the graspbox(array with the coordinates
    of the four vertices of the gras). The function outputs
    an (n,m,3) array that can then be saved as an image of the grasp.
    This code was in part adapted from a MATLAB code written by
    Ian Lenz and Kevin Lai's from Cornell's Robotics Lab. For practical purposes
    we chose to work with MATLAB's column-major rule to parse arrays. This is why 
    you will see many 'F' arguments.
    """
    rectPoints=graspbox
    im=np.array(im)
    im=im[:,:,:3]
    
    #build an index for each pixel in the image
    t1,t2=np.shape(im)[:2]
    [imX,imY]=np.meshgrid(np.arange(0,t2),np.arange(0,t1))
    imPoints=np.transpose([imX.flatten('F'),imY.flatten('F')])
    
    #find rotation angle and call rotation matrix
    ang=np.arctan2(rectPoints[0][1]-rectPoints[1][1],rectPoints[0][0]-rectPoints[1][0])
    rotMat=rotMat2D(ang)
    
    #rotated graspbox and image index:
    rectPointsRot = np.array(rectPoints) @ np.array(rotMat)
    imPointsRot = np.array(imPoints)@ np.array(rotMat)
    
    #determine if a pixel is inside the graspbox
    #and build a truth matrix to identify those inside
    inRect = pointsInAARect(imPointsRot,rectPointsRot)
    
    #keep only relevant pixels
    newPoints=imPointsRot[inRect]
    
    #turn the rotated image index of the pixels inside
    #the matrix into a coordinate system
    newPoints=newPoints-[newPoints[:,0].min(),newPoints[:,1].min()]
    
    #create an array the size of the grasp
    I2=np.zeros([int(np.around(newPoints[:,1].max()))+1,int(np.around(newPoints[:,0].max()))+1,3])
    
    newPoints=np.uint32(np.around(newPoints))
    
    #build a linear index both for the location of the pixel in the output and in the source image.
    newInd=[]
    for i in newPoints:
        newInd.append(sub2ind(np.shape(I2),np.uint32(i[1]),np.uint32(i[0])))

    inRect2=[]
    for i in range(len(inRect)):
        if inRect[i]==True:
            inRect2.append(i)
            
    #for each color in r,g,b:
    for i in range(3):
        #extract the image layer
        channel =im[:,:,i]
        #create a new channel for the output grasp
        newChannel=np.zeros((np.shape(I2)[0],np.shape(I2)[1]))
        
        #fill the output grasp layer with the correct pixels
        newChannel[np.unravel_index(newInd,newChannel.shape,'F')] = channel[np.unravel_index(inRect2,channel.shape,'F')]
                
        I2[:,:,i]=newChannel
        
    I2=I2.astype(np.uint8)
    grasp_image = Image.fromarray(I2)
    return grasp_image

def stockage_rect(rect,rect_nb,classe,rootFolder,im_name):
    """Stocker le rectangle dans des sous-dossiers, que la fonction créée s'ils n'existent pas déjà."""
    
    filename=str(classe)+'_'+str(rect_nb)+im_name[:-5]+'.png'
    outputFolder=rootFolder
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    rect.save(outputFolder +'/'+ filename, "PNG")
    return

def save_labeled_rect(imagefilename, dataFolder, saveFolder, classArray):
    """L'ensemble du process pour UNE image. Renvoi le nombre de rectangles de chaque classe 
    et une array avec la liste ordonnée des classes des rectangles de l'image."""
    os.chdir(dataFolder)
    im = Image.open(imagefilename)
    
    cpos=imagefilename[:-5]+'cpos.txt'
    rects_pos,indices_pos=process_rectangles(process_graspbox(cpos))
    
    cneg=imagefilename[:-5]+'cneg.txt'
    rects_neg,indices_neg=process_rectangles(process_graspbox(cneg))
    
    i,j=0,0
    
    for i in range(len(rects_pos)):
        grasp=extract_grasp(rects_pos[i],im)
        #grasp=extract_grasp2(rects_pos[i],im)
        stockage_rect(grasp,i,1,saveFolder,imagefilename)
        classArray.append(1)
    
    for j in range(len(rects_neg)):
        grasp=extract_grasp(rects_neg[j],im)
        #grasp=extract_grasp2(rects_pos[i],im)
        stockage_rect(grasp,j,0,saveFolder,imagefilename)
        classArray.append(0)
        
    return i,j,classArray



def walkdir(folder):
    """Walk through each files in a directory. Only used for the progress bar
    at this point.
    """
    
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))

def process_whole_dataset(dataFolder,saveFolder):
    """Parcours l'ensemble du jeu de donnée sauvée dans dataFolder et extrait tous les rectangles labelisés pour
    les stocker sous forme d'images dans une hierarchie de fichier dont le dossier root est saveFolder. Renvoi la liste
    ordonnée des classes des rectangles sous forme d'array."""
    classArray=[]
    directory = os.fsencode(dataFolder)
    
    filecounter = 0
    for filepath in walkdir(directory):
        filecounter += 1
    
    
    for file in tqdm(os.listdir(directory), total=filecounter,unit="files", leave=False):
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            i,j,classArray = save_labeled_rect(filename,dataFolder,saveFolder,classArray)
            continue
        else:
            continue
    return classArray

Le code suivant permet de télécharger la base de données directement sur le site de Cornell, et d'extraire les images zippées et d'obtenir les rectangles de grasps

for i in tqdm(range(1,11)):
  
  if i<10:
    url='http://pr.cs.cornell.edu/grasping/rect_data/temp/data0'+str(i)+'.tar.gz'
    data = wget.download(url)
    Udata= tarfile.open('data0'+str(i)+'.tar.gz','r:gz')
    uncompressed_data=Udata.extractall('/content/CompleteDataset')
    Udata.close()
    listeClasses=process_whole_dataset('/content/CompleteDataset/0'+str(i),'/content/ExtractedDataset/'+str(i))
  
  else:
    url='http://pr.cs.cornell.edu/grasping/rect_data/temp/data10'+'.tar.gz'
    data = wget.download(url)
    Udata= tarfile.open('data10'+'.tar.gz','r:gz')
    uncompressed_data=Udata.extractall('/content/CompleteDataset')
    Udata.close()
    listeClasses=process_whole_dataset('/content/CompleteDataset/10','/content/ExtractedDataset/'+str(i))

print('Done !')

Compresser la base de données extraite

def make_tarfile(output_dir,output_filename, source_dir):
    os.chdir(output_dir)
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

for i in range(1,11):
  make_tarfile('/content','compressed_grasps_'+str(i)+'.tar.gz','/content/ExtractedDataset/'+str(i))

Importer le code depuis un dossier Google Drive 

# Pour executer le code suivant, il faut avoir créé la connection avec un Google Drive (cf fin du code)
for i in tqdm(range(1,11)):
  shutil.copy('/content/compressed_grasps_'+str(i)+'.tar.gz','/content/drive')

Importer la base de données déjà extraite depuis Google Drive.

!ls /content/drive/'Arts et Métiers'/'PJT Grasping'/'Dataset 20_05/CompleteDataset.zip'

#import zipped dataset from google drive and extract it. 
#Requires connection with Google Drive (execute the code at the bottom of this notebook)

shutil.copy('/content/drive/Arts et Métiers/PJT Grasping/Dataset 20_05/CompleteDataset.zip','/content')

!ls /content/CompleteDataset.zip

Udata= zipfile.ZipFile('/content/CompleteDataset.zip','r')
uncompressed_data=Udata.extractall('/content/ExtractedDataset')
Udata.close()

!ls /content/ExtractedDataset

Le code suivant permet de créer les répertoires pour séparer, pour chacune des classes POS et NEG, les échantillons de train / validation / test

original_dataset_dir = '/content/ExtractedDataset'
base_dir = '/content/SeparatedDataset'
if not os.path.exists(base_dir):
  os.mkdir(base_dir)
train_dir = os.path.join(base_dir,'train')
if not os.path.exists(train_dir):
  os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
if not os.path.exists(validation_dir):
  os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
if not os.path.exists(test_dir):
  os.mkdir(test_dir)
train_pos_dir = os.path.join(train_dir,'pos')
if not os.path.exists(train_pos_dir):
  os.mkdir(train_pos_dir)
train_neg_dir = os.path.join(train_dir,'neg')
if not os.path.exists(train_neg_dir):
  os.mkdir(train_neg_dir)
validation_pos_dir = os.path.join(validation_dir,'pos')
if not os.path.exists(validation_pos_dir):
  os.mkdir(validation_pos_dir)
validation_neg_dir = os.path.join(validation_dir,'neg')
if not os.path.exists(validation_neg_dir):
  os.mkdir(validation_neg_dir)
test_pos_dir = os.path.join(test_dir,'pos')
if not os.path.exists(test_pos_dir):
  os.mkdir(test_pos_dir)
test_neg_dir = os.path.join(test_dir,'neg')
if not os.path.exists(test_neg_dir):
  os.mkdir(test_neg_dir)

Répartition des rectangles grasps par classe dans les échantillons train / validation / test

k=0
for name in glob.glob('/content/ExtractedDataset/0_*'):
  name=name[26:]
  if k<1500:
    src=os.path.join(original_dataset_dir, name)
    dst=os.path.join(train_neg_dir, name)
    shutil.copyfile(src,dst)
    k+=1
  elif k>=1500 and k<2500:
    src=os.path.join(original_dataset_dir, name)
    dst=os.path.join(validation_neg_dir, name)
    shutil.copyfile(src,dst)
    k+=1
  elif k>=2500 and k<2900:
    src=os.path.join(original_dataset_dir, name)
    dst=os.path.join(test_neg_dir, name)
    shutil.copyfile(src,dst)
    k+=1

k=0
for name in glob.glob('/content/ExtractedDataset/1_*'):
  name=name[26:]
  if k<1500:
    src=os.path.join(original_dataset_dir, name)
    dst=os.path.join(train_pos_dir, name)
    shutil.copyfile(src,dst)
    k+=1
  elif k>=1500 and k<2500:
    src=os.path.join(original_dataset_dir, name)
    dst=os.path.join(validation_pos_dir, name)
    shutil.copyfile(src,dst)
    k+=1
  elif k>=2500 and k<3000:
    src=os.path.join(original_dataset_dir, name)
    dst=os.path.join(test_pos_dir, name)
    shutil.copyfile(src,dst)
    k+=1

print('total training pos images:', len(os.listdir(train_pos_dir)))
print('total validation pos images:', len(os.listdir(validation_pos_dir)))
print('total test pos images:', len(os.listdir(test_pos_dir)))

print('total training neg images:', len(os.listdir(train_neg_dir)))
print('total validation neg images:', len(os.listdir(validation_neg_dir)))
print('total test neg images:', len(os.listdir(test_neg_dir)))




# Bibliothèques Deep Learning

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image


# Réseau CNN de base

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=((50, 50, 3))))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen= ImageDataGenerator(rescale=1./255)
test_datagen= ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(50,50),
                                                    batch_size=20, 
                                                    class_mode='binary')
validation_generator = train_datagen.flow_from_directory(validation_dir, 
                                                    target_size=(50,50),
                                                    batch_size=20, 
                                                    class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=100,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=50)

model.save('base_20_05.h5')

import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc = history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Réseau CNN avec Data Augmentation

Pas forcément une très bonne idée finalement, on perd la géométrie des grasps dans certains cas. Peut_être est-il pertinent de réduir le data augmentation à quelques transformations basiques.

datagen = ImageDataGenerator(
      rotation_range=90,
      horizontal_flip=True,
      fill_mode='constant',
    cval=0)

# This is module with image preprocessing utilities
from keras.preprocessing import image

fnames = [os.path.join(train_pos_dir, fname) for fname in os.listdir(train_pos_dir)]

# We pick one image to "augment"
img_path = fnames[200]

# Read the image and resize it
img = image.load_img(img_path, target_size=(50, 50))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(50, 50, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
      rotation_range=90,
      horizontal_flip=True,
      fill_mode='constant',
    cval=0)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(50, 50),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(50, 50),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

model.save('augmented_angle90_21_05_largeDataset.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

Ca semble plutot moins bien...

# Réseau Pré-entrainé avec VGG 16 / ImageNet

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(50,50,3))

## Features extraits puis réinjectés

conv_base.summary()

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
  features = np.zeros(shape=(sample_count, 4, 4, 512))
  labels = np.zeros(shape=(sample_count))
  generator = datagen.flow_from_directory(directory,
                                          target_size=(50, 50),
                                          batch_size=batch_size,
                                          class_mode='binary')
  i = 0
  for inputs_batch, labels_batch in generator:
    features_batch = conv_base.predict(inputs_batch)
    features[i * batch_size : (i + 1) * batch_size] = features_batch
    labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= sample_count:
      break
  return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

## Réseau dense en bout de VGG 16

Deuixième méthode: réseau dense mis au bout de conv_base. Nous permet d'utiliser le data augmentation

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

print('Nombre de poids entrainables',
       len(model.trainable_weights))
conv_base.trainable = False


print('Nombre de poids entrainables après avoir fixé conv_base '
      , len(model.trainable_weights))

"""train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')"""
train_datagen = ImageDataGenerator(rescale=1./255)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(50, 50),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(50, 50),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

Sauver le modèle sur Google Drive

model.save('vgg16_30_05_30epochs_largeDataset.h5')

!cp vgg16_21_05_30epochs_largeDataset.h5 /content/drive

Importer un modèle enregistrer déjà sauvé.

from keras.models import load_model
model=load_model('vgg16_21_05_30epochs_largeDataset.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Predictions et utilisation du réseau

## Importation d'images

Importer une image à tester. Dans certains cas, soulève une erreur sur firefox. Dans ce ca-là, utilisez chrome.

os.chdir('/content')

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

## Prediction

!pip install imutils

from sklearn import feature_extraction as sk
import cv2,imutils
from PIL.ImageDraw import Draw

def center_crop(img,output_size):
    crop_width=output_size[0]/2
    crop_height=output_size[1]/2
    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    img_out = img.crop(
    (
        half_the_width - crop_width,
        half_the_height - crop_height,
        half_the_width + crop_width,
        half_the_height + crop_height
    ))
    return img_out

def extract_patch_coord(img_array,patch_size,stride):
    """returns 2 2D arrays: one with all the patches in the image, 
    another with the same chape but filled with the coordinates of the patches in pixels.
    The coordinates correspond to the top left edge of the square patch in the order (horizontal_axis,vertical_axis).
    Requires sklearn.feature_extraction.image and numpy"""   
    
    patches_img=sk.image.extract_patches(img_array, patch_size, extraction_step=stride)
    t1,t2=np.shape(patches_img)[:2]
    coordinates=np.zeros((t1,t2,2))
    coordinates[:,:,0],coordinates[:,:,1]=np.meshgrid(np.arange(0,t2*stride,stride),np.arange(0,t1*stride,stride))
    size_array=np.ones(np.shape(patches_img[:2]))*patch_size[0]
    return patches_img, coordinates

def rotated_patch(patch,final_angle,stride):
  patches_rotated=[]
  for angle in np.arange(0, final_angle, stride):
    rotation = imutils.rotate_bound(patch, angle)
    #rotation = imutils.rotate(patch, angle)
    rotation=cv2.resize(rotation,(50,50))
    patches_rotated.append(rotation)
  return patches_rotated

plt.imshow(rot[4])

def find_best_grasp_no_angle(img_filename, patch_size,stride,threshold,crop=True,display=True,crop_size=(400,200)):
  img=Image.open(img_filename)
  if crop==True:
    img=center_crop(img,crop_size)
  img_array= np.asarray(img)[:,:,:3]
  img_array=img_array/255
  patches,coord = extract_patch_coord(img_array,patch_size,stride)
  L=len(patches)
  prediction_all_patches=[]
  potential_grasps=[]
  max_prediction=0
  max_index=0
  max_grasp_coord=0
  threshold=0.6
  for i in range(len(patches)):
    for j in range(len(patches[0])):
      x=patches[i][j]
      c=coord[i][j]
      resized_patch=np.resize(x,(1,)+patch_size)
      prediction=model.predict(resized_patch)
      prediction_all_patches.append(float(prediction))
      if float(prediction) > threshold:
        potential_grasps.append(x)
        if float(prediction)>max_prediction:
            max_prediction=float(prediction)
            max_index=(i,j)
            max_grasp_coord=c
  if  len(potential_grasps)==0:
    print('No grasp found.')
  else:
    print(np.shape(potential_grasps)[0], "potential grasps found.\nThreshold :", threshold)
    print("Max probability: ",max_prediction)
    print("Corresponding Index: ",max_index)
    print("Max coordinates (px): ",max_grasp_coord)
    if display==True:
      plt.imshow(patches[max_index[0]][max_index[1]][0])
      plt.show()
  return patches, potential_grasps, max_prediction, max_index, max_grasp_coord

def display_grasp_on_src_image(img_filename,max_grasp_coord,patch_size,crop=True):
  """Works only for vertical grasps (no patch rotation) for now."""
  img=Image.open(img_filename)
  if crop==True:
    img=center_crop(img,(400,200))
  x_base=max_grasp_coord[0]
  y_base=max_grasp_coord[1]
  img_rect=Draw(img)
  img_rect.rectangle([(x_base,y_base),(x_base+patch_size[0],y_base+patch_size[1])],outline='blue')
  plt.imshow(img)
  plt.show()
  return
  
  

display_grasp_on_src_image("/content/drive/11.jpg",(50,50),(50,50,3),crop=False)

def find_best_grasp_angle(img_filename, patch_size,stride,final_angle,angle_stride,threshold,crop=True,display=True):
  img=Image.open(img_filename)
  if crop==True:
    img=center_crop(img,(400,200))
  img_array= np.asarray(img)[:,:,:3]
  img_array=img_array/255
  patches,coord = extract_patch_coord(img_array,patch_size,stride)
  #rotated_patches=np.zeros((np.shape(patches)+(final_angle//angle_stride,)+patch_size))
  rotated_patches=np.zeros((np.shape(patches)[0],np.shape(patches)[1],6,50,50,3))
  for a in range(len(patches)):
    for b in range(len(patches[0])):
      #rotated_patches[a][b][:]=rotated_patch(patches[a][b][0],final_angle,angle_stride)  
      rotated_patches[a][b][:]=rotated_patch(patches[a][b][0],90,15)  
  prediction_all_patches=[]
  potential_grasps=[]
  max_prediction=0
  max_index=0
  max_grasp_coord=0
  max_angle=0
  for i in tqdm(range(len(rotated_patches))):
    for j in range(len(rotated_patches[0])):
      for k in range(len(rotated_patches[0][0])):
        x=rotated_patches[i][j][k]
        c=coord[i][j]
        angle=k
        resized_patch=np.resize(x,(1,)+patch_size)
        prediction=model.predict(resized_patch)
        prediction_all_patches.append(float(prediction))
        if float(prediction) > threshold:
          potential_grasps.append(x)
          if float(prediction)>max_prediction:
            max_prediction=float(prediction)
            max_index=(i,j)
            max_grasp_coord=c
            max_angle=k
  if  len(potential_grasps)==0:
    print('No grasp found.')
  else:
    print(np.shape(potential_grasps)[0], "potential grasps found.\nThreshold :", threshold)
    print("Max probability: ",max_prediction)
    print("Corresponding Index: ",max_index, max_angle)
    print("Max coordinates (px): ",max_grasp_coord)
    if display==True:
      plt.imshow(rotated_patches[max_index[0]][max_index[1]][max_angle])
      plt.show()
  return rotated_patches, potential_grasps, max_prediction, max_index, max_grasp_coord, max_angle

!cp /content/drive/ColabFiles/'content/drive/11.jpg' /content

img=Image.open("/content/drive/11.jpg")
plt.imshow(img)
plt.show()

rotated_patches, potential_grasps, max_prediction, max_index, max_grasp_coord=find_best_grasp_no_angle("/content/drive/11.jpg",(50,50,3),20,0.6,crop=False, display=True)

rotated_patches, potential_grasps, max_prediction, max_index, max_grasp_coord,max_angle=find_best_grasp_angle("/content/drive/11.jpg",(50,50,3),20,90,15,0.6,crop=False, display=True)



img33=Image.open("24.jpg")
plt.imshow(img33)
plt.show()

rotated_patches, potential_grasps, max_prediction, max_index, max_grasp_coord,max_angle=find_best_grasp_angle("24.jpg",(50,50,3),20,90,15,0.6,crop=False, display=True)

img=Image.open("24 (1).jpg")
plt.imshow(img)

rotated_patches, potential_grasps, max_prediction, max_index, max_grasp_coord,max_angle=find_best_grasp_angle("24 (1).jpg",(50,50,3),20,90,15,0.6,crop=False, display=True)

# Importer / Sauver des fichiers depuis Google Drive

Pour ne pas réexecuter l'ensemble du code à chaque fois, on peut sauver le modèle sur un drive google, et simplement le réuploader quand il faut.
On considère avec le programme suivant le drive comme un dossier dans la machine virtuelle.

# Install a Drive FUSE wrapper.
# https://github.com/astrada/google-drive-ocamlfuse
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse



# Generate auth tokens for Colab
from google.colab import auth
auth.authenticate_user()


# Generate creds for the Drive FUSE library.
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}


# Create a directory and mount Google Drive using that directory.
!mkdir -p drive
!google-drive-ocamlfuse drive

!ls /content/drive

!cp /content/drive/ColabFiles/vgg16_21_05_30epochs_largeDataset.h5 /content
