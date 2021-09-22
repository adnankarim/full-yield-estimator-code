
import albumentations as album
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import warnings
from glob import glob
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from natsort import natsorted
from PIL import Image,ImageOps
from itertools import product




def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst



def tile( dir_in, dir_out, d,start_no):
    ext='.jpg'
    img = Image.open(dir_in)
    w, h = img.size
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    number=start_no
    con_num=''
    for i, j in grid:
        number=number+1
        if number<10:
          con_num='0000'+str(number)
        elif number <100:
          con_num='000'+str(number)
        elif number <1000:
          con_num='00'+str(number)
        elif number <10000:
          con_num='0'+str(number)
        elif number <100000:
          con_num=''+str(number)

        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{con_num}{ext}')
        img.crop(box).save(out)

  #deeplab

  

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


# In[ ]:


class BuildingsDataset(torch.utils.data.Dataset):

    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            class_rgb_values=None, 
            preprocessing=None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in natsorted(os.listdir(images_dir))]

        self.class_rgb_values = class_rgb_values
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
    
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        return image
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)


# In[ ]:





def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=200, min_width=200, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def crop_image(image, target_image_dims=[200,200,3]):
       
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
MODEL_PATH='best_model.pth'

if os.path.exists(MODEL_PATH):
    best_model = torch.load(MODEL_PATH, map_location=DEVICE)
    print('Loaded DeepLabV3 model from this run.')
    best_model.eval()
    
def get_prediction (unique_id):
    #model path
    # input image
    input_image='1.jpg'
    #folders
    INPUT_IMAGE_DIR='app/'+unique_id+'/INPUT_IMAGE'
    INPUT_IMAGE_PATCHES_DIR='app/'+unique_id+'/INPUT_IMAGE_PATCHES'
    OUTPUT_IMAGE_PATCHES_PREDS='app/'+unique_id+'/OUTPUT_IMAGE_PATCHES_PREDS'
    OUTPUT_HORIZONTAL_PATCHES_DIR='app/'+unique_id+'/OUTPUT_HORIZONTAL_PATCHES'
    OUTPUT_IMAGE_DIR='app/'+unique_id+'/OUTPUT_IMAGE'
  
    PATHS=[INPUT_IMAGE_DIR,
    INPUT_IMAGE_PATCHES_DIR,
    OUTPUT_HORIZONTAL_PATCHES_DIR,
    OUTPUT_IMAGE_PATCHES_PREDS,
        OUTPUT_IMAGE_DIR]

    #creating paths
    for i in PATHS:
        if not os.path.exists(i):
            os.makedirs(i)
    
    sample_preds_folder = OUTPUT_IMAGE_PATCHES_PREDS
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)


    # input config
    Image_sizes_width=(4000,3000)
    horizontal_patches=Image_sizes_width[0]//200
    vertical_patches=Image_sizes_width[1]//200
    total_input_patches=horizontal_patches*vertical_patches

  


#deinition

    passed=0
    for i in range(1):
        tile(INPUT_IMAGE_DIR+'/' +input_image,INPUT_IMAGE_PATCHES_DIR,200,passed)
        passed=passed+300
    # In[ ]:

    LABELS_DIR='./label_class_dict.csv'
    DATA_DIR = './'

    x_test_dir = os.path.join(DATA_DIR, INPUT_IMAGE_PATCHES_DIR)
    class_dict = pd.read_csv(LABELS_DIR)
    # Get class names
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r','g','b']].values.tolist()

    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = ['background', 'orange']

    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]


    # ### Helper functions for viz. & one-hot encoding/decoding

    # In[ ]:


    ENCODER = 'efficientnet-b2'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = class_names
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation


    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    


    test_dataset = BuildingsDataset(
    x_test_dir, 
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)


    for idx in range(total_input_patches):

        image = test_dataset[idx]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # Predict test image
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask,(1,2,0))
        # Get prediction channel corresponding to building
        pred_building_heatmap = pred_mask[:,:,select_classes.index('orange')]
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))
        # Convert gt_mask from `CHW` format to `HWC` format

        cv2.imwrite(os.path.join(OUTPUT_IMAGE_PATCHES_PREDS, f"{idx}.jpg"), pred_mask)



    one_file = natsorted(glob(OUTPUT_IMAGE_PATCHES_PREDS+'/*'))
    border=2
    color='red'
    # horz=np.array()
    i=0
    z=0
    l=0
    while i <total_input_patches:
        j=0
        k=j+(horizontal_patches-1)
        im1=Image.open(one_file[l])
        list=[]
        im1 = ImageOps.expand(im1)
        while j<k:
            list.append(i+j)
            im2=Image.open(one_file[i+j+1])
            im2 = ImageOps.expand(im2)
            im1=get_concat_h(im1, im2)
            if j%(k-1)==0 and not ( j== 0 ) :
                im1.save(OUTPUT_HORIZONTAL_PATCHES_DIR+'/'+str(z)+'.jpg')
                # im1.convert('RGB').save(OUTPUT_HORIZONTAL_PATCHES_DIR+'/'+str(z)+'.jpg')
            j=j+1
        l=l+horizontal_patches  
        i=i+horizontal_patches
        z=z+1

        


    # In[125]:


    one_file = natsorted(glob(OUTPUT_HORIZONTAL_PATCHES_DIR+'/*'))
    # list=Tcl().call('lsort', '-dict', one_file)
    # # vertical
    im1=Image.open(one_file[0])
    i=1
    while i<vertical_patches:
        im2=Image.open(one_file[i])
        im3=get_concat_v(im1, im2)
        # im3.convert('RGB').save(OUTPUT_IMAGE_DIR+'/new.jpg')
        im3.save(OUTPUT_IMAGE_DIR+'/new.jpg')
        im1=Image.open(OUTPUT_IMAGE_DIR+'/new.jpg')
        i=i+1


    # # In[ ]:





    # In[126]:


    img = cv2.imread(OUTPUT_IMAGE_DIR+'/new.jpg',0)
    img2= cv2.imread(INPUT_IMAGE_DIR+'/'+input_image)
    # img2=np.flip(img2, axis=-1) 
    # plt.imshow(img2)

    #convert the image to grayscale
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #blur image to reduce the noise in the image while thresholding
    blur = cv2.blur(img, (45,45))
    #Apply thresholding to the image
    ret, thresh = cv2.threshold(blur, 66, 100, cv2.THRESH_OTSU)
    #find the contours in the image
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #draw the obtained contour lines(or the set of coordinates forming a line) on the original image
    cv2.drawContours(img2, contours, -1, (0,255,0), 20)
    #show the image
    # img2=np.flip(img2, axis=-1) 
    cv2.imwrite(OUTPUT_IMAGE_DIR+'/pred.jpg', img2) 

    # In[127]:

    return len(contours)


# In[ ]:




