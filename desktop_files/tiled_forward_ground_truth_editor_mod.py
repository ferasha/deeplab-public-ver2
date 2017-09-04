import time
import caffe 
import numpy as np
import os
from scipy import io as sio
from matplotlib import pyplot as plt
import h5py
from scipy import ndimage
from scipy import signal


Vertical = True

def show_image(image, block=True):      
    plt.imshow(image, cmap = plt.get_cmap('gray'))
    plt.show(block) 
def show_weights(image, block=True, colorbar=True):      
    ax = plt.subplot('111')
    
    cax = plt.imshow(image, cmap = plt.get_cmap('jet'),vmin = np.min(image), vmax = np.max(image))
    ax.set_title("Weight Map")
    
    levels = np.linspace(np.min(image),np.max(image),num=8)
    
    plt.colorbar(ticks=levels, format='%.3f')
        
    plt.show(block) 
def vis_square(data, fname=''):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
  #  print data.shape
    plt.imshow(data.squeeze(), cmap = plt.get_cmap('gray'))
    plt.axis('off')
    if( fname != '' ):
        plt.savefig(fname, dpi=300)
        return
    plt.show(True)

def compute_dilation_iteration( label ):
    h, w = label.shape
    area = float( np.sum( label > 0 ) )
    # Devide by 2 as it is up and down
    it   = int( np.ceil( (area/float(w)) / 2.0 ) )
    return it   
    
def gaussian_around( label, gaussian ):
    res = np.copy( label )
    
    for g in gaussian:
        if( g > 1.e-5 ):
            mask = ndimage.morphology.binary_dilation(res>0, iterations=1) * g
            res  = np.maximum( res, mask )
    
        else:
            break
    return res

def generate_weights_class_frequency(label, l_type=''):
    h, w = label.shape
    num_c = len(np.unique(label))
    weights = np.zeros((h,w), dtype='float32')
    
    f_0 = 1.0 / (float(np.sum(label==0))/(w*h))
    if( np.sum(label==1) != 0):
        f_1 = 1.0 / (float(np.sum(label==1))/(w*h))
    else:
        f_1 = 1.0
  #  show_image(label)
    if( l_type != 'area' ):
        f_1 = 1.0 / (float(np.sum(label==1)+np.sum(label==3))/(w*h))
        f_2 = 1.0 / (float(np.sum(label==2)+np.sum(label==3))/(w*h))
        weights[label==2] = f_2
        
    weights[label==0] = f_0
    weights[label==1] = f_1
    
    if( np.sum(label==3) > 0 ):
        f_3 = 1.0 / (float(np.sum(label==3))/(w*h))
        weights[label==3] = max([f_1, f_2])
    
    if( np.min(weights) != 0 ):
        weights = weights/np.min(weights)
    #show_weights(weights)
    return weights

def generate_weights_spatial(label_o, sig, l_type='', label_area_o = []):
    
    label      = np.copy(label_o)
    label_area = np.copy(label_area_o)
    
    h, w    = label.shape
    num_c   = len(np.unique(label))
    weights = np.zeros((h,w,num_c-1), dtype='float32')
    
    weights.fill(1.e-20)
    
    g = signal.gaussian(h*2, sig)
    
    if( l_type == 'extend_area' or  l_type == 'drusen'):
        
        it = compute_dilation_iteration( label_area )
        label_area = ndimage.morphology.binary_dilation(label_area, iterations=it)
        label.fill(0)
        for j in range(w):
            col = label_area[:, j]
            loc = np.where(col > 0)[0]
            if( loc != [] ):
                if( np.min(loc) == np.max(loc) ):
                    label[np.max(loc), j] = 3
                else:
                    label[np.min(loc), j] = 2
                    label[np.max(loc), j] = 1
 
    for c in range(num_c):
        if( c == 0 ): # Skip the background
            continue
        i, j = np.where(label == c)
        ind = 0
        for k in i:
            weights[:, j[ind], c-1] = g[h-k:h+h-k]
            ind += 1
    if( weights.shape[2] != 0 ):    
        weights = np.max(weights, axis=2)
    else:
        weights = np.zeros((h,w), dtype='float32')
    if(  l_type == 'area' or l_type == 'extend_area'):
        weights[label_area==1] = np.max(weights)
    #show_weights(weights)
    if( l_type == 'drusen' ):
        weights = gaussian_around( label_area, g[h:])
        weights[label_area==1] = np.max(weights)
        return weights, label_area
    #show_weights(weights)
    if( l_type == 'extend_area' ):
        return weights, label_area
    return weights

def no_drusen(labels, threshold):
    ll = labels[:,:,0,0]
    h, w = ll.shape
    n = 0
    avg = 0
    #show_image(ll)
    for i in range(w):
        col = ll[:, i]
        j1 = np.where(col==1)
        j2 = np.where(col==2)
        
        if( len(j1[0]) != 0 and len(j2[0]) != 0 ):
            avg += abs(j1[0][0]-j2[0][0])
            n += 1
    if( n == 0 ):
        return True
    avg /= n
       
#    print avg
    if( avg < threshold ):
        return True
    return False
        
def sanity_check(labels):
    ll = labels[:,:,0,0]    
    h, w = ll.shape
    num_l1 = np.sum(ll==1)
    num_l2 = np.sum(ll==2)
    # Labels must be rather full
    if((num_l1/float(w)) < 0.9 or  (num_l2/float(w)) < 0.9):
        return False
    return True
    
def find_area_between_seg_lines(label):
    h, w = label.shape
    label_area = np.copy(label)
  
    for j in range(w):
        col = label[:, j]
        l_1 = np.where( col == 1 )
        l_2 = np.where( col == 2 )
        if(len(l_1[0]) != 0 and len(l_2[0]) != 0 ):

            label_area[l_1[0][0]:l_2[0][0], j] = 1
            label_area[l_2[0][0]:l_1[0][0], j] = 1
            
    # Replace all the labels with 1
    label_area[label_area > 0] = 1
   
    return label_area
            
def mycaffe_tiled_forward5( data, labels, opts=dict() ):
    
    # In the end return the way this image is treated. Is it a Test image, 
    # or a Train image, or neither
    image_category = None
    test_category  = None
    #
    #  compute input and output sizes (for v-shaped 4-resolutions network)
    #
    #script_dir = opts['scriptdir']

    data_path = opts['datapath']
    save_path = opts['savepath']
  #  print data.shape
    if( not Vertical ):
        data = np.transpose(data, (1,0,2,3))
        labels = np.transpose(labels,(1,0,2,3))
    
    a = np.asarray([data.shape[0], np.ceil(data.shape[1])/opts['nTiles']])
    b = opts['padOutput']
    c = opts['downsampleFactor']
    d4a_size = np.ceil((a - b)/c)
    input_size = opts['downsampleFactor']*d4a_size + opts['padInput']
    output_size = opts['downsampleFactor']*d4a_size + opts['padOutput']

     
    #
    #  create padded volume mit maximal border
    #

    
    border = np.round(input_size-output_size)/2
   
    paddedFullVolume = np.zeros((int(data.shape[0] + 2*border[0]),\
                                 int(data.shape[1] + 2*border[1]),\
                                 int(data.shape[2]),\
                                 int(data.shape[3])),dtype='float32',order='F')
                                 
    labelpaddedFullVolume = np.zeros((int(data.shape[0] + 2*border[0]),\
                                 int(data.shape[1] + 2*border[1]),\
                                 int(data.shape[2]),\
                                 int(data.shape[3])),dtype='float32',order='F')
    
    paddedFullVolume[int(border[0]):int(border[0]+data.shape[0]),\
                     int(border[1]):int(border[1]+data.shape[1]),\
                     :, : ] = data
                     
    labelpaddedFullVolume[int(border[0]):int(border[0]+data.shape[0]),\
                     int(border[1]):int(border[1]+data.shape[1]),\
                     :, : ] = labels
                     
                     
                     
   # print "paddedVol:", paddedFullVolume.shape               

    if( opts['padding'] == 'mirror' ):
        # Mirror boundaries
        xpad  = int(border[0])
        xfrom = int(border[0] + 1)
        xto   = int(border[0] + data.shape[0])
        
        paddedFullVolume[0:xfrom-1,:,:,:] =\
            np.flipud(paddedFullVolume[xfrom:xfrom+xpad,:,:,:])
        paddedFullVolume[xto:,:,:,:] =\
            np.flipud(paddedFullVolume[xto-xpad-1:xto-1,:,:,:])
        
        
        labelpaddedFullVolume[0:xfrom-1,:,:,:] =\
            np.flipud(labelpaddedFullVolume[xfrom:xfrom+xpad,:,:,:])
        labelpaddedFullVolume[xto:,:,:,:] =\
            np.flipud(labelpaddedFullVolume[xto-xpad-1:xto-1,:,:,:])        
        
        ypad  = int(border[1])
        yfrom = int(border[1]+1)
        yto   = int(border[1]+data.shape[1])
        paddedFullVolume[:, 0:yfrom-1,:,:] =\
            np.fliplr(paddedFullVolume[ :, yfrom:yfrom+ypad,:,:])
        paddedFullVolume[:, yto:,:,:] =\
            np.fliplr(paddedFullVolume[ :, yto-ypad-1:yto-1,:,:])
            
        labelpaddedFullVolume[:, 0:yfrom-1,:,:] =\
            np.fliplr(labelpaddedFullVolume[ :, yfrom:yfrom+ypad,:,:])
        labelpaddedFullVolume[:, yto:,:,:] =\
            np.fliplr(labelpaddedFullVolume[ :, yto-ypad-1:yto-1,:,:])
 

 
    
   #  do the classification (tiled)
   #  average over flipped images
    per_data = (3,2,1,0)
    per_scor = (2,1,0)
    scores = []
    for num in range(data.shape[3]):
     
      tic = time.time()
      validReg = [0,0]
      ln = 0
      el = np.unique(labels)
      el.sort()
      
      for l in el:
          labels[labels==l] = ln
          labelpaddedFullVolume[labelpaddedFullVolume == l ] = ln
          ln+=1
    
  
 #     if( not no_drusen(np.transpose(labels,(1,0,2,3)), 5) ):
 #         continue
      # MAke the training set with the prob. 0.9 with the images with
    # drusen                
      rand_num     = np.random.rand(1)[0]   
      act_opposit  = rand_num < opts['rand_t']
      no_drusen_flag = no_drusen(np.transpose(labels,(1,0,2,3)), 5)

     
      image_tiles = np.zeros((1,1,int(input_size[1]), int(input_size[0]), opts['nTiles']))
      label_tiles = np.zeros((1,1,int(input_size[1]), int(input_size[0]), opts['nTiles']))
      weght_tiles = np.zeros((1,1,int(input_size[1]), int(input_size[0]), opts['nTiles']))
      
      # crop input data
      for yi in range(opts['nTiles']):
        
        paddedInputSlice = np.zeros((int(input_size[0]), int(input_size[1]),\
                            data.shape[2],1),dtype='float32',order='C')
        validReg[0] = int(min(input_size[0], paddedFullVolume.shape[0]))
        validReg[1] = int(min(input_size[1], paddedFullVolume.shape[1] - yi*output_size[1]))
        
        paddedInputSlice[0:validReg[0],0:validReg[1],:,0] =\
           paddedFullVolume[0:validReg[0], int(yi*output_size[1]):int(yi*output_size[1]+validReg[1]), :, num]
          
        labelpaddedInputSlice = np.zeros((int(input_size[0]), int(input_size[1]),\
                            data.shape[2],1),dtype='float32',order='C')
           
        labelpaddedInputSlice[0:validReg[0],0:validReg[1],:,0] =\
           labelpaddedFullVolume[0:validReg[0], int(yi*output_size[1]):int(yi*output_size[1]+validReg[1]), :, num]
          
        data_train=(np.transpose(paddedInputSlice,per_data))
        label_train = (np.transpose(labelpaddedInputSlice,per_data))
        
        #data_train = np.transpose(data,per_data)
        #label_train = np.transpose(labels,per_data)
        
        img_train = data_train[0,0,:,:]
        #img_train = data[:,:,0,0]
        validReg[0] = int(min(output_size[0], data.shape[0]))
        
        validReg[1] = int(min(output_size[1], data.shape[1] - yi*output_size[1]))
         
        labels_train = np.zeros((output_size))
        
        labels_train[0:int(validReg[0]), 0:int(yi*output_size[1]+validReg[1])-int(yi*output_size[1])] =\
            labels[0:int(output_size[0]), int(yi*output_size[1]):int(yi*output_size[1]+validReg[1]),0,num] 
        
        diff = np.abs(output_size - validReg)
        labels_train[int(validReg[0]):,:] =\
            np.flipud(labels_train[int(validReg[0]-diff[0]):int(validReg[0]),:] )
            
        labels_train[:,int(validReg[1]):] =\
            np.fliplr(labels_train[:,int(validReg[1]-diff[1]):int(validReg[1])] ) 
           
        
        labels_train = labels_train.transpose((1,0))
        labels_train = label_train[0,0,:,:]
        
        #labels_train = np.transpose(labels[:,:,0,0],(1,0))
          #  labels_train = ndimage.binary_dilation(labels_train, iterations=10)
        #        print labels_train.shape
        
        num_b = 1
        num_c = 1 # number of channels
        height, width = img_train.shape
           
        
        total_size =  height * width * num_b * num_c
        img_train  = img_train.reshape((num_b, num_c, height, width))
          
        img_train = img_train.astype('float32')
           
        l_height, l_width = labels_train.shape
        label = np.zeros((l_height, l_width), dtype='uint8')
        
        label[:,:] = labels_train
           # print img_train.shape, label.shape
        
        img_train = np.transpose(img_train, (0,1,2,3))
        if( not Vertical ):
           
            label = np.transpose(label, (1,0))
            
        l_height, l_width = label.shape
        weight_type = opts['lType']
        
        if(weight_type == 'extend_area' or weight_type == 'area' ) :
            label_area = find_area_between_seg_lines(label)
        else:
            label_area = label
        
        # Prev value = 10
        # Prev , w0=10, sig =7
        w0 = opts['alpha']
        sig = opts['betta']
        
        
        
        res = generate_weights_spatial(label, sig, weight_type, label_area) 
        
        if( weight_type == 'extend_area' or weight_type == 'drusen' ):        
            weight = generate_weights_class_frequency(res[1], 'area') 
            weight = weight + w0 * (res[0]+ (1-np.min(weight)))
        
        elif(weight_type == 'area'):
            weight = generate_weights_class_frequency(label_area, 'area') 
            weight = weight + w0 * (res+ (1-np.min(weight)))
        else:
            
            weight = generate_weights_class_frequency(label_area, 'layer') 
           # print np.min(weight), np.min(res), np.max(res)
          #  show_image(res+ (1-np.min(weight)))
          #  show_image(weight)
            weight = weight + w0 * (res+ (1-np.min(weight)))
           # just_drusen = True
           # if( just_drusen ):
           #     weight = generate_weights_class_frequency(res[1], 'area') 
        
           # show_weights(weight)
        #label[np.where(label > 0)] = 1
        weight = weight.reshape((num_b, 1, l_height, l_width))
        label = label.reshape((num_b, num_c, l_height, l_width))
        
        label[0,0,:,:] = label_area
           # img_id = yi * opts['n_imgs'] + opts['id']
        #show_weights(weight[0,0,:,:])
        #show_image(label[0,0,:,:])
        #show_image(img_train[0,0,:,:])
        image_tiles[:,:,:,:,yi] = img_train
        label_tiles[:,:,:,:,yi] = label
        weght_tiles[:,:,:,:,yi] = weight
        if( np.sum(label) == 0.0 ):
            #show_image(label[0,0,:,:])
            print "The tile is fully with 0 label."
            

            #print img_train.shape, label.shape, weight.shape
            #print "------------>", img_id, save_path_split[-1], save_path + '/'+ save_path_split[-1]+ '-' +str(img_id) + '-data.h5'
            #show_weights(weight[0,0,:,:])
            #show_image(label[0,0,:,:])
            #show_image(img_train[0,0,:,:])
        
                    
                   
            
              

    return image_tiles, label_tiles, weght_tiles
    
    
def test_weights():
    w0 = 30
    sig = 15
     
    filename = '/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/4-18/MOD011/220211_19/11-44339EB0-BinSeg.tif'
    filename_area = filename.replace('-BinSeg.tif', '-AreaSeg.tif')
    image = read_image(filename)
    label = encode_label(image)

    label_area = read_image(filename_area)

    res = generate_weights_spatial(label, sig, 'area', label_area) 
    weight = generate_weights_class_frequency(label_area, 'area') 
    weight = weight + w0 * (res+ (1-np.min(weight)))
        
    show_weights(weight)
    l_height, l_width = label.shape
    print 'height and weight', l_height, l_width
    weight = weight.reshape((1, 1, l_height, l_width))    
    
    
    
test_weights() 
    
    
    

