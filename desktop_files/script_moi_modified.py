from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import imsave
from scipy import ndimage
import os.path
from scipy import signal
import scipy.io as sio
import cv2

np.set_printoptions(threshold='nan')

def read_image(filename):
    img = io.imread(filename)
    dim = img.shape
    return img


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
    white = np.sum(label_area == 1)
    black = np.sum(label_area == 0)
    all_ =  white+black
#    print white, black, all_, black*1.0/white, all_*1.0/black, all_*1.0/white
#    print label_area.shape
    h, w  = label_area.shape
#    print h *w
#    el = np.unique(label_area)
#    print el
    return label_area
    
def find_seg_lines(label):
    h, w = label.shape
    label_area = np.copy(label)
    
    label_area[label_area==2] = 1
  
    el = np.unique(label_area)
    print el
    return label_area

def encode_label(image_lbl):
    labels = np.copy(image_lbl)
    el = np.unique(image_lbl)
    el.sort()
    ln = 0      
    for l in el:
        labels[labels==l] = ln
        ln+=1
    return labels

def keep_largest_component( img ):
    
    label, num_label = ndimage.label(img)
#    print 'label', label
    print 'num_label', num_label
    size = np.bincount(label.ravel())
    print 'size',size
    largest_comp = size[1:].argmax() + 1
    print 'largest_comp', largest_comp
    result = label == largest_comp
    
    return result
    
    
def keep_largest_component_mod( img ):
    
    label, num_label = ndimage.label(img)
    sizes = np.bincount(label.ravel())
    largest_comp = sizes[1:].argmax() + 1
        
    x_len_l = []
    for l in range(sizes.shape[0]):
        if l == 0:
            continue
        y,x = np.where(label == l)
        x_len_l.append(max(x) - min(x))
    
    a = np.argmax(x_len_l)    
    a = a+1

    if a != largest_comp:
        print 'largest_comp different from largest x length'
    result = label == a
      
    return result

def extract_seg_layers( label ):
    h, w = label.shape
    res  = np.copy( label.astype('float') )
    res.fill(0.)
    prev_max = -1
    prev_min = -1
    for j in range(w):
        col = label[:, j]
        loc = np.where(col > 0)[0]
        if( loc != [] ):
            
            if( prev_max == -1 ):
                prev_max = np.max(loc)
            if( prev_min == -1 ):
                prev_min = np.min(loc)
                
            if( np.min(loc) == np.max(loc) ):
                res[prev_max:np.max(loc)+1, j] = 3.
                res[np.max(loc):prev_max+1, j] = 3.
                
            else:
                res[np.min(loc):prev_min+1, j] = 2.
                res[prev_min:np.min(loc)+1, j] = 2.
                res[prev_max:np.max(loc)+1, j] = 1.
                res[np.max(loc):prev_max+1, j] = 1.
                
            prev_max = np.max(loc)
            prev_min = np.min(loc)
    
    return res

def main1():
 #   image = read_image('/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/1-3/MOD001/001_101115_15/15-631C3070-BinSeg.tif')
    image = read_image('/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/34-45/MOD038/210513_145/28-25512CF0-BinSeg.tif')
  
    #plt.imshow(image, cmap = plt.get_cmap('gray'))
    #plt.show(True)

    encoded = encode_label(image)
#    image_area = find_area_between_seg_lines(encoded)
    image_area = find_seg_lines(encoded)
    imsave('/home/rasha/Desktop/test_image.tif', image_area)
    
    image1 = read_image('/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/34-45/MOD038/210513_145/28-25512CF0-Seg.tif')
    plt.imshow(image1, cmap = plt.get_cmap('gray'))
    plt.show(True)

 #   image = read_image('/home/rasha/Desktop/test_area/15-631C3070-Input.tif')
 #   plt.imshow(image, cmap = plt.get_cmap('gray'))
 #   plt.show(True)


def convert_to_layers(filename):
  #  net_pred = read_image('/home/rasha/Desktop/test_area/network_prediction/15-631C3070-BinSeg.png')
  #  filename = '/home/rasha/work/deeplab-public-ver2/drusen_area/features/RESNET-50/val/fc1/post_none/results/drusen_area/Segmentation/comp6_val_cls/4-18/MOD006/030214_19/1-447AD320-pred.png'
    filename_save = filename.replace('-pred.png', '-comp_layers.png')
    
#    if os.path.exists(filename_save):
#        print 'file exists ' + filename_save
#        return
    
    net_pred = read_image(filename)
#    plt.imshow(net_pred, cmap = plt.get_cmap('gray'))
#    plt.show(True)

    lc = keep_largest_component_mod(net_pred)
    #plt.imshow(lc, cmap = plt.get_cmap('gray'))
    #plt.show(True)

    seg_layers = extract_seg_layers(lc)
#    imsave('/home/rasha/Desktop/test_image.tif', seg_layers)
    imsave(filename_save, seg_layers)
#    plt.imshow(seg_layers, cmap = plt.get_cmap('gray'))
#    plt.show(True)


def create_network_list_files(input_file, output_file):
    i=0
    with open(input_file,'r') as f_read:
        with open(output_file,'w') as f_write:
            for line in f_read:
                if '-1-data.h5' in line:
                    continue
                i+=1
                rep = line.replace('HDF5-LineSeg-Area','Input-With-Label')
                rep = rep.replace('gorgi', 'rasha')
                rep = rep.replace('-0-data.h5\n','.tif')
    #            print rep   # should write this
                        
                area_seg = rep.replace('-Input', '-AreaSeg')

                if os.path.exists(area_seg):
                    print 'file exists ' + area_seg
                    continue
           
                if i % 100 == 0:
                    print i, area_seg    # and this       
                    


                bin_seg = rep.replace('-Input', '-BinSeg')
    #            print bin_seg
                
                bin_seg_img = read_image(bin_seg)
    #            plt.imshow(bin_seg_img, cmap = plt.get_cmap('gray'))
    #            plt.show(True)
    
                encoded = encode_label(bin_seg_img)
                area_seg_img = find_area_between_seg_lines(encoded)
    #            plt.imshow(area_seg_img, cmap = plt.get_cmap('gray')) 
    #            plt.show(True)
    
                imsave(area_seg, area_seg_img)
    
                f_write.write(rep + ' ' + area_seg + '\n')


def getIds(input_file, output_file):
    i=0
    with open(input_file,'r') as f_read:
        with open(output_file,'w') as f_write:
            for line in f_read:
                if '-1-data.h5' in line:
                    continue
                i+=1
                index1 = line.find('HDF5-LineSeg-Area')  + len('HDF5-LineSeg-Area') + 1           
                index2 = line.find('-Input-0-data.h5')
                unique = line[index1 : index2]
#                print index1, index2, unique
                comp = unique.split('/')
#                print comp
                new_id = comp[0] + '__' + comp[1] + '__' + comp[2] + '__' + comp[3] 
                print new_id
                f_write.write(new_id + '\n')
 

old_input_file = '/home/rasha/Desktop/sample_train_list_all_training.txt'
old_output_file = '/home/rasha/Desktop/sample_train_list_mod_all_training.txt'


old_input_file = '/home/rasha/Desktop/sample_train_list.txt'
old_output_file = '/home/rasha/Desktop/sample_train_list_mod_small.txt'

def create_list_files():
    main_folder =  '/home/rasha/Desktop/folds_list/'
    files_header = ['Fold01/test_files', 'Fold02/train_files', 'Fold02/test_files', 'Fold03/train_files', 'Fold03/test_files', 'Fold04/train_files', 'Fold04/test_files', 'Fold05/train_files', 'Fold05/test_files']

    for f in files_header:
        print "creating list for " + f
        filename_input = main_folder + f + '.txt'
        filename_output = filename_input.replace('_files', '_files_mod')

        create_network_list_files(filename_input, filename_output)

def create_test_id_files():
    main_folder =  '/home/rasha/Desktop/folds_list/'
    files_header = ['Fold01/test_files', 'Fold02/test_files', 'Fold03/test_files', 'Fold04/test_files', 'Fold05/test_files']

    for f in files_header:
        print "creating id file for " + f
        filename_input = main_folder + f + '.txt'
        filename_output = filename_input.replace('_files', '_files_id')

        getIds(filename_input, filename_output)

#create_test_id_files()

def convert_all_to_layers():
#    area_pred_folder = '/home/rasha/work/deeplab-public-ver2/drusen_area/features/RESNET-50/val/fc1/post_none/results/drusen_area/Segmentation/comp6_val_cls'
    area_pred_folder = '/home/rasha/work/deeplab-public-ver2/drusen_area/features/RESNET-50/val/small/snapshot_/matlab_images'
 #   area_pred_folder = '/home/rasha/work/deeplab-public-ver2/drusen_area/features/RESNET-50/val/small/snapshot_/post_densecrf'
    i = 0
    for d1 in os.listdir(area_pred_folder):
        for d2 in os.listdir(area_pred_folder + '/' + d1):
            for d3 in os.listdir(area_pred_folder + '/' + d1 + '/' + d2):
                fix = os.listdir(area_pred_folder + '/' + d1 + '/' + d2 + '/' + d3)
                for f in fix:
                    if '-pred.png' not in f:
                        continue
                    complete_path = area_pred_folder + '/' + d1 + '/' + d2 + '/' + d3 + '/' + f
                    convert_to_layers(complete_path)
                    print i, complete_path
                    i+=1
      
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
      
      

def generate_weights(filename):
    w0 = 30
    sig = 15
     
#    filename = '/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/4-18/MOD011/220211_19/11-44339EB0-BinSeg.tif'
    filename = filename.replace('-Input.tif', '-BinSeg.tif')
    filename_area = filename.replace('-BinSeg.tif', '-AreaSeg.tif')
#    filename_weights = filename.replace('-BinSeg.tif', '-Weights.png')
    filename_weights = filename.replace('-BinSeg.tif', '-Weights.mat')
    
    
    if os.path.exists(filename_weights):
        print 'file exists ' + filename_weights
        return filename_weights
                
    image = read_image(filename)
    label = encode_label(image)

    label_area = read_image(filename_area)
    
    res = generate_weights_spatial(label, sig, 'area', label_area) 
    weight = generate_weights_class_frequency(label_area, 'area') 
    weight = weight + w0 * (res+ (1-np.min(weight)))
    
    sio.savemat(filename_weights, {"data":weight})
 
#    unique_w = np.unique(weight)
#    print 'unique_w', unique_w
#    print 'length', len(unique_w)
 
#    unique_w_round = np.unique(np.round(unique_w))
#    print 'unique_w_round', unique_w_round
#    print 'length', len(unique_w_round)
#    weight_img = io.imread(filename_weights)
#    ind_w = np.where(label_area ==1)
#    print weight_img[ind_w]     
  
 #   imsave(filename_weights, weight)
    
#    weights_read = read_image(filename_weights)
#    unique_read = np.unique(weights_read)
#    print 'unique_read', unique_read, ' length', len(unique_read)
    
    return filename_weights
    
#    unique_w = np.unique(weight)
#    print unique_w
#   
#    vmin = np.min(weight)
#    vmax = np.max(weight)
#
#    print 'vmin',vmin,'vmax',vmax    
#
##    w_normalized = 255*(weight - vmin)/(vmax-vmin)        
##    unique_w_n = np.unique(w_normalized)
##    print unique_w_n
#
#    w_height, w_width = weight.shape
#    print 'weight height and width', w_height, w_width
#
##    cv2.imshow('cv weight', weight)
##    cv2.waitKey(0)
#            
#    weights_disk = read_image(filename_weights)
#    unique_w_disk = np.unique(weights_disk)
#    print unique_w_disk
#    vmin_disk = np.min(weights_disk)
#    vmax_disk = np.max(weights_disk)
#    print 'vmin_disk',vmin_disk,'vmax_disk',vmax_disk   
#    
#    show_weights(weight)
#    l_height, l_width = label.shape
#    print 'label height and width', l_height, l_width
#    weight = weight.reshape((1, 1, l_height, l_width))    
#        
##    sio.savemat(filename_weights,{'data':weight})
    
    
def generate_weights_fold1():
 #   input_file = '/home/rasha/Desktop/train_fold1.txt'
 #   output_file = '/home/rasha/Desktop/test_weights_file_output_mat.txt'
    
    input_file = '/home/rasha/Desktop/val_fold1.txt'
    output_file = '/home/rasha/Desktop/val_fold1_weights_mat.txt'
    
    i=0
    with open(input_file,'r') as in_file:
        with open(output_file,'w') as out_file:
            for line in in_file:
                names = line.split(' ')
                weights_filename = generate_weights(names[0])
                print i, weights_filename
                i+=1
                rep = line.replace('\n', ' '+ weights_filename + '\n')
                out_file.write(rep)
                
    
    

def show_weights(image, block=True, colorbar=True):      
    ax = plt.subplot('111')
    
    cax = plt.imshow(image, cmap = plt.get_cmap('jet'),vmin = np.min(image), vmax = np.max(image))
    ax.set_title("Weight Map")
    
    levels = np.linspace(np.min(image),np.max(image),num=8)
    
    plt.colorbar(ticks=levels, format='%.3f')
        
    plt.show(block) 
    
    
    
def test_matlab_file():
#    ar = np.arange(10)
    ar = np.array([[1.,2.5,6.,8.2],[5.,33.2,-5.1,9]], dtype='Float32')
#    ar = np.array([2.6, 3.5], dtype='Float32')
  #  ar = float(-86.7)
#    h,w = ar.shape
    print ar
    print type(ar)
    sio.savemat("/home/rasha/Desktop/nump_c.mat", {"data":ar})
    
    print sio.whosmat('/home/rasha/Desktop/nump_c.mat')
#    print h,w
    
    
    
#generate_weights('/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/4-18/MOD017/290311_19/6-1EFE5550-Input.tif')

#test_weights() 

generate_weights_fold1()
            
#area_seg_img = io.imread('/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/4-18/MOD011/220211_19/11-44339EB0-AreaSeg.tif')
#plt.imshow(area_seg_img, cmap = plt.get_cmap('gray')) 
#plt.show(True)
     
     
#fn = '/home/rasha/work/deeplab-public-ver2/drusen_area/features/RESNET-50/val/small/snapshot_/matlab_images/19-33/MOD024/090413_19/1-F9CC83D0-pred.png'
#convert_to_layers(fn)
#net_pred = read_image(fn)
#keep_largest_component_mod(net_pred)
 
#convert_all_to_layers()

#main1()

#generate_weights('/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/34-45/MOD038/210513_145/28-25512CF0-Input.tif')

#area_pred_folder = '/home/rasha/work/deeplab-public-ver2/drusen_area/features/RESNET-50/val/small/snapshot_/post_densecrf'
#convert_to_layers(area_pred_folder + '/34-45__MOD038__070616_19__17-9D2937F0__2_1_3-pred.png')        
#convert_to_layers(area_pred_folder + '/34-45__MOD038__070616_19__17-9D2937F0__4_1_16-pred.png')
#convert_to_layers(area_pred_folder + '/34-45__MOD038__070616_19__17-9D2937F0__4_2_1_16-pred.png')
#convert_to_layers(area_pred_folder + '/34-45__MOD038__070616_19__17-9D2937F0__8_2_1_16-pred.png')

#test_matlab_file()

print 'Done'






