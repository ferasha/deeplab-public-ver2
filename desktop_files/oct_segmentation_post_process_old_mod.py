import numpy as np
import scipy as sc
from scipy import misc
from scipy import signal
import os
from os import listdir
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from os.path import isfile, join
from skimage import data, io, external
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from collections import namedtuple
from skimage import filters
from Image import Image as ImageObj
#import filterfunctions as FF
import pickle
#import oct_segmentation_post_process as pp
import re

h_threshold = 0
max_h_t = 2 # For the ground truth it is 2
w_over_h_ratio_threshold = 10000

def show_image( image, block = True ):
    plt.imshow( image, cmap = plt.get_cmap('gray'))
    plt.show(block)
    
def show_image_rgb( image, title = "", d = 0, save_path = "", name = "", block = True ):
    fig = plt.figure(figsize=(15.0, 15.0))
    
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.xaxis.set_visible( False )
    ax.yaxis.set_visible( False )
    plt.imshow(image)

    if( save_path != "" ):
        plt.savefig(save_path + "res/"+str(d)+name+".png")
        plt.close()
    else:
        plt.show(block)

def show_images( images, r, c, titles = [], d = 0 , save_path = "" , block = True):
    fig = plt.figure(figsize=(15.0, 15.0))
    i = 1
    
    for img in images:
        ax = plt.subplot( r, c, i )
      #  ax.xaxis.set_visible( False )
      #  ax.yaxis.set_visible( False )
        if( len(titles) != 0 ):
            ax.set_title( titles[i-1] )
        if( len(img.shape) > 2 ):
            plt.imshow( img )
        else:
            plt.imshow( img , cmap = plt.get_cmap('gray'))
        i += 1
    if( save_path != "" ):
        plt.savefig(path + "res/drusen-"+str(d)+".png")
        plt.close()
    else:
        plt.show(block)

def show_PED_volume( b_scans, value, interplolation = 'bilinear' , block = True):
    print b_scans.shape
   
    h, w, d = b_scans.shape
    img = (b_scans>200).astype(float)
    img = sc.ndimage.filters.gaussian_filter(img,0.5)
    show_image(img[:,:,0])
    Z, X, Y = np.where( img >= 0.2 )
    '''
    i  = np.arange(len(Z))
    ii = np.where( i%10 == 0 )
    X  = X[ii]
    Y  = Y[ii]
    Z  = Z[ii]
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.plot_trisurf(X, Y, Z,cmap=plt.cm.RdYlBu_r, antialiased=False,edgecolor='none')
    ax.set_zlim(180, 300)
  #  ax.scatter(X, Y, Z, c='r', marker='o')   
    plt.show(block)
   
   
def increase_resolution( b_scans, factor ):
    new_size = (b_scans.shape[0], b_scans.shape[1], b_scans.shape[2]*factor)
   # res = misc.imresize(b_scans, new_size, interp = 'nearest')
    res = np.zeros(new_size)
    for i in range(new_size[0]):
        slice_i = b_scans[i, :,:]
        res[i,:,:] = misc.imresize(slice_i, (new_size[1], new_size[2]))
    return res
    
def get_RPE_location( seg_img ):
    y = []
    x = []
    tmp = np.copy(seg_img)
    if( np.sum(seg_img)==0.0):
        return y, x
    if( len(np.unique(tmp)) == 4 ):
        tmp2 = np.zeros(tmp.shape)
        tmp2[np.where(tmp==170)] = 255
        tmp2[np.where(tmp==255)] = 255
        y, x = np.where(tmp2==255)
      
    else:
        y, x = np.where(tmp==255)
    return y, x
    
def get_BM_location( seg_img ):
    y = []
    x = []
    tmp = np.copy(seg_img)
    if( np.sum(seg_img)==0.0):
        return y, x
    if( len(np.unique(tmp)) == 4 ):
        tmp2 = np.zeros(tmp.shape)
        tmp2[np.where(tmp==170)] = 255
        tmp2[np.where(tmp==85)] = 255
        y, x = np.where(tmp2==255)
      
    else:
        y, x = np.where(tmp==127)
    return y, x    

def compute_distance_btw_layers( rpe_mask, bm_mask ):
    '''
    rpe_mask = np.logical_or(b_scan == 255, b_scan == 170).astype('int')
    bm_mask  = np.logical_or(b_scan == 170, b_scan == 85)
    bm_mask  = np.logical_or(bm_mask, b_scan == 127).astype('int')
    '''
    h, w = rpe_mask.shape
    distance = 0.0
    for j in range(w):
        rpe_col = rpe_mask[:, j]
        bm_col  = bm_mask[:, j]
        rpe_loc = np.where(rpe_col > 0.0)
        bm_loc  = np.where(bm_col  > 0.0)
       # print rpe_loc, bm_loc
        if( len(rpe_loc[0]) != 0 and len(bm_loc[0]) != 0 ):
            distance += abs(bm_loc[0][0] - rpe_loc[0][0])

    return distance

def denoise_BM(seg_img, farDiff=2,max_deg = 8, it = 5):
    #print "here"
    by, bx = get_BM_location( seg_img )    
   # print np.unique(seg_img)
  #  show_image( seg_img)
    for deg in range(5,6):
        
        tmpx = np.copy(bx)
        tmpy = np.copy(by)
        finD = 0
     #   print "Degree--------:",deg
        for i in range(it):
           # print i
            z = np.polyfit(tmpx, tmpy, deg = deg)
            p = np.poly1d(z)
            
            new_y = p(bx).astype('int')
            if( True ):
                tmpx = list()
                tmpy = list()
                c = 0
                for j in range(0,len(bx)):
                  diff=new_y[j]-by[j]
                  finD+=abs(diff)
                  if abs(diff)<farDiff:                          
                      tmpx.append(bx[j])
                      tmpy.append(by[j])
      #          print "Dist=", finD/float(len(bx))
        z = np.polyfit(tmpx, tmpy, deg = deg)
        p = np.poly1d(z)
        
        newY = p(bx).astype('int')
        newMask = np.zeros(seg_img.shape)
        
        a = np.zeros(seg_img.shape)
        a[get_RPE_location(seg_img)] = 1.0
        b = np.zeros(seg_img.shape)
        b[newY, bx] = 1.0
        union = a*b
   #     show_image(union)
        if( np.sum(union)==0 ):
            newMask[get_RPE_location(seg_img)] = 255.0
            newMask[newY, bx]  = 127.0
        else:
            newMask[get_RPE_location(seg_img)] = 255.0
            newMask[newY, bx]  = 85.0
            newMask[(union)>0] = 170.0
       # print deg
    #show_image(newMask)
    return newMask
        
    
# Given the RPE layer, estimate the normal RPE    
def normal_RPE_estimation( b_scan , degree = 3, it = 3, s_ratio = 1, \
                        farDiff = 5, ignoreFarPoints=True, returnImg=False,\
                        useBM = False,useWarping=True,xloc=[],yloc=[]):   
                            
    if(useWarping):
        y, x = get_RPE_location(b_scan)
        yn, xn = warp_BM(b_scan)
        #print "*********", len(x), len(xn)
        return yn, xn
    if( useBM ):
        y_b, x_b = get_BM_location( b_scan ) 
        y_r, x_r = get_RPE_location( b_scan )  
        
        z = np.polyfit(x_b, y_b, deg = degree)            
        p = np.poly1d(z)        
        y_b = p(x_r).astype('int')
        
        prev_dist = np.inf
        offset = 0
        for i in range(50):
             newyb = y_b - i 
            # rpe_mask = np.zeros(b_scan.shape)
            # bm_mask  = np.zeros(b_scan.shape)
            # rpe_mask[y_r, x_r] = 1.0
            # rpe_mask[newyb, x_b] = 1.0
             diff  = np.sum(np.abs(newyb-y_r))
             if( diff < prev_dist ):
                  prev_dist = diff
                  continue
             offset = i
             break
        if( returnImg ):
            img = np.zeros(b_scan.shape)
            img[y_b-offset, x_r] = 255.0
            return y_b-offset, x_r, img
        return y_b-offset, x_r
        
    tmp = np.copy(b_scan)
    y = []
    x = []
    if(xloc==[] or yloc==[]):
        if( np.sum(b_scan)==0.0):
            return y, x
        if( len(np.unique(tmp)) == 4 ):
            tmp2 = np.zeros(tmp.shape)
            tmp2[np.where(tmp==170)] = 255
            tmp2[np.where(tmp==255)] = 255
            y, x = np.where(tmp2==255)
          
        else:
            y, x = np.where(tmp==255)
    else:
        y = yloc
        x = xloc
    tmpx = np.copy(x)
    tmpy = np.copy(y)
    origy = np.copy(y)
    origx = np.copy(x)
    finalx = np.copy(tmpx)
    finaly = tmpy
    for i in range(it):
        if( s_ratio > 1 ):
            s_rate = len(tmpx)/s_ratio
            rand   = np.random.rand(s_rate) * len(tmpx)
            rand   = rand.astype('int')            
            
            sx = tmpx[rand]
            sy = tmpy[rand]
            
            z = np.polyfit(sx, sy, deg = degree)
            
        else:
            z = np.polyfit(tmpx, tmpy, deg = degree)
            
        p = np.poly1d(z)
        
        new_y = p(finalx).astype('int')
        if( ignoreFarPoints ):
            tmpx = []
            tmpy = []
            for i in range(0,len(origx)):
              diff=abs(new_y[i]-origy[i])
    
              if diff<farDiff:
                  tmpx.append(origx[i])
                  tmpy.append(origy[i])
        else:
            tmpy = np.maximum(new_y, tmpy)
        
        #finalx = tmpx
        finaly = new_y
    
    #tmp.fill(0.0) 
    #tmp[finaly, finalx] = 127
   # tmp[y,x] = 255
    #finaly, finalx = np.where(tmp>0)
    if( returnImg ):
        return finaly, finalx, tmp
    
    return finaly, finalx
    
    
    
    

    
def map_bm_to_rpe( b_scan ):
    bm_mask  = (b_scan == 127).astype('float')
    rpe_mask = (b_scan == 255).astype('float')
    
    h, w = bm_mask.shape

    cc = signal.fftconvolve(rpe_mask, np.fliplr(np.flipud(bm_mask)), mode='same')
    loc_max = np.where(cc==(np.max(cc)))
    
    center  = np.asarray([h/2, w/2])
    loc_max = np.asarray([loc_max[0][0], loc_max[1][0]])

    offset  = loc_max - center

    shifted_bm = sc.ndimage.interpolation.shift(bm_mask,offset,mode='constant')
    result = np.zeros((h, w))
    result[shifted_bm>0.5] = 1
    return result
   # result = np.zeros((h, w))
   # result[rpe_mask==1] = 255
   # result[shifted_bm>0.5] = 127
    
   # show_image(result)
    
def compute_distance_image( b_scan ):
    rpe_mask   = (b_scan == 255).astype('int')
    shifted_bm = map_bm_to_rpe( b_scan )
    
    h, w = b_scan.shape
    distance = np.zeros((w))
    distance_img = np.zeros((h, w))
    for j in range(w):
        rpe_col = rpe_mask[:, j]
        bm_col  = shifted_bm[:, j]
        rpe_loc = np.where(rpe_col > 0.0)
        bm_loc  = np.where(bm_col  > 0.0)
       # print rpe_loc, bm_loc
        if( len(rpe_loc[0]) != 0 and len(bm_loc[0]) != 0 ):
           
            if ( (bm_loc[0][0] - rpe_loc[0][0])>0 ):
                distance[j] = bm_loc[0][0] - rpe_loc[0][0]
    i = np.arange(w)       
    j = distance.astype('int')  + 250
    distance_img[j,i] = 1
    return distance_img
    
# Compute the average distant btw the given layer with given value in prediction 
# and ground truth
def compute_curve_distance( gt, pr, layer_type = 'RPE' ):

    gt_x = [];gt_y = []
    pr_x = [];pr_y = []
    gt_mask = np.zeros( gt.shape )
    pr_mask = np.copy( gt_mask )
    
    if( len(np.unique(gt)) == 4 ):
        if( layer_type == 'RPE' ):
            gt_mask[np.where(gt == 255)] = 1
            gt_mask[np.where(gt == 170)] = 1
        elif ( layer_type == 'BM' ):
            gt_mask[np.where(gt == 255)] = 1
            gt_mask[np.where(gt == 85)]  = 1            

    else:
        if( layer_type == 'RPE' ):
            gt_mask[np.where(gt == 255)] = 1  
        elif ( layer_type == 'BM' ):
            gt_mask[np.where(gt == 127)] = 1
            
    if( len(np.unique(pr)) == 4 ):
        if( layer_type == 'RPE' ):
            pr_mask[np.where(pr == 255)] = 1
            pr_mask[np.where(pr == 170)] = 1
        elif ( layer_type == 'BM' ):
            pr_mask[np.where(pr == 255)] = 1
            pr_mask[np.where(pr == 85)]  = 1            

    else:
        if( layer_type == 'RPE' ):
            pr_mask[np.where(pr == 255)] = 1  
        elif ( layer_type == 'BM' ):
            pr_mask[np.where(pr == 127)] = 1  
                
    
    sum_dist = 0
    num_data = 0    
    h, w = gt.shape    
    for j in range( w ):
        col1 = gt_mask[:, j]
        col2 = pr_mask[:, j]
        l_1 = np.where( col1 == 1 )
        l_2 = np.where( col2 == 1 )
        
        if(len(l_1[0]) != 0 and len(l_2[0]) != 0 ):
            
            sum_dist += abs( l_2[0][0] - l_1[0][0] )
            num_data += 1
            
    avg_dist = 0 if num_data == 0 else sum_dist/float(num_data)
    
    return avg_dist
    
# Only for the parts that have drusen, according to ground truth
def compute_curve_distance_on_drusens( gt, pr, layer_type = 'RPE' ):

    gt_x = [];gt_y = []
    pr_x = [];pr_y = []
    gt_mask = np.zeros( gt.shape )
    pr_mask = np.copy( gt_mask )
    
    if( len(np.unique(gt)) == 4 ):
        if( layer_type == 'RPE' ):
            gt_mask[np.where(gt == 255)] = 1
            gt_mask[np.where(gt == 170)] = 1
        elif ( layer_type == 'BM' ):
            gt_mask[np.where(gt == 255)] = 1
            gt_mask[np.where(gt == 85)]  = 1            

    else:
        if( layer_type == 'RPE' ):
            gt_mask[np.where(gt == 255)] = 1  
        elif ( layer_type == 'BM' ):
            gt_mask[np.where(gt == 127)] = 1
            
    if( len(np.unique(pr)) == 4 ):
        if( layer_type == 'RPE' ):
            pr_mask[np.where(pr == 255)] = 1
            pr_mask[np.where(pr == 170)] = 1
        elif ( layer_type == 'BM' ):
            pr_mask[np.where(pr == 255)] = 1
            pr_mask[np.where(pr == 85)]  = 1            

    else:
        if( layer_type == 'RPE' ):
            pr_mask[np.where(pr == 255)] = 1  
        elif ( layer_type == 'BM' ):
            pr_mask[np.where(pr == 127)] = 1  
                
    # Find drusen mask       
    drusen_mask = compute_drusen_mask( gt )
    drusen_mask = filter_drusen_by_size( drusen_mask)
    #show_image(drusen_mask)
    drusen_mask = sc.ndimage.morphology.binary_dilation( drusen_mask, iterations = 10)
    #show_image(drusen_mask)
    y, x = np.where( drusen_mask == 1 )    
    
    sum_dist = 0
    num_data = 0    
    h, w = gt.shape    
    for j in range( w ):
        col1 = gt_mask[:, j]
        col2 = pr_mask[:, j]
        l_1 = np.where( col1 == 1 )
        l_2 = np.where( col2 == 1 )
        
        if(len(l_1[0]) != 0 and len(l_2[0]) != 0 and (j in x) ):
           # print l_2[0][0] , l_1[0][0] 
            sum_dist += abs( l_2[0][0] - l_1[0][0] )
            num_data += 1
            
    avg_dist = 0 if num_data == 0 else sum_dist/float(num_data)
    
    return avg_dist

def overlay_prediction_on_gt( img, gt, p ):
    img_rgb = np.dstack((img, img, img))
   # print d, np.unique(gt), np.unique(p)
    
    if( len(np.unique(gt)) == 4 ):
        # Draw ground truth on the image
        img_rgb[gt == 255,:] = [255,0,0]
        img_rgb[gt == 170,:] = [255,0,0]  
        img_rgb[gt == 85,:] = [255, 238, 0]
    else:
        # Draw ground truth on the image
        img_rgb[gt == 255,:] = [255,0,0]  
        img_rgb[gt == 127,:] = [255, 238, 0]
    
    if( len(np.unique(p)) == 4 ):
        img_rgb[p  == 255,:] = [0, 182, 255]
        img_rgb[p  == 170,:] = [0, 182, 255]
        img_rgb[p  == 85,:] = [255, 0, 255]
    else:
        img_rgb[p  == 255,:] = [0, 182, 255]
        img_rgb[p  == 127,:] = [255, 0, 255]
    

  #  misc.imsave(path + "res/"+str(d)+".png", img_rgb)
    return img_rgb

def compute_drusen_mask( seg_img, useWarping= True):
    y = []
    x = []
    
    mask = np.zeros( seg_img.shape )
    
    if( len(np.unique(seg_img)) == 4 ):
        tmp = np.zeros(seg_img.shape)
        tmp[np.where(seg_img==170)] = 255
        tmp[np.where(seg_img==255)] = 255
        y, x = np.where(tmp==255)
      
    else:
        y, x = np.where(seg_img==255)
    
    y_n, x_n = normal_RPE_estimation( seg_img, useWarping=useWarping )
    aa = np.copy(seg_img)
    aa[y_n,x_n]=70
   # show_image(aa)
    label = np.zeros(seg_img.shape)
    label[y ,x] = 2
    label[y_n, x_n] = 1
  #  print "Here"
  #  show_image( label )
    # Draw the drusens on the red channel
    h, w = seg_img.shape
    for j in range( w ):
        col = label[:, j]
        l_1 = np.where( col == 1 )
        l_2 = np.where( col == 2 )
        if(len(l_1[0]) != 0 and len(l_2[0]) != 0 ):
            mask[l_2[0][0]:l_1[0][0], j] = 1
            
    return mask

def compute_average_heigh_in_area_seg( area_image ):
    heights = np.sum( area_image, axis = 0 )
    denom   = np.sum( ( heights > 0.0 ).astype('float') )
    average_height = float(np.sum(heights))/denom if denom != 0.0 else 0.0
    return average_height
    
# Compute how large is a component and then replace the pixel value in each 
# component with the size of the area it belongs to
# return the extent and the sizes of the regions
def compute_extent( cca ):
    labels = np.unique( cca )
    extent = np.zeros( cca.shape )  
    c_size = []      
    for l in labels:
        region = cca == l
        r_size = np.sum( region ) 
        extent[np.where(cca==l)] = r_size
        c_size.append( r_size )
    return extent, c_size
    
def compute_heights( cca ):
    bg_lbl  = get_label_of_largest_component( cca )
    mask  = cca != bg_lbl
    mask  = mask.astype('int')
    cvr_h = np.sum( mask, axis = 0 )
    hghts = np.tile( cvr_h, cca.shape[0] ).reshape(cca.shape)
    #show_image(hghts)
    mask  = mask * hghts
    return mask
    
def find_rel_maxima( arr ):
    val = []
    pre = -1
    for a in arr:
        if( a != pre ):
            val.append(a)
        pre = a
    val = np.asarray(val)
    return val[sc.signal.argrelextrema(val, np.greater)]
    
def find_argrel_minima( arr ):
    val = []
    ind = []
    i = 0
    pre = -1
    for a in arr:
        if( a != pre ):
            val.append(a)
            ind.append(i)
        pre = a
        i += 1
    val = np.asarray(val)
    minima =  sc.signal.argrelextrema(val, np.less)
    gen_ind = []
   
    ind = np.asarray( ind )
    minima = np.asarray(minima)
  
    for i in minima:
        
        gen_ind.append(ind[i])
    #print val, minima, gen_ind
    return gen_ind
    
def compute_component_sum_local_max_height( cca ):
    labels  = np.unique( cca )
    max_hs  = np.zeros( cca.shape )  
    bg_lbl  = get_label_of_largest_component( cca )
    heights = compute_heights( cca )
    for l in labels:
        if( l != bg_lbl ):
            region = cca == l
            masked_heights = region * heights
            col_h = np.max( masked_heights, axis = 0 )
            local_maxima   = find_rel_maxima( col_h )
          #  print col_h[col_h>0], sc.signal.argrelextrema(col_h, np.greater)
            if( len(local_maxima) == 0 ):
                local_maxima = np.asarray([np.max(masked_heights)])
           # print local_maxima
            max_hs[region] = np.sum(local_maxima)        

    
    return max_hs
    
def compute_component_max_height( cca ):
    labels  = np.unique( cca )
    max_hs  = np.zeros( cca.shape )  
    bg_lbl  = get_label_of_largest_component( cca )
    heights = compute_heights( cca )
    for l in labels:
        if( l != bg_lbl ):
            region = cca == l
            max_hs[region] = np.max( region * heights )
        
    return max_hs

def compute_component_width( cca ):
    labels  = np.unique( cca )
    max_ws  = np.zeros( cca.shape )  
    bg_lbl  = get_label_of_largest_component( cca )
    for l in labels:
        if( l != bg_lbl ):
            y, x = np.where( cca == l )
            w = np.max(x) - np.min(x)
            max_ws[cca == l] = w
    return max_ws

def compute_width_height_ratio_height_local_max( cca ):
    mx_h = compute_component_sum_local_max_height( cca )
    mx_w = compute_component_width( cca )
    mx_h[mx_h == 0] = 1
    return mx_w.astype('float')/(mx_h.astype('float')), mx_h
    
def compute_width_height_ratio( cca ):
    mx_h = compute_component_max_height( cca )
    mx_w = compute_component_width( cca )
    mx_h[mx_h == 0] = 1
    return mx_w.astype('float')/(mx_h.astype('float')), mx_h
def separate_by_local_minima( cca ):
    mask = cca>0.0
    
    labels  = np.unique( cca )
    max_hs  = np.zeros( cca.shape )  
    bg_lbl  = get_label_of_largest_component( cca )
    mask[cca==bg_lbl] = 0.0
    
    heights = compute_heights( cca )
    for l in labels:
        if( l != bg_lbl ):
            region = cca == l
            masked_heights = region * heights
            col_h = np.max( masked_heights, axis = 0 )
            local_minima   = find_argrel_minima( col_h )
          #  print col_h[col_h>0], sc.signal.argrelextrema(col_h, np.greater)
            #if( len(local_minima) == 0 ):
            #    local_minima = np.asarray([np.min(masked_heights)])
           # print local_minima
            mask[:,local_minima] = 0.0
            

    
    return mask
    
def filter_drusen_by_size( dmask, slice_num=-1 ): 
    drusen_mask = np.copy( dmask )
    
    # Compute connected component analysis    
    #cca, num_drusen = sc.ndimage.measurements.label( drusen_mask )
    #drusen_mask = separate_by_local_minima(cca)
    
    if( h_threshold == 0.0 and  max_h_t == 0.0 and w_over_h_ratio_threshold == 10000.0 ):
        return drusen_mask
        
    
    
    
    cca, num_drusen = sc.ndimage.measurements.label( drusen_mask )
  #  show_images([drusen_mask, aaa],1,2)
    # Find the size of each component    
    #extent = compute_extent( cca )[0]
    filtered_mask = np.ones( drusen_mask.shape )
    h  = compute_heights( cca )
    filtered_mask[np.where( h <=  h_threshold )] = 0.0
  
    h  = compute_component_max_height( cca )
    filtered_mask[np.where( h <=  max_h_t )] = 0.0
    
    cca, num_drusen = sc.ndimage.measurements.label( filtered_mask )
    
    # Find the ratio of height over component width and  maximum hight of each component
    w_o_h, height  = compute_width_height_ratio_height_local_max( cca )

   
    filtered_mask = np.ones( drusen_mask.shape ).astype('float')
    
    filtered_mask[np.where(w_o_h  >  w_over_h_ratio_threshold)] = 0.0
    filtered_mask[np.where(w_o_h == 0.0)] = 0.0
   
    if( slice_num >= 700 and slice_num<= 75 ):
        show_images([drusen_mask, w_o_h,height,filtered_mask ], 2,2,["input","woh","height", "final"])
           
    # Set the maximum component, which is background to zero
    #extent[np.where(extent==extent.max())] = 0.0
    
    # Remove false positives based on size threshold
    #extent[np.where(extent <= 0)] = 0.0
    
   # filtered_mask = extent > 0.0
    
    return filtered_mask
    
def quantify_drusen( drusen_mask ):
        
    # Compute connected component analysis    
    cca, num_drusen = sc.ndimage.measurements.label( drusen_mask )
    
    # Find the size of each component    
    extent, sizes = compute_extent( cca )
    
    # Set the maximum component, which is background to zero
    extent[np.where(extent==extent.max())] = 0.0
    
    drusen_sizes = np.asarray(sorted(sizes)[:-1])
    
    return num_drusen, drusen_sizes
 

def remove_drusen_with_1slice_size( projection_mask ):
    mask = np.copy( projection_mask )
    cca, numL = sc.ndimage.measurements.label( mask )
    bgL = get_label_of_largest_component( cca )
    for l in np.unique( cca ):
        if( l != bgL ):
            
            y, x = np.where( cca == l )
            if(len(np.unique(y)) == 1 ):
                mask[y,x] = 0.0
    return mask
    
def get_drusen_quantification_info( seg_img ):
    y = []
    x = []
    if( len(np.unique(seg_img)) == 4 ):
        tmp = np.zeros(seg_img.shape)
        tmp[np.where(seg_img==170)] = 255
        tmp[np.where(seg_img==255)] = 255
        y, x = np.where(tmp==255)
      
    else:
        y, x = np.where(seg_img==255)
        
    # Find the perfect RPE through polynomial fitting
    y_n, x_n = normal_RPE_estimation( seg_img )
    
    # Find drusen mask       
    drusen_mask = compute_drusen_mask( seg_img )
    drusen_mask = filter_drusen_by_size( drusen_mask)
    
   
    num_drusen , drusen_sizes = quantify_drusen( drusen_mask )
    sizes_str = ""
    #print drusen_sizes
    for s in drusen_sizes:
        sizes_str += str(int(s)) + ', '
    sizes_str = sizes_str[:-2]
    
    info =  'Drusen #:' + str(num_drusen) +  '   Drusen Size: ' + sizes_str
    return info
    
def mark_drusen_on_b_scan( b_scan, seg_img ):
    '''
    y = []
    x = []
    if( len(np.unique(seg_img)) == 4 ):
        tmp = np.zeros(seg_img.shape)
        tmp[np.where(seg_img==170)] = 255
        tmp[np.where(seg_img==255)] = 255
        y, x = np.where(tmp==255)
      
    else:
        y, x = np.where(seg_img==255)
        
    # Find the perfect RPE through polynomial fitting
    y_n, x_n = normal_RPE_estimation( seg_img )
    ''' 
    # Make RGB image
    img_rgb = np.dstack((b_scan, b_scan, b_scan))
    
    # Find drusen mask       
    drusen_mask = compute_drusen_mask( seg_img )
    drusen_mask = filter_drusen_by_size( drusen_mask)
    
    # Draw the drusens on the red channel
    i, j = np.where(drusen_mask==1)
    img_rgb[i, j, 0]  = 255
    img_rgb[i, j, 1]  = img_rgb[i, j, 1] * 0.5
    img_rgb[i, j, 2]  = img_rgb[i, j, 2] * 0.5
    
    return img_rgb
def my_sort(yn,xn):
    xy = sorted(zip(xn,yn))
    dtype=[('x','int'),('y','int')]
    xyarr = np.array(xy, dtype=dtype)
    x = xyarr['x']
    y = xyarr['y']
    return y, x
'''    
def warpScans(seg_imgs):
    warpedSegImgs = np.zeros(seg_imgs.shape)
    for i in range(seg_imgs.shape[2]):
        img = seg_imgs[:,:,i]
'''        
def warp_BM( seg_img, returnWarpedImg=False ):
    h, w = seg_img.shape
    yr, xr = get_RPE_location( seg_img )
    yb, xb = get_BM_location( seg_img )
    rmask  = np.zeros((h, w), dtype='int')
    bmask  = np.zeros((h, w), dtype='int')
  
    rmask[yr, xr] = 255
    bmask[yb, xb] = 255
  #  show_image(seg_img)
    vis_img = np.copy(seg_img)
    shifted = np.zeros(vis_img.shape)
    wvector = np.empty((w), dtype='int')
    wvector.fill(h-(h/2))
    nrmask = np.zeros((h,w), dtype='int')
    nbmask = np.zeros((h,w), dtype='int')
    
    zero_x =[]
    zero_part = False  
    last_nonzero_diff = 0
    for i in range(w):
        bcol = np.where(bmask[:,i]>0)[0]
        wvector[i] = wvector[i]-np.max(bcol) if len(bcol) > 0 else 0
        #print "currecnt diff",i, ":",wvector[i]
        #print "zero_x:", zero_x
        if( len(bcol) == 0  ):
            zero_part = True
            zero_x.append(i)
        if( len(bcol)>0 and zero_part ):
            diff = wvector[i]
            zero_part = False
            wvector[zero_x]=diff
         #   print "Edited diff ",i,": ", wvector[zero_x]
         #   print zero_x
            zero_x=[]
        if( len(bcol)>0):
            last_nonzero_diff = wvector[i]
        if( i == w-1 and zero_part):
            wvector[zero_x]=last_nonzero_diff
        
       # print bcol,wvector[i]
        
        #show_images([rmask, bmask, nrmask, nbmask],2,2)
    # Where wvector is zero, set the displacing to the displaceing of a non zero
    # neighbour
    for i in range(w):
        nrmask[:, i] = np.roll(rmask[:,i], wvector[i])
        nbmask[:, i] = np.roll(bmask[:,i], wvector[i])
        shifted[:, i] = np.roll(vis_img[:,i], wvector[i])
    shifted_yr =[]   
    for i in range(len(xr)):
        shifted_yr.append(yr[i] + wvector[xr[i]])
   # show_images([rmask, bmask, nrmask, nbmask],2,2)

    yn, xn = normal_RPE_estimation( rmask,it=5,useWarping=False, xloc=xr, yloc=shifted_yr )
    for i in range(len(xn)):
        yn[i] = yn[i] - wvector[xn[i]]
  
   # print "---------------b  ",len(xn) 
   # finMask = np.zeros((h,w))
   # finMask[yn,xn]=255
   # finMask[yr,xr]=127
   # show_image(finMask)
    #yn, xn = np.where(finMask>0)
    '''
    print "---------------c  ",len(xn) 
    for i in range(w):  
        
        #show_images([rmask, bmask, nrmask, nbmask],2,2)
        finMask[:, i] = np.roll(finMask[:,i], -1*wvector[i])
    '''
  #  yn, xn = np.where(finMask>0)
    
   # print "---------------c  ",len(xn) 
  #  seg_i = np.copy(seg_img)
  #  seg_i[yn, xn]=70
    
    #nrmask[yn,xn]=127
    # Warp back
   # show_images([vis_img, shifted],1,2)
    #yn, xn = my_sort(yn,xn)
    if(returnWarpedImg):
        return shifted
        
    return yn, xn
def make_overlay_of_drusen_masks( b_scan, gt, pr ): 
    
    # Find drusen mask       
    drusen_mask = compute_drusen_mask( gt )
    drusen_mask_gt = filter_drusen_by_size( drusen_mask)
    
    drusen_mask = compute_drusen_mask( pr )
    drusen_mask_pr = filter_drusen_by_size( drusen_mask)
    
    sim = np.logical_and(drusen_mask_gt , drusen_mask_pr)
    
    # Make RGB image
    img_rgb = np.dstack((b_scan, b_scan, b_scan))
    i, j = np.where(drusen_mask_gt==True)
    img_rgb[i, j, 0]  = 255
    img_rgb[i, j, 1]  = img_rgb[i, j, 1] * 0.1
    img_rgb[i, j, 2]  = img_rgb[i, j, 2] * 0.1
    

    i, j = np.where(drusen_mask_pr==True)
    img_rgb[i, j, 2]  = 255
    img_rgb[i, j, 1]  = img_rgb[i, j, 1] * 0.1
    img_rgb[i, j, 0]  = img_rgb[i, j, 0] * 0.1
    
    i, j = np.where(sim==True)
    img_rgb[i, j, :] = [255,255,0]
      
    return img_rgb
    
def compute_true_and_false_positives( gt, pr , input_type='line_segments'):
    if( input_type == 'line_segments' ):
        # Find drusen mask       
        drusen_mask = compute_drusen_mask( gt )
        drusen_mask_gt = filter_drusen_by_size( drusen_mask)
        
        drusen_mask = compute_drusen_mask( pr )
        drusen_mask_pr = filter_drusen_by_size( drusen_mask)
        
    elif( input_type == 'drusen'):
        drusen_mask_gt = gt
        drusen_mask_pr = pr
    
    sim = np.logical_and(drusen_mask_gt , drusen_mask_pr)*1.0
       
    true_positive  = np.sum(sim)
    false_positive = np.sum((drusen_mask_gt.astype('int')-drusen_mask_pr.astype('int'))==-1)

    neg_drusen_mask_gt = np.logical_not(drusen_mask_gt)
    neg_drusen_mask_pr = np.logical_not(drusen_mask_pr)
    
    neg_sim = np.logical_and(neg_drusen_mask_gt , neg_drusen_mask_pr)*1.0
    true_negative  = np.sum(neg_sim)
    false_negative = np.sum((neg_drusen_mask_gt.astype('int')-neg_drusen_mask_pr.astype('int'))==-1)
    
    OR_val = float(true_positive/float(np.sum(drusen_mask_gt))) if float(np.sum(drusen_mask_gt)) != 0 else np.inf
    
    # Intersection Over Union
    IOU = float(true_positive)/float(true_positive+false_positive+false_negative) if float(true_positive+false_positive+false_negative) != 0 else 1.0
    
    return int(true_positive), int(false_positive),\
           OR_val,\
           int(abs(np.sum(drusen_mask_gt)-np.sum(drusen_mask_pr))), IOU
    
    
    
   # show_images([drusen_mask_gt, drusen_mask_pr],1,2)
    
def read_b_scans( path , img_type = "None"):
    d2 = [f for f in listdir(path) if isfile(join(path, f))]

    rawstack = list()
    ind = list()
    for fi in range(len(d2)):
         filename = path+'/'+d2[fi]
         
         ftype = d2[fi].split('-')[-1]
         
         if(ftype==img_type or img_type=="None"):
             ind.append(int(d2[fi].split('-')[0]))
             
             raw = io.imread(filename)
             rawSize = raw.shape
             if( len(rawstack)==0 ):
                 rawstack = raw
                 
             else:
                 rawstack = np.dstack((rawstack, raw))
    
    ind = np.asarray(ind)
    sin = np.argsort(ind)
  
    rawstack = rawstack[:,:,sin]

    return rawstack
    
def sort_all_file_names_in( path, splitter='-' ):
    d2 = [f for f in listdir(path) if isfile(join(path, f))]
    ind = []
    for fi in range(len(d2)):
        
        if(d2[fi].split('.')[1] == 'tif' ):
            ind.append(int(d2[fi].split(splitter)[0]))
        d2[fi] = path + d2[fi]  
    ind = np.asarray(ind)
    sin = np.argsort(ind)

    d2 = np.asarray(d2)
  
    d2  = d2[sin]
    
    return d2
    
def sort_all_file_names_in_recursive_dir( path, splitter='-' ):
    for d1 in path:
        for d2 in os.listdir(path+'/'+d1):
            for d3 in os.listdir(path+'/'+d1+'/'+d2):
                
                for f in os.listdir(path+'/'+d1+'/'+d2+'/'+d3):
                    filename = path+'/'+d1+'/'+d2+'/'+d3+'/'+f
    
def get_label_of_largest_component( labels ):
    size = np.bincount(labels.ravel())
    largest_comp_ind = size.argmax() 
    return largest_comp_ind
    

def draw_histogram( data, bins, save_name="", block = True ):
    plt.hist(data, bins)
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    if( save_name != "" ):
        plt.savefig(save_name)
        plt.close()
        return
    plt.show(block)

def box_plot(data, xlabels, title,save_name = ""):
    fig = plt.figure()
    ax = plt.axes()
    plt.title(title)
    # basic plot    
    plt.boxplot(data, positions = np.arange(len(data))+1,sym='')
    ax.set_xticklabels(xlabels)
    ax.set_xticks(np.arange(len(data))+1)
    if( save_name != "" ):
        plt.savefig(save_name)
        plt.close()
        return
    plt.show(True)
    
def filter_drusen_img_with_line_seg_img( dr_pr, ln_pr,useWarping=True ):
    # Find drusen mask       
    drusen_mask = compute_drusen_mask( ln_pr , useWarping=useWarping)
    drusen_mask_ln = filter_drusen_by_size( drusen_mask)

    # Apply CCA on the drusen mask
    # Compute connected component analysis    
    cca, num_drusen = sc.ndimage.measurements.label( drusen_mask_ln )
    
    labels = np.unique( cca )
    mask   = np.zeros(drusen_mask.shape)
    
    starts = []
    end    = []
    
    bg_label = get_label_of_largest_component( cca )

    for l in labels:
        if( l != bg_label ):
          
            xs = np.where( cca == l )[1]
      
            mask[:,np.min(xs):np.max(xs)] = 1.0
  
    mask = mask * dr_pr
    cca, num_drusen = sc.ndimage.measurements.label( mask )
    drusen_mask_ln = drusen_mask_ln * cca

    
    labels_to_keep = np.unique( drusen_mask_ln )
    bg_label = get_label_of_largest_component( cca )
    cca[np.where(cca==bg_label)] = 0.0
    mask_labels = np.unique(cca)
    for l in mask_labels:
        # Not background
        if( (not l in labels_to_keep) ):
            cca[np.where(cca == l)] = 0.0
    mask = cca > 0.0
    return mask
    

def get_RPE_layer( seg_img ):
    y = []
    x = []
    if( len(np.unique(seg_img)) == 4 ):
        tmp = np.zeros(seg_img.shape)
        tmp[np.where(seg_img==170)] = 255
        tmp[np.where(seg_img==255)] = 255
        y, x = np.where(tmp==255)
      
    else:
        y, x = np.where(seg_img==255)
    tmp = np.zeros(seg_img.shape)
    tmp[y,x] = 255
    y,x = np.where(tmp>0)
    return y, x
    
def get_BM_layer( seg_img ):
    y = []
    x = []
  
    if( len(np.unique(seg_img)) == 4 ):
        tmp = np.zeros(seg_img.shape)
        tmp[np.where(seg_img==170)] = 255
        tmp[np.where(seg_img==85)] = 255
        y, x = np.where(tmp==255)
      
    else:
        y, x = np.where(seg_img==127)
    return y, x    
class Point:
    def __init__(self, x, y, degree = 0):
        self.x = x
        self.y = y
        self.degree = degree
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_degree(self):
        return self.degree
    def set_degree(self, degree):
        self.degree = degree

class Line:
    def __init__(self, s, e):
        self.s = s
        self.e = e
    def get_s(self):
        return self.s
    def get_e(self):
        return self.e


def overlaps(l1, l2):
    
    min_all = min(l1.get_s().get_x(), l2.get_s().get_x())
    max_all = max(l1.get_e().get_x(), l2.get_e().get_x())
    if( abs(max_all-min_all)+1 < (abs(l1.get_s().get_x()-l1.get_e().get_x())+1+abs(l2.get_s().get_x()-l2.get_e().get_x())+1) ):
        return True
    return False
    
def get_lines_from_two_slices(pnts, i ):
    lines = []
    ind1  = 0
    
    for l1 in pnts[i]:
        found_correspondance = False
        ind2  = 0
        for l2 in pnts[i+1]:
            
            if( overlaps(l1, l2) ):
                pnts[i][ind1].get_s().set_degree(pnts[i][ind1].get_s().get_degree()+1)
                pnts[i+1][ind2].get_s().set_degree(pnts[i+1][ind2].get_s().get_degree()+1)
                pnts[i][ind1].get_e().set_degree(pnts[i][ind1].get_e().get_degree()+1)
                pnts[i+1][ind2].get_e().set_degree(pnts[i+1][ind2].get_e().get_degree()+1)
                
                # Connect
                ll1 = Line(l1.get_s(), l2.get_s())
                lines.append(ll1)
                ll2 = Line(l1.get_e(), l2.get_e())
                lines.append(ll2)
                found_correspondance = True
                break
   
            ind2 += 1
        ind1 += 1
    return lines
    
def draw_drusen_boundary( image, drusen_points , pr_drusen_points=[],scale=1, block=True):
    imagee = sc.misc.imresize(image, (image.shape[0]*scale, image.shape[1]))
    plt.imshow( imagee, cmap = plt.get_cmap('gray'))
    num_slices = image.shape[0]
  
    for i in range(num_slices):
        # Find corresponding points in the next slice
        if( i < num_slices - 1):
            connections  = get_lines_from_two_slices(drusen_points, i )
            for c in connections:
                p1 = c.get_s()
                p2 = c.get_e()
                plt.plot((p1.get_x(),p2.get_x()),(p1.get_y()*scale,p2.get_y()*scale),c = 'r')
                
    for i in range(num_slices):
        for l in drusen_points[i]:
            if(l.get_s().get_degree() == 1 and l.get_e().get_degree() == 1 ):
                plt.plot((l.get_s().get_x(),l.get_e().get_x()),(l.get_s().get_y()*scale,l.get_e().get_y()*scale),c = 'r')
     

    if( pr_drusen_points != [] ):
        num_slices = image.shape[0]
  
        for i in range(num_slices):
            # Find corresponding points in the next slice
            if( i < num_slices - 1):
                connections  = get_lines_from_two_slices(pr_drusen_points, i )
                for c in connections:
                    p1 = c.get_s()
                    p2 = c.get_e()
                    plt.plot((p1.get_x(),p2.get_x()),(p1.get_y()*scale,p2.get_y()*scale),c = 'b')
                    
        for i in range(num_slices):
            for l in pr_drusen_points[i]:
                if(l.get_s().get_degree() == 1 and l.get_e().get_degree() == 1 ):
                    plt.plot((l.get_s().get_x(),l.get_e().get_x()),(l.get_s().get_y()*scale,l.get_e().get_y()*scale),c = 'b')
         
    plt.show(block)

def find_boundaries_in_binary_mask( mask ):
    a = sc.ndimage.binary_erosion( mask , iterations = 2)
    return mask - a
    
def binarize( mask ):
    mask[mask>0.0] = 1.0
    return mask
 

def find_area_between_seg_lines(label):
    h, w = label.shape
    label_area = np.copy(label)
    ls = np.sort(np.unique( label_area ))
    if( len(ls) >= 3):

        for j in range(w):
            col = label[:, j]
            l_1 = np.where( col == ls[1] )
            l_2 = np.where( col == ls[2])
            if(len(l_1[0]) != 0 and len(l_2[0]) != 0 ):
    
                label_area[l_1[0][0]:l_2[0][0], j] = 1
                label_area[l_2[0][0]:l_1[0][0], j] = 1
                
        # Replace all the labels with 1
        label_area[label_area > 0] = 1
       
        return label_area   
    return label
    
def remove_wrt_w_o_h( projection_height_mask, woh_t = 6, lwo_t = 0.25,size=0):
    mask = projection_height_mask > 0.0
    cca, numL = sc.ndimage.measurements.label( mask )
    bgL = get_label_of_largest_component(cca)
    mask = np.zeros(mask.shape,dtype='float')
    for l in np.unique(cca):
        if( l != bgL ):
            #show_images([cca, projection_height_mask, projection_height_mask*(cca==l)],1,3)
            max_h = np.max(projection_height_mask*(cca==l))
            y, x = np.where(cca==l)
            w = np.max(x) - np.min(x) + 1
            h = np.max(y) - np.min(y) + 1
            r1 = float(w)/float(max_h)
            r2 = float(h)/float(max_h)
            r3 = float(w)/float(h)
            if( (woh_t >= r3 and lwo_t <= r3 ) and  np.sum(cca==l)>size ):
                mask[y,x] = 1.0
    return mask

def remove_wrt_max_drusen_height( projection_height_mask, mask, height = 5 ):
    #mask = projection_height_mask > 0.0
    cca, numL = sc.ndimage.measurements.label( mask )
    bgL = get_label_of_largest_component(cca)
    mask = np.zeros(mask.shape,dtype='float')
    for l in np.unique(cca):
        if( l != bgL ):
            #show_images([cca, projection_height_mask, projection_height_mask*(cca==l)],1,3)
            max_h = np.max(projection_height_mask*(cca==l))
            
            if( max_h > height ):
                mask[cca==l] = 1.0
    return mask
    
def drusen_wise_otsu_thresholding( projection_image, mask ):
    
    cca, numL = sc.ndimage.measurements.label( mask )
    bgL = get_label_of_largest_component( cca )
    for l in np.unique( cca ):
        if( l != bgL ):
            m = cca==l
            m = sc.ndimage.binary_dilation(m, iterations=5)
            reg_i = projection_image[m]
            trshl = filters.threshold_otsu( reg_i )
            a = (projection_image>trshl)*(cca==l)
         #   show_images([projection_image, cca==l, a], 1,3)
            

   
def gaus(x,a,x0,sigma):
    
    return (a*np.exp(-(x-x0)**2/(2*sigma**2)))

def fix_special_cases(x, y, xl, yl, shape):
    resx = list(x)
    resy = list(y)
    resxl = list(xl)
    resyl = list(yl)
    
    img = np.zeros(shape)
    
    if( len(x) == 1 and x[0]>= 2 and x[0]<shape[1]-2):
        resy.append(y[0])
        resy.append(y[0])  
        resy.append(y[0])
        resy.append(y[0])
        resx.append(x[0]-1)
        resx.append(x[0]+1)
        resx.append(x[0]-2)
        resx.append(x[0]+2)
        
        resyl.append(yl[0])
        resyl.append(yl[0])  
        resyl.append(yl[0])
        resyl.append(yl[0]) 
        resxl.append(xl[0]-1)
        resxl.append(xl[0]+1)
        resxl.append(xl[0]-2)
        resxl.append(xl[0]+2)
    elif( len(x) == 2 and x[0]>= 3 and x[0]<shape[1]-3 ):
        resy.append(y[0])
        resy.append(y[1])  
        resy.append(y[0])
        resy.append(y[1])  
        resx.append(x[0]-1)
        resx.append(x[1]+1)
        resx.append(x[0]-2)
        resx.append(x[1]+2)
        
        resyl.append(yl[0])
        resyl.append(yl[1]) 
        resyl.append(yl[0])
        resyl.append(yl[1])  
        resxl.append(xl[0]-1)
        resxl.append(xl[1]+1)
        resxl.append(xl[0]-2)
        resxl.append(xl[1]+2)
    if( len(resx) > 2 ):
        print resx[np.argmin(resx)], resy[np.argmin(resx)]
        print resx[np.argmax(resx)], resy[np.argmax(resx)]
        # Bend the upper layer
        resy[np.argmin(resx)] = resy[np.argmin(resx)]+1 if resy[np.argmin(resx)]<shape[0] else shape[0]-1
        resy[np.argmax(resx)] = resy[np.argmax(resx)]+1 if resy[np.argmax(resx)]<shape[0] else shape[0]-1
    img[y,x] = 1.0
    img2 = np.zeros(shape)
    img2[resy, resx] = 1.0
   # show_images([img, img2], 1, 2)
    return resx, resy, resxl, resyl
        
def gaussian_fit_to_data( x, y, xl, yl, shape, slice_num = -1 ):
    if( False and  (slice_num == 42 or slice_num == 66 or slice_num == 74 or slice_num == 79 or slice_num == 100  )):
        x, y, xl, yl = fix_special_cases(x, y, xl, yl, shape)
    else:
        x, y, xl, yl = fix_special_cases(x, y, xl, yl, shape)
    x = np.asarray(x)
    y = np.asarray(y)
   
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    
    mu = x[np.argmin( y )]
   # print "mu:",mu
    new_x = np.arange((np.max(x)-np.min(x))*6+1)-3*(np.max(x)-np.min(x))+mu
    
    new_y = p(new_x).astype('int')
    max_new_y = np.min(new_y)
    max_y = np.min(y)
   # print "maxy:", max_y, " max newy:", max_new_y
    new_y = new_y - (max_new_y-max_y)
   # print "------------------"
   # print new_y, new_x
    xl = np.asarray(xl)
    yl = np.asarray(yl)
   
    z = np.polyfit(xl, yl, 2)
    p = np.poly1d(z)
    
   # mu = int((np.max(xl)-np.min(xl))/2.0)+np.min(xl)
    #new_xl = np.arange((np.max(xl)-np.min(xl))*2+1)-(np.max(xl)-np.min(xl))+mu
    new_xl = new_x
    new_yl = p(new_xl).astype('int')
    min_xl = xl[np.argmax(yl)]
    min_yl = np.max(yl)
    min_new_yl = int(p(min_xl))
    
    new_yl = new_yl - (min_new_yl-min_yl)
   # print "*********************"
   # print new_yl, new_xl
   # print "maxy:", min_yl, " max newy:", min_new_yl
    return np.asarray(new_y), np.asarray(new_x), np.asarray(new_yl), np.asarray(new_xl)
    
def fit_into_shape(x, y,xl,yl, shape):
    my = shape[0]
    mx = shape[1]
    new_x =[]
    new_y =[]
    for i in range(len(x)):
        if( x[i]>=0 and x[i]<mx and y[i]>=0 and y[i]<my):
            new_x.append(x[i])
            new_y.append(y[i])
    new_xl =[]
    new_yl =[]
    for i in range(len(xl)):
        if( xl[i]>=0 and xl[i]<mx and yl[i]>=0 and yl[i]<my):
            new_xl.append(xl[i])
            new_yl.append(yl[i])
    return new_y, new_x,new_yl, new_xl

def compute_extent_using_gaussian_fits( drusen_mask , slice_num = -1):
    ry, rx,by,bx,debug = fit_gaussian_to_drusen( drusen_mask , slice_num = slice_num)
    
    xs = -1
    xe = -1
    #if( debug == 5 ):
    #    print "Debug:----------------------a"
    #    print ry
    #    print rx
    #    print by
    #    print bx
        
   
    finding_start = True
    for i in range(len(rx)):
        if( ry[i] == by[i] ):
            
            if( finding_start ):
                xs = rx[i]
            else:
                xe = rx[i]
                break
        else:
            if( xs != -1 and finding_start ):
                finding_start = False
                
                
    #print "Debug:----------------------b"
    #print xs, xe 
    if( xs == -1 or xe == -1 ):
        y, x = np.where(drusen_mask>0.0)
        xs = np.min(x)
        xe = np.max(x)
    #print "Debug:----------------------c"
    #print xs, xe
    return xs, xe
    
    
def fit_gaussian_to_drusen( drusen_mask , slice_num=-1):
    y, x = np.where(drusen_mask>0.0)
    xy_top = dict()
    xy_bot = dict()
    # Find the upper curve of the drusen
    for i in range(len(x)):
        col = np.where(drusen_mask[:,x[i]]>0.0)        
        xy_top[x[i]] = np.min(col)
        xy_bot[x[i]] = np.max(col)
        
    dx = xy_top.keys()
    dy = xy_top.values()
    bx = xy_bot.keys()
    by = xy_bot.values()
    
    gy, gx, ly,lx = gaussian_fit_to_data( dx, dy, bx, by, drusen_mask.shape, slice_num )
    gy, gx,ly,lx = fit_into_shape(gx, gy,lx,ly, drusen_mask.shape)
    
    if( False and (slice_num == 42 or slice_num == 66 or slice_num == 74 or slice_num == 79 or slice_num == 100  )):
        show_image(drusen_mask)
        drusen_mask[gy, gx] = 2.0
        drusen_mask[ly, lx] = 3.0
        
        show_image(drusen_mask)
        return gy, gx,ly,lx,5
    # Return the estimated RPE and BM locations   
    return gy, gx,ly,lx,[]
    
def fit_gaussian_to_drusens( mask , slice_num=-1):
    cca, numL = sc.ndimage.measurements.label( mask )
    bgL = get_label_of_largest_component( cca )
    extent = np.zeros((mask.shape[1]))
    for l in np.unique(cca):
        if( l!=bgL ):
           # print "Drusen:"
            
            drusen_mask = (cca==l).astype('int')
            xs, xe = compute_extent_using_gaussian_fits( drusen_mask , slice_num=slice_num)
            
            extent[xs:xe+1] = 1.0
    
    if( False and (slice_num == 42 or slice_num == 66 or slice_num == 74 or slice_num == 79 or slice_num == 100  )):
        img = np.tile(extent, mask.shape[0]).reshape(mask.shape)
        show_images([mask, img],1,2)
        
    return extent
    
def get_label_from_projection_image_using_scale( orig_projection,scale_projection, labels ):
    h, w, s = labels.shape
    new_labels = np.empty((h,w,s))
    new_projection_mask = np.zeros((s,w))
    drusens = get_label_from_projection_image(orig_projection, labels)
    for i in range( s ):
        print "Slice_num:-----------", i
        label = labels[:,:,i]
        #drusen_mask = extract_drusens_from_line_segments( label )
        drusen_mask = drusens[:,:,i]
        filter1 = np.tile(scale_projection[i,:], h).reshape((h, w))
        new_mask = drusen_mask * filter1
        # Use the gaussian fit to complete drusen extent
        eg = fit_gaussian_to_drusens( new_mask , slice_num = i)
        filter2 = np.tile(eg, h).reshape((h, w))
        new_mask2 = drusen_mask * filter2
        eg2 = fit_gaussian_to_drusens( new_mask2 , slice_num = i)
        filter3 = np.tile(eg2, h).reshape((h, w))
        #layer_area = find_area_between_seg_lines( labels[:,:,i] )
        final_mask = drusen_mask * filter3
        new_labels[:,:,i] = final_mask
        if( False and (i == 42 or i == 66 or i == 74 or i == 79 or i == 100 )):
            print i
            show_images([drusen_mask,filter1, new_mask,filter2, new_mask2, filter3, final_mask], 4, 2)
        new_projection_mask[i,:] = np.max(final_mask, axis=0)
    #show_images([scale_projection, new_projection_mask],1,2)
    return new_labels, new_projection_mask
    
def put_mask_on_img( img, mask ):
    rgb_img = np.dstack((img,img,img))
    y,x = np.where(mask>0.0)
    rgb_img[y,x,0] = 255.0
    rgb_img[y,x,1] = rgb_img[y,x,1]*0.5
    rgb_img[y,x,2] = rgb_img[y,x,2]*0.5
    return rgb_img

def get_average_of_boundary_intensity( img, mask ):
    mask2 = sc.ndimage.binary_erosion( mask , iterations=1)
    mask3 = sc.ndimage.binary_dilation( mask2, iterations=2 )
    boundary_mask = mask3 - mask2
    if( np.sum(boundary_mask) == 0.0):
        boundary_mask = mask
    filtered = img*boundary_mask
   # show_images([img, mask,boundary_mask, filtered],2,2)
    
    return float(np.sum(filtered*boundary_mask))/float(np.sum(boundary_mask))
    
def use_component_wise_otsu_thresholding( height_mask, component_mask ):
    final_mask = np.zeros(component_mask.shape)
    cca, numL = sc.ndimage.measurements.label( component_mask )
    bgL = get_label_of_largest_component( cca )
    test = np.zeros(component_mask.shape)
    test[72:80,245:282] = 1.0
    for l in np.unique( cca ):
        if( l!=bgL ):
            avg = get_average_of_boundary_intensity( height_mask, cca==l )
            img = np.copy(height_mask)
            img[cca==l] = avg
            reg = sc.ndimage.binary_dilation(cca==l, iterations=5)
            y,x = np.where(reg>0.0)
            values = img[y,x]
            
            print avg
            th = filters.threshold_otsu( values )
            seg_mask = img>th
            component = seg_mask * reg
            final_mask[component>0.0] = 1.0
            print th
            #if( np.sum((cca==l)*test)>0.0):
             #   show_images([height_mask, cca==l, img, reg, img>th,final_mask],2,3)
  # show_images([height_mask, height_mask>0.0,component_mask, final_mask],2,2)
    return final_mask
    
def remove_false_positive_using_scale_image( projection, masks ):
    height_mask = find_drusen_in_stacked_slices( masks ).astype('float')

        
   # show_image(height_mask)
    return height_mask
    #show_image(height_mask)
    ii = ImageObj()
    scale_img, e_val, e_vec, mask_on = ii.vesselness_measure( height_mask, sigma_list=[0.1,0.2,0.3,0.4,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], general_sigma=0.01, rho=0.0, c_var=[],\
        alpha_var=0.0, beta_var=np.inf, theta_var=0.1, crease_type = "r",\
        postprocess = False, detect_planes=False, hx = 1.0, hy = 1.0, hz = 1.0,\
        RETURN_HESSIAN = False, ignore_boundary=False, BOUNDARY_CONDITION='None')
        
    
    #show_images([height_mask,scale_img, scale_img>0.0],1,3)
    smask = scale_img>0.0
    component_mask = smask * (height_mask>0.0)
    new_projection_mask = use_component_wise_otsu_thresholding( height_mask, component_mask=component_mask )    
   # new_labels, new_projection_mask = get_label_from_projection_image_using_scale( height_mask>0.0,smask, masks )
  #  show_images([height_mask,put_mask_on_img(projection, height_mask>0.0),\
 #           put_mask_on_img(projection, smask),put_mask_on_img(projection, new_projection_mask)],2,2)
            
    return  new_projection_mask
    
def remove_false_positives( projection_image, seg_imgs, intensity_t = 5.0, size=3, useWarping=True ): 
    height_mask = find_drusen_in_stacked_slices( seg_imgs,useWarping=useWarping )
    
    mask = (height_mask>0.0).astype('int')
    isHighRes = seg_imgs.shape[2]>70
    if( isHighRes ):
        mask2= remove_drusen_with_1slice_size( np.copy(mask) )
        m1 = np.copy(mask2)
        mask *= mask2
        mask = remove_wrt_w_o_h(height_mask*mask2, woh_t = 20, lwo_t = 0.0,size=size) 
        m2 = np.copy(mask)
        mask = (mask>0.0).astype('float')
        mask = remove_non_bright_spots( projection_image, mask, intensity_t)
        m3 = np.copy(mask)
        gauss_sig   = 0.5
        intensity_t = 0.1
        
        mask = sc.ndimage.filters.gaussian_filter(mask , gauss_sig) > intensity_t
      #  mask = remove_wrt_max_drusen_height( height_mask, mask, height = 5 )
    else:
       
        mask = remove_wrt_w_o_h(height_mask, woh_t = 20*7, lwo_t = 0.0,size=size) 
        m1 = np.copy(mask)
        mask = (mask>0.0).astype('float')
        mask = remove_non_bright_spots( projection_image, mask, intensity_t)
        m2 = np.copy(mask)
        gauss_sig   = 0.25
        intensity_t = 0.05
        
        mask = sc.ndimage.filters.gaussian_filter(mask , gauss_sig) > intensity_t
        m3 = np.copy(mask)
       # mask = remove_wrt_max_drusen_height( height_mask, mask, height = 5 )
    
    #show_images([height_mask,put_mask_on_img(projection_image, height_mask>0.0),\
     ##       put_mask_on_img(projection_image, mask),\
       #     put_mask_on_img(projection_image, m1),put_mask_on_img(projection_image, m2),\
        #    put_mask_on_img(projection_image, m3)],3,2)
    
   # drusen_wise_otsu_thresholding( sc.ndimage.filters.gaussian_filter(projection_image,1.0), mask)
    return mask  
    
def draw_drusen_boundary_over_projection_image( b_scans, gts, prs=[], show = False , scale=1, input_type='line_segments'):
    useWarping = True
    print "1",useWarping
    k = produce_drusen_projection_image( b_scans, gts, useWarping=useWarping )
    k /= np.max(k) if np.max(k) != 0.0 else 1.0
    
    projection_image = sc.misc.imresize(k, (k.shape[0]*scale,k.shape[1])).astype('float')
    projection_image /= np.max(projection_image) if np.max(projection_image) != 0.0 else 1.0
    
    
    k = remove_false_positives(projection_image,gts,intensity_t=0.0,useWarping=useWarping)
    #k = (find_drusen_in_stacked_slices( gts )>0.0).astype('float')
   # k2= remove_drusen_with_1slice_size( np.copy(k) )
   # k3 = remove_wrt_max_drusen_height( height_mask, k2, height = 5 )
 #   k = sc.ndimage.filters.gaussian_filter(k2.astype('float') , 0.3) > 0.1
    k = k.astype('float')
    #print k.shape
    
  #  show_image(k)
    #show_images([scale,k,k2,k3],2,2)
   # th = filters.threshold_otsu(projection_image.astype('float').ravel())
    #print th

    #show_images([projection_image, projection_image>th], 1, 2)
   # k = remove_false_positives(projection_image,gts)
  # k = find_drusen_in_stacked_slices( gts )
  #  k = (k>0.0).astype('int')
    #footprint = np.asarray([[1,1,1],[1,1,1],[1,1,1]])
    #tt =  sc.ndimage.filters.maximum_filter( kp.astype('float') , 7)
  #  show_images([tt, kp*tt, kp2, projection_image], 2, 2)
    
    mm = get_label_from_projection_image(k, gts, method='rpeToNormRpe',useWarping=useWarping)
    
   # save_array("/home/gorgi/Desktop/DataFromUniClinic/test-05-04-2017/test/",b_scans,gts,mm)
    #show_image(projection_image)
    #show_images([k,k2],1,2)

    gts_extent   = binarize(sc.misc.imresize(k, (k.shape[0]*scale,k.shape[1]),interp='nearest' ))
    
    gts_boundary = find_boundaries_in_binary_mask( gts_extent )
   # save_array("/home/gorgi/Desktop/DataFromUniClinic/test-05-04-2017/test23/",b_scans,gts,mm,useWarping=useWarping)
    if( show ):
        rgb_img = np.empty((projection_image.shape[0],projection_image.shape[1], 3), dtype='float')
        
        rgb_img[:,:,0] = projection_image
        rgb_img[:,:,1] = projection_image 
        rgb_img[:,:,2] = projection_image 
    
        rgb_img = (rgb_img/np.max(rgb_img))*0.8      
        
        rgb_img[gts_boundary == 1.0,0] = 1.0
        rgb_img[gts_boundary == 1.0,1] *= 0.5
        rgb_img[gts_boundary == 1.0,2] *= 0.5
    
    if( prs != [] ):
       
        k = remove_false_positives( projection_image, prs ,intensity_t=0,useWarping=True)
        #k = find_drusen_in_stacked_slices( prs , input_type=input_type)
        #k = remove_non_bright_spots( projection_image, k )
        prs_extent = binarize(sc.misc.imresize(k, (k.shape[0]*scale,k.shape[1]), interp='nearest'))
        
        prs_boundary = find_boundaries_in_binary_mask( prs_extent )
        mm = get_label_from_projection_image(k, prs, method='rpeToNormBm')
        
        if( show ):
            
            rgb_img[prs_boundary == 1.0,0] *= 0.5
            rgb_img[prs_boundary == 1.0,1] *= 0.5
            rgb_img[prs_boundary == 1.0,2] = 1.0
            
    if( show ):
        show_image_rgb(rgb_img)  
        
    if( prs != [] ):
        return gts_extent, prs_extent
        
    return gts_extent
    
def produce_drusen_projection_image( b_scans, gts, use_BM_as_lower_layer=False, useWarping=False ):
    b_scans = b_scans.astype('float')
    projection = np.zeros((b_scans.shape[2], b_scans.shape[1]))
    projection2 = np.zeros((b_scans.shape[2], b_scans.shape[1]))
    total_y_max = 0
    max_i = 0
    img_max = np.zeros(b_scans[:,:,0].shape)
    for i in range(b_scans.shape[2]):
        b_scan = b_scans[:,:,i]
        gt     = gts[:,:,i]
       # warp_BM( gt )
        if( np.sum(gt)==0.0):
            continue
        
       # print i,"==================================",i
      #  show_images([b_scan,gt],1,2)
        y, x     = get_RPE_layer( gt )        
        y_n, x_n = normal_RPE_estimation( gt,useWarping=useWarping )
        vr = np.zeros((b_scans.shape[1]))
        vr[x] = y
        vn = np.zeros((b_scans.shape[1]))
        vn[x_n] = y_n
        
        #show_image(img)
        y_diff   = np.abs(y-y_n)
        #y_diff[vr==0] = 0
        #y_diff[vn==0] = 0
        y_max    = np.max(y_diff)
        if( total_y_max < y_max ):
         
            img_max.fill(0)
            img_max[y,x]=255
            img_max[y_n,x_n]=127
            total_y_max = y_max
            max_i = i
   
            
            
   

    print max_i, total_y_max
   # show_images([gts[:,:,max_i], img_max],1,2)
    for i in range(b_scans.shape[2]):
        b_scan = b_scans[:,:,i]
        gt     = gts[:,:,i]
        if( np.sum(gt) == 0.0):
            continue
        n_bscan  = np.copy(b_scan)
        y, x     = get_RPE_layer( gt )        
        y_n, x_n = normal_RPE_estimation( gt, useWarping=useWarping )
        y_b, x_b = get_BM_layer( gt )
       
        y_max    = total_y_max
        upper_y  = (y_n - y_max)
        c = 0
    
        #if(upper_y[c]==-1):
        #    gt[y_n, x_n] = 70
        #    show_image(gt)
        '''
        bm = np.zeros(b_scan.shape)
        bm[y_b, x_b] = 1.0
        bm[y_n, x_n] = 2.0
        bm[y, x] = 3.0
        show_image(bm)
        '''
        for ix in x:
           # print upper_y[c],y_n[c],ix
            n_bscan[y[c]:y_n[c],ix] = np.max(b_scan[upper_y[c]:y_n[c],ix])
          
            projection[i,ix] =  np.sum(n_bscan[upper_y[c]:y_n[c]+1,ix])
            projection2[i,ix] =  np.sum(n_bscan[upper_y[c]:y[c]+1,ix])*0.5 + np.sum(n_bscan[y[c]:y_n[c]+1,ix])
            c += 1
        
    '''      
    

    rgb_projection = sc.misc.imresize(rgb_projection,(512,512,3),interp='bilinear')
 
    show_image_rgb(rgb_projection)
   # show_image(projection)
#        show_image(label)
    '''
    #show_images([projection.astype('float'),projection2.astype('float')], 1, 2)
    #show_image( sc.misc.imresize(projection,(145*2,512),interp='bilinear') )
    return projection.astype('float')
    
def find_drusen_in_stacked_slices( b_scans ,  input_type = 'line_segments',useWarping=True):
    drusen_extent = np.zeros((b_scans.shape[2], b_scans.shape[1]))
 
    for i in range(b_scans.shape[2]):
        v = find_drusen_extent_in_one_b_scan( b_scans[:,:,i], slice_num=i, input_type= input_type,useWarping=useWarping)
        drusen_extent[i,:] = v
    
    return drusen_extent

def extract_drusens_from_line_segments( seg_img, slice_num=-1, input_type = 'line_segments',useWarping=True):
    if( input_type == 'line_segments' ):
        # Find drusen mask       
        drusen_mask = compute_drusen_mask( seg_img ,useWarping=useWarping)
        drusen_mask = filter_drusen_by_size( drusen_mask, slice_num)
        
    elif( input_type == 'drusen'):
        drusen_mask = seg_img
    a = np.zeros(seg_img.shape)
    print "3",useWarping
    a[drusen_mask>0] = 2
    a[seg_img>0] = 1
#    show_images([a],1,1)
    return drusen_mask
    
def find_drusen_extent_in_one_b_scan( seg_img, slice_num=-1, input_type = 'line_segments',useWarping=True):
    
    if( input_type == 'line_segments' ):
        # Find drusen mask       
        drusen_mask = compute_drusen_mask( seg_img ,useWarping=useWarping)
        drusen_mask = filter_drusen_by_size( drusen_mask, slice_num)
        
    elif( input_type == 'drusen'):
        drusen_mask = seg_img
        
        
    #shekoufeh
  #  if(slice_num>50 and slice_num<53):
  #      show_images( [seg_img,drusen_mask],1,2 )  
    #fit_gaussian_to_drusens( drusen_mask )
    '''
    # Compute connected component analysis    
    cca, num_drusen = sc.ndimage.measurements.label( drusen_mask )
    bg_label = get_label_of_largest_component( cca )
    labels   = np.unique(cca)
    vector = np.zeros(seg_img.shape[1])
    
    for l in labels:
        if( l != bg_label ):
            y, x = np.where( cca == l )
            vector[np.min(x):np.max(x)] = 1.0
    '''
    drusen_mask  = drusen_mask.astype('int')
    cvr_h = np.sum( drusen_mask, axis = 0 )       
    return cvr_h

def get_outer_boundary( mask, size ):
    a = sc.ndimage.binary_dilation(mask, iterations = size)
    return a.astype('float') - mask.astype('float')
    
def remove_non_bright_spots( projection_img, drusen_mask, threshold = 4 ):   
    projection_img = (projection_img.astype('float')-np.min(projection_img))/\
                     (np.max(projection_img)-np.min(projection_img))* 255.0
    cca, num_lbls = sc.ndimage.measurements.label( drusen_mask )
    bg_lbl = get_label_of_largest_component( cca )
    labels = np.unique( cca )
    res_m  = np.zeros( drusen_mask.shape, dtype='float' )
    
    for l in labels:
        if( l != bg_lbl ):
            reg_mask = cca == l
            vreg = projection_img[reg_mask>0.0]
            vreg = vreg[vreg>0.0]
            boundary = get_outer_boundary( reg_mask , size = 2 )
            vbnd = projection_img[boundary>0.0]
            vbnd = vbnd[vbnd>0.0]
            reg_mean = np.mean( vreg )
            
            bnd_mean = np.mean( vbnd )
            
            #print reg_mean, bnd_mean, reg_mean - bnd_mean
      #      show_images([reg_mask, boundary], 1,2)
            # If the mean intensity of the spot is larger by a threshold than the 
            # region then keep this spot, otherwise remove it
            if( reg_mean - bnd_mean > threshold ):
                print "======================================"
                res_m[cca==l] = 1.0
                
                print int(np.mean(np.where(cca==l)[0])), int(np.mean(np.where(cca==l)[1])), int(reg_mean),int( bnd_mean)
                #print vreg
                print "---------"
                #print vbnd
                print "======================================"
               # show_images([reg_mask, boundary, reg_mask*projection_img, boundary*projection_img], 2,2)
    return res_m
    
def find_drusen_boundary_in_one_b_scan( seg_img, slice_num=0, input_type = 'line_segments',useWarping=True):
    
    if( input_type == 'line_segments' ):
     
        # Find drusen mask       
        drusen_mask = compute_drusen_mask( seg_img,useWarping=useWarping )
        drusen_mask = filter_drusen_by_size( drusen_mask, slice_num)
    elif( input_type == 'drusen'):
        drusen_mask = seg_img
    #show_image( drusen_mask )
    # Compute connected component analysis    
    cca, num_drusen = sc.ndimage.measurements.label( drusen_mask )
    bg_label = get_label_of_largest_component( cca )
    labels   = np.unique(cca)
    vector = np.zeros(seg_img.shape[1])
    lines = []
    for l in labels:
        if( l != bg_label ):
            y, x = np.where( cca == l )

            p_s = Point(np.min(x), slice_num)
            p_e = Point(np.max(x), slice_num)
            vector[np.min(x):np.max(x)] = 1.0
            lines.append(Line(p_s, p_e))
    return lines, vector
    
def get_max_id( names ):
    max_id = -1
    for i in range( len(names)):
        l_id = int(names[i].split('-')[0])
        if( l_id>max_id ):
            max_id = l_id
            
    return max_id
    
def save_array(path, b_scans, line_segs, drusens, method="rpeToNormRpe", useWarping=True ):
    
    for i in range(b_scans.shape[2]):
        img = np.dstack((b_scans[:,:,i],b_scans[:,:,i],b_scans[:,:,i]))
        y, x = np.where( drusens[:,:,i] > 0.0 )
        img[y, x,0] = 255.0
        img[y, x,1] = 0.0
        img[y, x,2] = 0.0
        
        yr, xr = get_RPE_location( line_segs[:,:,i] )
        if( method == "rpeToNormRpe"):
            yb, xb = normal_RPE_estimation(line_segs[:,:,i], useWarping=useWarping)
        elif( method == "rpeToNormBm"):
            yb, xb = get_BM_location( line_segs[:,:,i] )
    
        img[yb, xb,1] = 255.0
        img[yb, xb,0] = 0.0
        img[yb, xb,2] = 0.0
        img[yr, xr,2] = 255.0
        img[yr, xr,0] = 0.0
        img[yr, xr,1] = 0.0
        #print np.unique(line_segs[:,:,i] ),len(yb), len(xb), len(yr), len(xr)
        #show_image_rgb( img )
        misc.imsave(path+str(i+1)+".png", img)
def is_bm_always_below_rpe( label_path ):
    lbl=io.imread( label_path )
    mr = np.zeros(lbl.shape)
    mb = np.zeros(lbl.shape)
    mr[get_RPE_location(lbl)] = 1
    mb[get_BM_location(lbl)]  = 1
    for i in range(lbl.shape[1]):
        colr = mr[:,i]
        colb = mb[:,i]
        if(np.sum(colr) > 0 and np.sum(colb)>0):
            if( np.min(np.where(colr>0)) - np.max(np.where(colb>0)) > 5 ):
               # show_image(lbl)
                return False
    return True
def read_bscan_and_gt(  read_from,  image_list ):   
    img = io.imread( read_from + image_list.keys()[0] )
    max_id = get_max_id(image_list.keys())
    b_scan = np.zeros((img.shape[0], img.shape[1], max_id))
    labels = np.zeros(b_scan.shape)
    i = 0
    for k, v in image_list.iteritems():
        scan_id = int(k.split('-')[0])-1
        print scan_id, k
        b_scan[:,:,scan_id] = io.imread(read_from + k )
        labels[:,:,scan_id] = io.imread(read_from + v )
        i += 1
  #  show_image(b_scan[:,:,109])
    return b_scan, labels

def get_drusens_from_rpe_to_bm(mask,useWarping=True):

    l_area = find_area_between_seg_lines( mask )
    d_area = compute_drusen_mask(mask,useWarping=useWarping)
    d_area = filter_drusen_by_size(d_area)
    d_mask = np.tile(np.sum(d_area, axis=0)>0, mask.shape[0]).reshape(mask.shape)    

    return l_area * d_mask
   
def get_label_from_projection_image(projected_labels, labels, method='rpeToNormRpe',useWarping=True):
    print "2",useWarping
    lbls = np.copy(labels)
    for i in range( labels.shape[2] ):
        valid_drusens = np.tile(projected_labels[i, :], labels.shape[0]).reshape(labels.shape[0],labels.shape[1])
        if( method=='rpeToNormBm' ):
            l_area = find_area_between_seg_lines( lbls[:,:,i] )
        if( method=='rpeToNormRpe'):
            l_area = extract_drusens_from_line_segments(lbls[:,:,i] ,useWarping=useWarping)
        lbls[:,:,i] = l_area * valid_drusens
    return lbls

def save_b_scans_and_labels( save_to, image_list, b_scans, new_labels ):
    i = 0
   # show_image(b_scans[:,:,109])
    for k, v in image_list.iteritems():
        scan_id = int(k.split('-')[0])-1
        print scan_id, k
        #print i
        #show_images([b_scans[:,:,i],new_labels[:,:,i] ], 1,2 )
        misc.imsave(save_to+k, b_scans[:,:,scan_id])
        misc.imsave(save_to+v, new_labels[:,:,scan_id])
        i += 1
        
def convert_ground_truth_from_line_seg_to_drusen( read_from,  image_list, save_to):
    if( len(image_list.keys())==0 ):
        return
    b_scans, labels = read_bscan_and_gt( read_from, image_list )
   # print b_scans.shape, labels.shape
    labels_extent  = draw_drusen_boundary_over_projection_image( b_scans, labels,scale= 1, input_type='line_segments' )
    #show_image(labels_extent)
  
    dru_labels     = get_label_from_projection_image(labels_extent, labels, method='rpeToNormBm')
    save_b_scans_and_labels( save_to, image_list, b_scans, dru_labels )


###############################################################################
############################### Chen's Method #################################
###############################################################################
rpe_scan_list  = []
nrpe_scan_list = []

def draw_lines_on_mask(rpe , nrpe, shape ):
    mask = np.zeros(shape)
    if(rpe==None):
        rpe=[]
    if(nrpe==None):
        nrpe=[]
    
    if(len(rpe)==0):
        return mask
    
    rpearr = np.asarray(rpe)
    nrpearr = np.asarray(nrpe)
    checkRpe = np.abs(rpearr)
    checkNrpe = np.abs(nrpearr)
    if(  np.array_equal(rpearr,checkRpe) and np.array_equal(nrpearr,checkNrpe)):
        mask[rpe[:,1].astype('int'),rpe[:,0].astype('int')] += 1.0
        mask[nrpe[:,1].astype('int'),nrpe[:,0].astype('int')] += 2.0
    return mask
    
def find_area_btw_RPE_normal_RPE( mask ):
    area_mask = np.zeros(mask.shape)
    for i in range( mask.shape[1] ):
        col = mask[:,i]
        v1  = np.where(col==1.0)
        v2  = np.where(col==2.0)
        v3  = np.where(col==3.0)
       
        v1 = np.min(v1[0]) if len(v1[0]) > 0  else -1
        v2 = np.max(v2[0]) if len(v2[0]) > 0  else -1
        v3 = np.min(v3[0]) if len(v3[0]) > 0  else -1
        
        if( v1 >= 0 and v2 >= 0 ):
            area_mask[v1:v2,i] = 1
    return area_mask
        
def seg_chen( inputImage ):
    
    filter1=FF.FilterBilateral(inputImage)
    threshold=FF.computeHistogrammThreshold(filter1)
    RNFL=FF.getTopRNFL(filter1,threshold,False)
    mask=FF.thresholdImage(filter1,threshold)
    if( len(RNFL) ==0):
        return [],[]
    mask2=FF.removeTopRNFL(filter1,mask,RNFL)
    centerLine1=FF.getCenterLine(mask2)
    itterations=3;
    dursenDiff=5;
    '''
    print "ImageShape:", inputImage.shape
    print "ImageVal:", np.unique(inputImage)
    print "filtershape:", filter1.shape
    print "centerlineshape:",centerLine1.shape
    '''
    idealRPE=FF.getIdealRPELine(filter1,centerLine1,itterations,dursenDiff)

    return centerLine1, idealRPE
    
def fill_inner_gaps( layer ):
        
    d_layer = dict(layer)
        
    prev = -1
    for i in range(np.max(d_layer.keys())):
        if( not i in d_layer.keys() and prev!=-1):
            d_layer[i] = prev
            
        if( i in d_layer.keys() ):
            prev = d_layer[i]
    return np.asarray([d_layer.keys(),d_layer.values()]).T
    
def get_connected_rpe_line_seg_chen( rpe, nrpe, shape ):
    mask = draw_lines_on_mask(rpe, nrpe, shape)
    mask[mask==2] = 0.0
    mask[mask>0.0] = 1.0
    preMask = np.copy(mask)
    for j in range(mask.shape[1]):
        cuCol = mask[:,j]
        neCol = mask[:,min(j+1,mask.shape[1]-1)]
        
        cuCol = np.where(cuCol>0)[0]
        neCol = np.where(neCol>0)[0]
        
        if( len(cuCol) > 0 and len(neCol) > 0 ):
            mask[np.min(cuCol):np.max(neCol), j] = 1.0
            mask[np.max(neCol):np.min(cuCol), j] = 1.0
            
  #  show_images([preMask, mask],1,2)
    return np.where(mask>0)
def produce_drusen_projection_image_chen( b_scans ):
    b_scans = b_scans.astype('float')
    projection = np.zeros((b_scans.shape[2], b_scans.shape[1]))
    projection2 = np.zeros((b_scans.shape[2], b_scans.shape[1]))
    total_y_max = 0
    max_i = 0
    img_max = np.zeros(b_scans[:,:,0].shape)
    for i in range(b_scans.shape[2]):
     #   if( i > 5 ):
     #       break
        b_scan = b_scans[:,:,i]
        b_scan = (b_scan - np.min(b_scan))/(np.max(b_scan)-np.min(b_scan)) if len(np.unique(b_scan))>1 else np.ones(b_scan.shape)
       # show_image(b_scan)
        rpe  = rpe_scan_list[i]
        nrpe = nrpe_scan_list[i]
        mask = draw_lines_on_mask( rpe, nrpe, b_scan.shape )
        area_mask = find_area_btw_RPE_normal_RPE( mask )
        y_diff = np.sum(area_mask, axis=0)
        y_max    = np.max(y_diff)
        if( total_y_max < y_max ):
            rpe = rpe.astype('int')
            nrpe = nrpe.astype('int')
            img_max = np.copy(b_scan)
            img_max[nrpe[:,1],nrpe[:,0]] = 0.5
            img_max[rpe[:,1],rpe[:,0]] = 1.0
            kk = nrpe[:,1]-y_max
            img_max[kk.astype('int'),nrpe[:,0]] = 1.0
            total_y_max = y_max
            max_i = i
            area_mask_max = np.copy(area_mask)
    print max_i, total_y_max
   # show_images([b_scan, img_max,area_mask_max],1,3)
    for i in range(b_scans.shape[2]):
        b_scan = b_scans[:,:,i]
        b_scan = (b_scan - np.min(b_scan))/(np.max(b_scan)-np.min(b_scan)) if len(np.unique(b_scan))>1 else np.ones(b_scan.shape)
        n_bscan  = np.copy(b_scan)
        rpe = rpe_scan_list[i]
        nrpe = nrpe_scan_list[i]

        rpe = fill_inner_gaps(rpe)

        rpe = rpe.astype('int')
        nrpe = nrpe.astype('int')
        drpe  = dict(rpe)
        dnrpe = dict(nrpe)
        upper_y  = np.copy(nrpe)        
        y_max    = total_y_max
        upper_y[:,1]  = (upper_y[:,1] - y_max)
        durpe = dict(upper_y)
        
        for ix in range(b_scan.shape[1]):
            if( (ix in drpe.keys()) and (ix in dnrpe.keys())):
                n_bscan[drpe[ix]:dnrpe[ix],ix] = np.max(b_scan[durpe[ix]:dnrpe[ix],ix])
                projection[i,ix] =  np.sum(n_bscan[durpe[ix]:dnrpe[ix]+1,ix])            
                projection2[i,ix] =  np.sum(n_bscan[durpe[ix]:drpe[ix]+1,ix])*0.5 + np.sum(n_bscan[drpe[ix]:dnrpe[ix]+1,ix])
            
        n_bscan[upper_y[:,1].astype('int'),nrpe[:,0].astype('int')] = 1
        n_bscan[rpe[:,1].astype('int'),rpe[:,0]] = 1
        n_bscan[nrpe[:,1].astype('int'),nrpe[:,0].astype('int')] = 0.5
       # if( max_i == i):
       #     show_images([b_scan, n_bscan],  1,2)
    return projection.astype('float')
    
def find_drusen_in_stacked_slices_chen( b_scans ):
    hmask = np.zeros((b_scans.shape[2], b_scans.shape[1]))
    for i in range(b_scans.shape[2]):
        b_scan = b_scans[:,:,i]
        rpe  = rpe_scan_list[i]
        nrpe = nrpe_scan_list[i]
        mask = draw_lines_on_mask(rpe, nrpe, b_scan.shape)
        area_mask = find_area_btw_RPE_normal_RPE( mask )
        hmask[i,:] = np.sum(area_mask, axis=0)
    return hmask
    
def remove_false_positives_chen( projection_image, b_scans, intensity_t = 15.0, size=10 ): 
    height_mask = find_drusen_in_stacked_slices_chen( b_scans )
    
    mask = (height_mask>0.0).astype('int')
    isHighRes = b_scans.shape[2]>70
    if( isHighRes ):
        mask2= remove_drusen_with_1slice_size( np.copy(mask) )
        m1 = np.copy(mask2)
        mask *= mask2
        mask = remove_wrt_w_o_h(height_mask*mask2, woh_t = 6, lwo_t = 0.0,size=size) 
        m2 = np.copy(mask)
        mask = (mask>0.0).astype('float')
        mask = remove_non_bright_spots( projection_image, mask, intensity_t)
        m3 = np.copy(mask)
        gauss_sig   = 0.5
        intensity_t = 0.1
        
        mask = sc.ndimage.filters.gaussian_filter(mask , gauss_sig) > intensity_t
      #  mask = remove_wrt_max_drusen_height( height_mask, mask, height = 5 )
    else:
       
        mask = remove_wrt_w_o_h(height_mask, woh_t = 20*7, lwo_t = 0.0,size=size) 
        m1 = np.copy(mask)
        mask = (mask>0.0).astype('float')
        mask = remove_non_bright_spots( projection_image, mask, intensity_t)
        m2 = np.copy(mask)
        gauss_sig   = 0.25
        intensity_t = 0.05
        
        mask = sc.ndimage.filters.gaussian_filter(mask , gauss_sig) > intensity_t
        m3 = np.copy(mask)
    return mask 
    
def delete_rpe_nrpe_lists(  ):
    del rpe_scan_list[:]
    del nrpe_scan_list[:]  
def initialize_rpe_nrpe_lists( b_scans ):
    
    for i in range(b_scans.shape[2]):
        print "#######################:",i
        #show_image(b_scans[:,:,i])
        rpe, nrpe = seg_chen(b_scans[:,:,i])
        rpe_scan_list.append(rpe)
        nrpe_scan_list.append(nrpe)
        

def save_array_chen(b_scans, projectionMask, savePath):
    for i in range(b_scans.shape[2]):
        img = np.dstack((b_scans[:,:,i],b_scans[:,:,i],b_scans[:,:,i]))
        
        b_scan = b_scans[:,:,i]
        rpe  = rpe_scan_list[i]
        nrpe = nrpe_scan_list[i]
        mask = draw_lines_on_mask(rpe, nrpe, b_scan.shape)
        area_mask = find_area_btw_RPE_normal_RPE( mask )        
        truePosMask = np.tile(projectionMask[i,:],b_scan.shape[0]).reshape(b_scan.shape)
        area_mask *= truePosMask
        y, x = np.where( area_mask > 0.0 )
        img[y, x,0] = 255.0
        img[y, x,1] = 0.0
        img[y, x,2] = 0.0
        
        yr, xr = get_connected_rpe_line_seg_chen( rpe, nrpe, b_scan.shape )
        
        yb, xb = np.where(mask==2)
       # yr, xr = np.where(mask==1)
        yn, xn = np.where(mask==3)
        
        img[yb, xb,1] = 255.0
        img[yb, xb,0] = 0.0
        img[yb, xb,2] = 0.0
        
        img[yn, xn,2] = 255.0
        img[yn, xn,0] = 0.0
        img[yn, xn,1] = 0.0
        
        img[yr, xr,2] = 255.0
        img[yr, xr,0] = 0.0
        img[yr, xr,1] = 0.0  
        
        misc.imsave(savePath+str(i)+".png", img)
def run_chen_method(b_scans,savePath=""):
    initialize_rpe_nrpe_lists(b_scans)
    
    projection = produce_drusen_projection_image_chen( b_scans )
    projection /= np.max(projection) if np.max(projection) != 0.0 else 1.0
    
    
    mask = remove_false_positives_chen(projection, b_scans)
    boundaries = find_boundaries_in_binary_mask(mask)
    rgb_img = np.empty((projection.shape[0],projection.shape[1], 3), dtype='float')
        
    rgb_img[:,:,0] = projection
    rgb_img[:,:,1] = projection 
    rgb_img[:,:,2] = projection 
    
    rgb_img[boundaries == 1.0,0] = 1.0
    rgb_img[boundaries == 1.0,1] = 0.0
    rgb_img[boundaries == 1.0,2] = 0.0
    save_array_chen(b_scans, mask,savePath)
    show_images([ rgb_img],1,1)
    delete_rpe_nrpe_lists()
    
def run_chen_method_on_a_pack(b_scans):
    h, w, d = b_scans.shape
    initialize_rpe_nrpe_lists(b_scans)
    masks = np.zeros(b_scans.shape)
    for i in range(b_scans.shape[2]):
        
        mask = draw_lines_on_mask(rpe_scan_list[i], nrpe_scan_list[i],(h,w))
        masks[:,:,i] = find_area_btw_RPE_normal_RPE(mask)
        #show_images([b_scans[:,:,i], draw_lines_on_mask(rpe,nrpe,b_scans[:,:,i].shape)],1,2)
    #print len(rpe_scan_list), len(nrpe_scan_list)
    delete_rpe_nrpe_lists()
    #print len(rpe_scan_list), len(nrpe_scan_list)
    #exit()
    return masks
    
def get_subject_ids_to_evaluate(path):
    subjects = set()
    with open( path ) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    for c in content:
        subjects.add(int(c.split(',')[0]))
    return subjects    
    
def compute_IoU(gt, gtDru, pr, gtType = 'area', label=1):
    if( gtType == 'area'):
        mGt = find_area_between_seg_lines(gt)
        mPr = find_area_between_seg_lines(pr)
    elif(gtType=='drusenNrpe'):    
        mGt = remove_nrpe_to_bm_part_from_gt(gt, gtDru)
        mGt = filter_drusen_by_size(mGt)
        mPr = pr
        mPr = filter_drusen_by_size(mPr)
    elif(gtType=='chen'):
        mGt = remove_nrpe_to_bm_part_from_gt(gt, gtDru)
        mGt = filter_drusen_by_size(mGt)
        mPr = pr
    else:
        mGt = gt
        mPr = pr
  
    mGt = mGt.astype('float')
    mPr = mPr.astype('float')
    mGt[mGt>0]  = 1.0
    mGt[mGt<=0] = 0.0
    mPr[mPr>0]  = 1.0
    mPr[mPr<=0] = 0.0
    
    mGt = np.abs(mGt - (1.0-label))
    mPr = np.abs(mPr - (1.0-label))
    
   # show_images([mGt,mPr],1,2) 
    
    sim = np.logical_and(mGt , mPr).astype('int')    
    intersection  = np.sum(sim)
    dnum = np.logical_or(mGt, mPr).astype('int')
    union = np.sum(dnum)
    # Intersection Over Union
    IoU = float(intersection)/float(union) if union > 0.0 else 1.0
    print IoU
  #  show_images([mGt,mPr],1,2)
    '''
    
    false_positive = np.sum((mGt.astype('int')-mPr.astype('int'))==-1)
    neg_mGt = np.logical_not(mGt)
    neg_mPr = np.logical_not(mPr)    
    neg_sim = np.logical_and(neg_mGt , neg_mPr)*1.0
    true_negative  = np.sum(neg_sim)
    false_negative = np.sum((neg_mGt.astype('int')-neg_mPr.astype('int'))==-1)
    
    IoU = float(true_positive)/float(true_positive+false_positive+false_negative) if float(true_positive+false_positive+false_negative) != 0 else 1.0
    print "IoU = ",IoU
    print "-------"
    print float(true_positive)
    print float(true_positive+false_positive+false_negative)
    print float(true_positive+false_positive)
    print "OR=",compute_OR(gt, pr)
    print "-------"
    show_images([mGt, mPr, sim,(mGt.astype('int')-mPr.astype('int'))==-1,neg_sim,(neg_mGt.astype('int')-neg_mPr.astype('int'))==-1],2,3,["GT", "PR","TP","FP","TN","FN"]) 
    '''
    
    
    
    return IoU
   
def total_evaluation( path, fold_type, measureName, save_path ):
    folders = [f for f in listdir(path) if isfile(join(path, f))]
    counter = 0
    save_path += (fold_type+'/')
    measure = list()
    for d1 in os.listdir(path):
       # if( d1 == 'Fold04' ):
       #     continue
        fileName = path+'/'+d1+'/'+fold_type+'/test-evaluation/'+measureName+'.pkl'
        print fileName
        data = read_pickle_data(fileName)
        print len(data)
        measure.extend(data)
        
    print "===========>>",len(measure)
    write_pickle_data( save_path+measureName , measure)
    #print measure
    a = np.mean(np.asarray(measure))
    print a
    box_plot( [measure],["Avg="+str(round(a,6))], measureName,save_name=save_path+measureName+".png")
    
    mean = np.mean(np.asarray(measure))
    
    std  = np.sqrt(np.mean((np.asarray(measure)-mean)**2))
 
    ff = open(save_path+measureName+".txt",'w')
    ff.write("mean:"+str(mean)+"\n")
    ff.write("std:"+str(std)+"\n")
    ff.close()

def draw_validation_diagram( path , savePath):
    loss_pattern=r"\] loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accu_pattern=r"\] accuracy = (?P<acc_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)" 
    acc=dict()
    los=dict()
    for l in os.listdir(path):
        it = int(l.split('-')[-1].split('.')[0])
    
        with open(path+l, 'r') as log_file:
            log = log_file.read()
            
        for r in re.findall(loss_pattern, log):
            los[it] = r[0]
            
        for r in re.findall(accu_pattern, log):
            acc[it] = r[0]
    accit =list()
    accvl =list()
    for key in sorted(acc.iterkeys()):
        accit.append(key)
        accvl.append(acc[key])
        #print key, acc[key]
        
    losit =list()
    losvl =list()    
    for key in sorted(los.iterkeys()):
        losit.append(key)
        losvl.append(los[key])
   # print np.log(np.array(losvl).astype(float))
    color_ind=0
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('test loss')
  
    modula = len(plt.rcParams['axes.color_cycle'])
    ax1.plot(losit, np.log(np.array(losvl).astype(float))-np.min(np.log(np.array(losvl).astype(float))), color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula])
    plt.savefig(savePath+'test-loss.png')
    ax1.cla()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('test accuracy ')
    ax1.plot(accit, np.abs(np.log(np.abs(np.log(np.array(accvl).astype(float))))), plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])
    plt.savefig(savePath+'accuracy.png')
    ax1.cla()
    plt.close()
    #plt.show(True)
def compute_ADAD(gt, gtDru, pr, mean=-1):
    
    mGt = remove_nrpe_to_bm_part_from_gt(gt, gtDru)
    mGt = filter_drusen_by_size(mGt)
    
    gt = mGt.astype('float')
    pr = pr.astype('float')
    gt[gt>0]  = 1.0
    gt[gt<=0] = 0.0
    pr[pr>0]  = 1.0
    pr[pr<=0] = 0.0
    
   # adad      = np.sum(np.abs(gt - pr))
    adad = np.abs(np.sum(gt)-np.sum(pr))
    #print adad
    numAscans = float(np.sum(np.sum(gt, axis=0)>0)) if np.sum(gt)>0.0 else 1.0
    adad = float(adad)/numAscans 
    print "====================>:",adad
    if (mean == -1):
        #print adad, numAscans
      #  show_images([gt,pr,np.abs(gt- pr)],1,3)
        return adad
    else:
        return (adad-mean)**2
        
def compute_OR(gt, gtDru, pr, mean=-1):
    
    mGt = remove_nrpe_to_bm_part_from_gt(gt, gtDru)
    mGt = filter_drusen_by_size(mGt)
   # show_images([gt,mGt,pr],1,3)
    gt = mGt.astype('float')
    pr = pr.astype('float')
    gt[gt>0]  = 1.0
    gt[gt<=0] = 0.0
    pr[pr>0]  = 1.0
    pr[pr<=0] = 0.0
    
    denom = float(np.sum(np.logical_or(gt>0, pr>0)>0))
    nom   = float(np.sum(np.logical_and(gt>0, pr>0)>0))
    numAscans = float(np.sum(np.sum(gt, axis=0)>0)) if np.sum(gt)>0.0 else 1.0
    OR      = nom/denom if denom > 0 else 1.0
    #OR = OR/numAscans if numAscans > 0 else OR
    print nom, denom, OR
    
    if (mean == -1):
        #print adad, numAscans
        #show_images([gt,pr,np.logical_or(gt>0, pr>0),np.logical_and(gt>0, pr>0)],2,2)
        return OR
    else:
        return (OR-mean)**2   
def read_pickle_data( data_path ):
    with open( data_path, 'rb' ) as input:
        return pickle.load( input )
        
def write_pickle_data(data_path, data):
    with open( data_path + ".pkl",'w') as output:
        print "Writing pickle data into file..."
        pickle.dump( data, output, pickle.HIGHEST_PROTOCOL )
        print "Writing is done."
def remove_nrpe_to_bm_part_from_gt(lineGt, drusenGt):
    mask = np.zeros(lineGt.shape)
    y_n, x_n = normal_RPE_estimation( lineGt )
    i = 0
    for x in x_n:
        mask[:y_n[i]+1,x] = 1.0
        i += 1
    return mask * drusenGt


    
def iterate_in_folders(path, predPath, fold_num, fold_type, calcMeasure='IoU',justDrusen=False,label=1):
    
    target_folder = "" 

    folders = [f for f in listdir(path) if isfile(join(path, f))]
    counter = 0
    subject_id_file="/home/rasha/Desktop/DataFromUniClinic/Multi-fold-training-data/Fold0"+str(fold_num)+"/test_subjects.txt"
    subject_ids = get_subject_ids_to_evaluate(path=subject_id_file)
    processed_subjects = set()
    measure = []
    for d1 in os.listdir(path):
        
        for d2 in os.listdir(path+'/'+d1):
            s_id = int(d2[3:6])
            
            if( not s_id in subject_ids or s_id in processed_subjects):
#                print "Skip ",d2
                continue
            print "Process ",d2
            #continue
            for d3 in os.listdir(path+'/'+d1+'/'+d2):
             #   if(counter>100):
             #       break
                print "Working dir: ", path+'/'+d1+'/'+d2+'/'+d3
               # if(target_folder!="" and params['inDir']+'/'+d1+'/'+d2+'/'+d3 != target_folder):
               #     print params['inDir']+'/'+d1+'/'+d2+'/'+d3
               #     continue
                print "=================================="
                rawstack = []
                ind = []
                for f in os.listdir(path+'/'+d1+'/'+d2+'/'+d3):
                    filename = path+'/'+d1+'/'+d2+'/'+d3+'/'+f
                 #   if( target_folder!="" and params['inDir']+'/'+d1+'/'+d2+'/'+d3 != target_folder):
                 #       break
                    
                    #filename = filename.split('/')[-1]
                    ftype = filename.split('/')[-1].split('-')[-1]
                    if( ftype != "BinSeg.tif"):
                        continue
                    fnum = int(filename.split('/')[-1].split('-')[0])
                    ind.append(fnum)
                    print fnum, filename.split('/')[-1]
                  
                    drusLocPath = "/home/rasha/Desktop/DataFromUniClinic/Input-With-Drusen-Label/"
                    filenameDrusenLoc = drusLocPath + d1+'/'+d2+'/'+d3+'/'+f
                    gtDrusenLocation  = io.imread(filenameDrusenLoc)>0
                    gt = io.imread(filename)
          #          pr = io.imread(predPath + 'Fold0' + str(fold_num) +'/'+fold_type+'/test-output/'+d1+'/'+d2+'/'+d3+'/'+str(fnum)+'-binmask.tif')
                    f_layers = f.replace('-BinSeg.tif','-comp_layers.png')
          #          pr = io.imread(predPath + 'Fold0' + str(fold_num) +'/'+fold_type+'/test-output/'+d1+'/'+d2+'/'+d3+'/'+f_layers)
                    pr = io.imread(predPath + 'matlab_images/' + d1+'/'+d2+'/'+d3+'/'+f_layers)
                    
                    if( calcMeasure == 'IoU'):
                        if(justDrusen):
                            if( fold_type == 'area'):
                                prDrusenLocation = get_drusens_from_rpe_to_bm(pr,useWarping=True)
                                l_IoU = compute_IoU(gtDrusenLocation, prDrusenLocation, gtType='drusen')
                                measure.append(l_IoU)
                            elif( fold_type == 'drusen'):
                                l_IoU = compute_IoU(gt, pr, gtType=fold_type)
                                measure.append(l_IoU)
                        else:
                      #      show_images([gt,gtDrusenLocation,pr],1,3)
                            if( fold_type == 'chen'):
                                pr = filter_drusen_by_size(pr)
                            l_IoU = compute_IoU(gt, gtDrusenLocation, pr, gtType=fold_type,label=label)
                            measure.append(l_IoU)
                        
                    elif( calcMeasure == 'ADAD'):

                        if( fold_type == 'area'):
                            prN = denoise_BM(pr, farDiff=10,max_deg = 5, it = 5)
                            #area = find_area_between_seg_lines(prN)
                            dru = compute_drusen_mask(prN)
                            dru = filter_drusen_by_size(dru)
                            
                            #prDrusenLocation = get_drusens_from_rpe_to_bm(pr,useWarping=True)
                            prDrusenLocation = dru
                           # show_images([gt,pr,prN,dru],2,2)
                        elif( fold_type == 'drusen' ):
                            prDrusenLocation = pr
                        elif( fold_type == 'drusenNrpe' ):
                            prDrusenLocation = pr
                            prDrusenLocation = filter_drusen_by_size(prDrusenLocation)
                        elif( fold_type == 'chen' ):
                            gtDrusenLocation=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)
                            prDrusenLocation = filter_drusen_by_size(pr)
                        
                        l_ADAD = compute_ADAD( gt, gtDrusenLocation, prDrusenLocation )
                        measure.append(l_ADAD)
                        
                    elif( calcMeasure == 'OR'):
                        drusLocPath = "/home/rasha/Desktop/DataFromUniClinic/Input-With-Drusen-Label/"
                        filenameDrusenLoc = drusLocPath + d1+'/'+d2+'/'+d3+'/'+f
                        gtDrusenLocation  = io.imread(filenameDrusenLoc)>0
                        if( fold_type == 'area'):
                            #prDrusenLocation = get_drusens_from_rpe_to_bm(pr,useWarping=True)
                            prN = denoise_BM(pr, farDiff=10,max_deg = 5, it = 5)
                            #area = find_area_between_seg_lines(prN)
                            dru = compute_drusen_mask(prN)
                            dru = filter_drusen_by_size(dru)
                            
                            #prDrusenLocation = get_drusens_from_rpe_to_bm(pr,useWarping=True)
                            prDrusenLocation = dru
                        elif( fold_type == 'drusen' ):
                            prDrusenLocation = pr
                        elif( fold_type == 'drusenNrpe' ):
                            prDrusenLocation = pr
                            prDrusenLocation = filter_drusen_by_size(prDrusenLocation)
                        elif( fold_type == 'chen' ):
                            gtDrusenLocation=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)
                            prDrusenLocation = filter_drusen_by_size(pr)
                            
                        l_OR = compute_OR(  gt,gtDrusenLocation, prDrusenLocation )
                        measure.append(l_OR)
                        
                    
                    counter += 1
                  #  show_images([gt, pr, gtDrusenLocation,prDrusenLocation],2,2)
    if( justDrusen ):
        nsave = predPath + 'Fold0' + str(fold_num) +'/'+fold_type+'/test-evaluation-just-for-drusen/'
    else:
        nsave = predPath + 'Fold0' + str(fold_num) +'/'+fold_type+'/test-evaluation/'
        
    nsave = predPath + '/test-evaluation/'
    
    if not os.path.exists(nsave):
        os.makedirs(nsave)
    
    write_pickle_data( nsave+calcMeasure+str(label) , measure)
    #print measure
    a = np.mean(np.asarray(measure))
    
    box_plot( [measure],["Avg="+str(round(a,6))], calcMeasure,save_name=nsave+calcMeasure+str(label)+".png")
    
    mean = np.mean(np.asarray(measure))
    
    std  = np.sqrt(np.mean((np.asarray(measure)-mean)**2))
   # print "--------------============="
   # print np.asarray(measure)-mean
   # print "--------------------"
    #print (np.asarray(measure)-mean)**2
    #print "--------------------"
    #print std
    
    print calcMeasure, mean, std
    ff = open(nsave+calcMeasure+str(label)+".txt",'w')
    ff.write("mean:"+str(mean)+"\n")
    ff.write("std:"+str(std)+"\n")
    ff.close()
    
    return mean, std
    
def main():

    if( True ):
        a = ""#-Drusen
        path = "/home/rasha/Desktop/DataFromUniClinic/Input-With"+a+"-Label/"
 #       predPath = "/home/rasha/Desktop/resnet_results/"
        predPath = '/home/rasha/work/deeplab-public-ver2/drusen_area/features/RESNET-50/val/small/'
        ma = iterate_in_folders(path, predPath, fold_num=1, fold_type='area',calcMeasure='ADAD',justDrusen=False, label = 1)
        mb = iterate_in_folders(path, predPath, fold_num=1, fold_type='area',calcMeasure='OR',justDrusen=False, label = 1)
        mc = iterate_in_folders(path, predPath, fold_num=1, fold_type='area',calcMeasure='IoU',justDrusen=False, label = 1)
        print
        print
        print 'ADAD', ma
        print 'OR', mb
        print 'IoU', mc
        exit()
        















main()
        