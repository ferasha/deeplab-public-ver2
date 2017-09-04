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
import networkx as nx
import skimage.graph as skg
import skimage.morphology as skmorph
import skimage.measure as skm
import timeit
import itertools
import cv2
import pandas as pd
import math 
import sklearn.metrics as sklearnm
import matplotlib.mlab as mlab 

octParams = dict()
octParams['bResolution'] = 'low' #high
octParams['hx'] = 200./16. #x axis in each B-scan
octParams['hy'] = 200./51. #y axis in each B-scan 
octParams['hz'] = 200./10. #z axis in direction of B-scan slices
octParams['zRate'] = 13 #every 13 pixels for low resolution, every two pixels for high res

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

def show_PED_volume( scans, values, interplolation = 'bilinear',titles=['GT','Layer Based'],\
                 block = True,savePath=""):
    
    print scans[0].shape
    fig = plt.figure(figsize=(15.0,6.0))
    for i in range(len(scans)):
        ax = plt.subplot( 1, len(scans), i+1 , projection='3d')
        ax.set_title( titles[i] )
        b_scans=scans[i]
        value=values[i]
        h, w, d = b_scans.shape
        img = (b_scans>200).astype(float)
        img = sc.ndimage.filters.gaussian_filter(img,0.5)
       # show_image(img[:,:,0])
        Z, X, Y = np.where( img >= 0.2 )
        '''
        i  = np.arange(len(Z))
        ii = np.where( i%10 == 0 )
        X  = X[ii]
        Y  = Y[ii]
        Z  = Z[ii]
        '''
        
        #ax = fig.gca(projection='3d')
        # Plot the surface.
        #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax.plot_trisurf(X, Y, Z,cmap=plt.cm.RdYlBu_r, antialiased=False,edgecolor='none')
        ax.set_zlim(180, 300)
        ax.view_init(45,26)
  #  ax.scatter(X, Y, Z, c='r', marker='o')   
    if( savePath != "" ):
        plt.savefig(savePath + ".png")
        plt.close()
    else:
        plt.show(block)
   
   
def increase_resolution( b_scans, factor , interp='nearest'):
    new_size = (b_scans.shape[0], b_scans.shape[1], b_scans.shape[2]*factor)
   # res = misc.imresize(b_scans, new_size, interp = 'nearest')
    res = np.zeros(new_size)
    for i in range(new_size[1]):
        slice_i = b_scans[:,i,:]
        
        res[:,i,:] = (misc.imresize(slice_i, (new_size[0], new_size[2]),interp = interp).astype('float'))/255.0
        mask=np.copy(np.cumsum(res[:,i,:],axis=0)*((res[:,i,:]>0).astype('int')))
        res[:,i,:]=(mask>=1.0).astype('float')
     #   if(np.sum(slice_i)>0):
      #      show_images([mask,res[i,:,:]],1,2)
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

def extract_seg_layers2( label, trim=False , debug=False):
    h, w = label.shape
    res  = np.copy( label.astype('float') )
    res.fill(0.)
    
    for j in range(w):
        col = label[:, j]
        loc = np.where(col > 0)[0]
        if( loc != [] ):
            
            if( np.min(loc) == np.max(loc) ):
                res[np.max(loc):np.max(loc)+1, j] = 3.
                res[np.max(loc)+1:np.max(loc), j] = 3.
                
            else:
                res[np.min(loc):np.min(loc)+1, j] = 2.
                res[np.min(loc)+1:np.min(loc), j] = 2.
                res[np.max(loc):np.max(loc)+1, j] = 1.
                res[np.max(loc)+1:np.max(loc), j] = 1.
                
    if( len(np.unique(res)) > 3):
        print "In extract layer First case"
        res[res==3]=255.
        res[res==2]=170.
        res[res==1]=85.
    else:
        print "In extract layer second case"
        res[res==2]=255.
        res[res==1]=127.
    return res
    
def extract_seg_layers( label, trim=False ):
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
    
def evaluate_net_outputs():
    gt_names = sort_all_file_names_in("/home/rasha/Desktop/DataFromUniClinic/test-05-04-2017/Label/71-78/MOD077/150813_145/", '-')
    pr_all_n = sort_all_file_names_in("/home/rasha/Desktop/DataFromUniClinic/test-05-04-2017/Prediction-All/71-78/MOD077/150813_145/", '-')          
    pr_dri_n = sort_all_file_names_in("/home/rasha/Desktop/DataFromUniClinic/test-05-04-2017/Prediction-Dist/71-78/MOD077/150813_145/", '-')
    pr_drs_n = sort_all_file_names_in("/home/rasha/Desktop/DataFromUniClinic/test-05-04-2017/Prediction-Drusen/71-78/MOD077/150813_145/", '-')
                
    all_avg_dist_rpe = []
    all_avg_dist_bm  = []
    all_OR   = []
    all_ADAD = []
    all_true = []
    all_fals = []
    all_IoU  = []
    all_avgs = np.zeros((7))
    
    dru_avg_dist_rpe = []
    dru_avg_dist_bm  = []
    dru_OR   = []
    dru_ADAD = []
    dru_true = []
    dru_fals = []
    dru_IoU  = []
    dru_avgs = np.zeros((7))
    
    drs_OR   = []
    drs_ADAD = []
    drs_true = []
    drs_fals = []
    drs_IoU  = []
    drs_avgs = np.zeros((7))
    
    for i in range(len(gt_names)):
           print i
          # if( i == 5 ):
          #     break
           gr = io.imread(gt_names[i])
           al = io.imread(pr_all_n[i])
           di = io.imread(pr_dri_n[i])
           ds = io.imread(pr_drs_n[i])
           #show_images([gr, al, di, ds], 2, 2)
           # Compute all the values for the Ground Truth and Net results 1
           true_pos, false_pos, or_value, adad, IoU = compute_true_and_false_positives( gr, al )
           avg_dist_rpe = compute_curve_distance( gr, al, layer_type='RPE')
           avg_dist_bm  = compute_curve_distance( gr, al, layer_type='BM')
           
           all_true.append(true_pos);all_fals.append(false_pos);all_IoU.append(IoU)
           all_ADAD.append(adad);all_avg_dist_rpe.append(avg_dist_rpe);all_avg_dist_bm.append(avg_dist_bm)
           if( or_value != np.inf ):
               all_OR.append(or_value)
           # Compute all the values for the Ground Truth and Net results 2
           true_pos, false_pos, or_value, adad, IoU  = compute_true_and_false_positives( gr, di )
           avg_dist_rpe = compute_curve_distance( gr, di, layer_type='RPE')
           avg_dist_bm  = compute_curve_distance( gr, di, layer_type='BM')
           
           dru_true.append(true_pos);dru_fals.append(false_pos);dru_IoU.append(IoU)
           dru_ADAD.append(adad);dru_avg_dist_rpe.append(avg_dist_rpe);dru_avg_dist_bm.append(avg_dist_bm)
           if( or_value != np.inf ):
               dru_OR.append(or_value)
            
            
           drusen_mask = compute_drusen_mask( gr )
           drusen_mask_gt = filter_drusen_by_size( drusen_mask)
           ds = filter_drusen_img_with_line_seg_img(ds,al)
           
           # Compute all the values for the Ground Truth and Net results 3
           true_pos, false_pos, or_value, adad, IoU  = compute_true_and_false_positives( drusen_mask_gt, ds , input_type = 'drusen')
          
           drs_true.append(true_pos);drs_fals.append(false_pos)
           drs_ADAD.append(adad);drs_IoU.append(IoU)
           if( or_value != np.inf ):
               drs_OR.append(or_value)
           #show_images([drusen_mask_gt,ds],1,2, ["Drusen","Filtered"])
           #show_images([gr,al,di,ds],2,2, ["Gr","All","Drusen","Seg"])
            
    all_avgs[0] = np.mean(all_avg_dist_rpe);all_avgs[1] =  np.mean(all_avg_dist_bm)   
    all_avgs[2] = np.mean(all_OR);all_avgs[3] =  np.mean(all_ADAD)   
    all_avgs[4] = np.mean(all_true);all_avgs[5] =  np.mean(all_fals)   
    all_avgs[6] = np.mean(all_IoU)
    
    dru_avgs[0] = np.mean(dru_avg_dist_rpe);dru_avgs[1] =  np.mean(dru_avg_dist_bm)   
    dru_avgs[2] = np.mean(dru_OR);dru_avgs[3] =  np.mean(dru_ADAD)   
    dru_avgs[4] = np.mean(dru_true);dru_avgs[5] =  np.mean(dru_fals)   
    dru_avgs[6] = np.mean(dru_IoU)
       
    drs_avgs[2] = np.mean(drs_OR);drs_avgs[3] =  np.mean(drs_ADAD)   
    drs_avgs[4] = np.mean(drs_true);drs_avgs[5] =  np.mean(drs_fals)   
    drs_avgs[6] = np.mean(drs_IoU)
    
    
    # Save Histograms
    path = "/home/rasha/Desktop/DataFromUniClinic/test-05-04-2017/"
    # Draw Histograms
    if( False ):
        nbin = 100
        draw_histogram( all_avg_dist_rpe, nbin, path+"all_avg_dis_RPE.png")
        draw_histogram( dru_avg_dist_rpe, nbin, path+"dru_avg_dis_RPE.png")
        draw_histogram( all_avg_dist_bm, nbin, path+"all_avg_dis_BM.png")
        draw_histogram( dru_avg_dist_bm, nbin, path+"dru_avg_dis_BM.png")
        draw_histogram( all_OR, nbin, path+"all_OR.png")
        draw_histogram( dru_OR, nbin, path+"dru_OR.png")
        draw_histogram( all_ADAD, nbin, path+"all_ADAD.png")
        draw_histogram( dru_ADAD, nbin, path+"dru_ADAD.png")
        draw_histogram( all_true, nbin, path+"all_true.png")
        draw_histogram( dru_true, nbin, path+"dru_true.png")
        draw_histogram( all_fals, nbin, path+"all_false.png")
        draw_histogram( dru_fals, nbin, path+"dru_false.png")
        draw_histogram( all_IoU, nbin, path+"all_IoU.png")
        draw_histogram( dru_IoU, nbin, path+"dru_IoU.png")
    if( True ):
        box_plot( [all_avg_dist_rpe, dru_avg_dist_rpe],["Net A\nAvg="+str(round(all_avgs[0],6)),"Net B\nAvg="+str(round(dru_avgs[0],6))], "RPE Distance",  path+"b-avg_dis_RPE.png")
        box_plot( [all_avg_dist_bm, dru_avg_dist_bm],["Net A\nAvg="+str(round(all_avgs[1],6)),"Net B\nAvg="+str(round(dru_avgs[1],6))], "BM Distance", path+"b-avg_dis_BM.png")
        box_plot( [all_OR, dru_OR, drs_OR],["Net A\nAvg="+str(round(all_avgs[2],6)),"Net B\nAvg="+str(round(dru_avgs[2],6)),"Net C\nAvg="+str(round(drs_avgs[2],6))], "Overlapping Ratio", path+"b-OR.png")
        box_plot( [all_ADAD, dru_ADAD, drs_ADAD],["Net A\nAvg="+str(round(all_avgs[3],6)),"Net B\nAvg="+str(round(dru_avgs[3],6)),"Net C\nAvg="+str(round(drs_avgs[3],6))], "Absolute Drusen Area Difference", path+"b-ADAD.png")
        box_plot( [all_true, dru_true, drs_true],["Net A\nAvg="+str(round(all_avgs[4],6)),"Net B\nAvg="+str(round(dru_avgs[4],6)),"Net C\nAvg="+str(round(drs_avgs[4],6))], "True Positive", path+"b-true.png")
        box_plot( [all_fals,dru_fals,drs_fals],["Net A\nAvg="+str(round(all_avgs[5],6)),"Net B\nAvg="+str(round(dru_avgs[5],6)),"Net C\nAvg="+str(round(drs_avgs[5],6))], "False Positive", path+"b-false.png")
        box_plot( [all_IoU,dru_IoU,drs_IoU],["Net A\nAvg="+str(round(all_avgs[6],6)),"Net B\nAvg="+str(round(dru_avgs[6],6)),"Net C\nAvg="+str(round(drs_avgs[6],6))], "Intersection Over Union (IoU)", path+"b-IoU.png")

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
    def printPoint(self):
        print "(x,y)=(", self.x,",",self.y,")"
        
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
    
def update_OCT_params(bScansRes):
    isHighRes = bScansRes>70
    if( isHighRes ):
        octParams['bResolution'] = 'hight'
        octParams['zRate'] = 2
    else:
        octParams['bResolution'] = 'low'
        octParams['zRate'] = 13
def save_drusen_quantification(projectionImage, labels,savePath,unit='Pixel',ccx=[],ccy=[],sortBasedOnClosestPoints=False):
    cx,cy,area, height, volume, largeR, smallR,theta = quantify_drusen(projectionImage, labels)
    
    
    if(sortBasedOnClosestPoints):
        sortInd=list()
        for i in range(len(ccx)):
            minJ=-1
            minD=np.inf
            for j in range(len(cx)):
                if(not j in sortInd):
                    dist=np.sqrt((ccx[i]-cx[j])**2+(ccy[i]-cy[j])**2)
                    if(dist<minD):
                        minD=dist
                        minJ=j
            if(not minJ in sortInd and minJ!=-1):
                sortInd.append(minJ)
        cx=cx[sortInd]
        cy=cy[sortInd]
        area=area[sortInd]
        height=height[sortInd]
        volume=volume[sortInd]
        largeR=largeR[sortInd]
        smallR=smallR[sortInd]
        theta=theta[sortInd]
        print sortInd
    areaM, heightM, volumeM, largeM, smallM=\
     convert_from_pixel_size_to_meter(area, height, volume, largeR, smallR, theta)
    
    drusenInfo=dict()
    drusenInfo['Center']=list()
    drusenInfo['Area']=list()
    drusenInfo['Height']=list()
    drusenInfo['Volume']=list()
    drusenInfo['Diameter']=list()
    for i in range(len(cx)):
        drusenInfo['Center'].append((int(cx[i]),int(cy[i])))
        if( unit=='Pixel'):
            drusenInfo['Diameter'].append((largeR[i],smallR[i]))
        else:
            drusenInfo['Diameter'].append((largeM[i],smallM[i]))
    if( unit == 'Pixel' ):
        drusenInfo['Area']=area.astype(int)
        drusenInfo['Height']=height.astype(int)
        drusenInfo['Volume']=volume.astype(int)
    else:
        drusenInfo['Area']=areaM
        drusenInfo['Height']=heightM
        drusenInfo['Volume']=volumeM
    df=pd.DataFrame(drusenInfo,index=(np.arange(len(area))+1),columns=['Center','Area',\
        'Height','Volume','Diameter'])
    print df
    #writer = pd.ExcelWriter('/home/rasha/Desktop/drusenQuantification/drusen.xlsx')
    df.to_csv(savePath, sep='\t')
   # writer.save()
    #exit()
    return cx,cy,area, height, volume, largeR, smallR,theta
def draw_drusen_properties(gcx,gcy,garea, gheight, gvolume, glargeR, gsmallR,gtheta,\
pcx,pcy,parea, pheight, pvolume, plargeR, psmallR, ptheta):
    plt.figure(figsize=(10.0, 8.0))
    ax1=plt.subplot(221)
    ax1.set_title('Area')
 #   plt.plot(np.arange(len(garea)), garea,'r',marker='o', label='GT')
    plt.plot(np.arange(len(garea)), garea, 'r', label='GT')
  #  plt.plot(np.arange(len(parea)), parea, 'b',marker='o', label='CNN')
    plt.plot(np.arange(len(parea)), parea, 'b', label='CNN')
    plt.legend(loc="upper left")
    
    ax2=plt.subplot(222)
    ax2.set_title('Height')
   # plt.plot(np.arange(len(gheight)), gheight, 'r',marker='o', label='GT')
    plt.plot(np.arange(len(gheight)), gheight, 'r', label='GT')
   # plt.plot(np.arange(len(pheight)), pheight, 'b',marker='o', label='CNN')
    plt.plot( np.arange(len(pheight)), pheight, 'b', label='CNN')
    plt.legend(loc="upper left")
    
    ax3=plt.subplot(223)
    ax3.set_title('Volume')
    #plt.plot(np.arange(len(gvolume)), gvolume, 'r',marker='o', label='GT')
    plt.plot(np.arange(len(gvolume)), gvolume, 'r', label='GT')
    #plt.plot(np.arange(len(pvolume)), pvolume, 'b',marker='o', label='CNN')
    plt.plot( np.arange(len(pvolume)), pvolume, 'b', label='CNN')
    plt.legend(loc="upper left")
    
    ax4=plt.subplot(224)
    ax4.set_title('Major and Minor Axis Length(Diameter)')
    #plt.plot(np.arange(len(glargeR)), glargeR, 'r',marker='o', label='GT')
    plt.plot(np.arange(len(glargeR)), glargeR, 'r', label='GT')
    #plt.plot(np.arange(len(plargeR)), plargeR, 'b',marker='o', label='CNN')
    plt.plot( np.arange(len(plargeR)), plargeR, 'b', label='CNN')
    #plt.plot(np.arange(len(gsmallR)), gsmallR, color='#ffa3c9',marker='o', label='GT')
    plt.plot(np.arange(len(gsmallR)), gsmallR, color='#e800e8', label='GT')
    #plt.plot(np.arange(len(psmallR)), psmallR, color='#87cbff', marker='o',label='CNN')
    plt.plot( np.arange(len(psmallR)), psmallR, color='#e8d000', label='CNN')
    plt.legend(loc="upper left")
    plt.show(True)
    
def draw_drusen_boundary_over_projection_image( b_scans, gts, prs=[], show = False , scale=1, input_type='line_segments'):
    '''
    for i in range(prs.shape[2]):
        if(i>33):
            drusen_mask1 = compute_drusen_mask( prs[:,:,i] ,useWarping=True)
            drusen_mask = filter_drusen_by_size( drusen_mask1, slice_num=1)
            show_images([prs[:,:,i],drusen_mask],1,2)
    '''
    unit='Pixel'
    update_OCT_params(b_scans.shape[2])
    useWarping = True
    print "1",useWarping
    k = produce_drusen_projection_image( b_scans, gts, useWarping=useWarping )
    k /= np.max(k) if np.max(k) != 0.0 else 1.0
   # show_image(k)
    projection_image = sc.misc.imresize(k, (k.shape[0]*scale,k.shape[1])).astype('float')
    projection_image /= np.max(projection_image) if np.max(projection_image) != 0.0 else 1.0
    
    #show_image(projection_image)
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
   # gcx,gcy,garea, gheight, gvolume, glargeR, gsmallR,gtheta = save_drusen_quantification(k,mm,savePath="/home/rasha/Desktop/drusenQuantification/GroundTruth-"+unit+".csv",unit=unit)
    
   # save_array("/home/rasha/Desktop/DataFromUniClinic/test-05-04-2017/test/",b_scans,gts,mm)
    #show_image(projection_image)
    #show_images([k,k2],1,2)

    gts_extent   = binarize(sc.misc.imresize(k, (k.shape[0]*scale,k.shape[1]),interp='nearest' ))
    
    gts_boundary = find_boundaries_in_binary_mask( gts_extent )
    save_array("/home/rasha/Desktop/DataFromUniClinic/test-05-04-2017/test23/",b_scans,gts,mm,useWarping=useWarping)
    if( show ):
        rgb_img = np.empty((projection_image.shape[0],projection_image.shape[1], 3), dtype='float')
        
        rgb_img[:,:,0] = projection_image
        rgb_img[:,:,1] = projection_image 
        rgb_img[:,:,2] = projection_image 
    
        rgb_img = (rgb_img/np.max(rgb_img))*0.8      
        
        rgb_img[gts_boundary == 1.0,0] = 1.0
        rgb_img[gts_boundary == 1.0,1] *= 0.5
        rgb_img[gts_boundary == 1.0,2] *= 0.5
       # show_image_rgb(rgb_img)  
    if( prs != [] ):
       
        k = remove_false_positives( projection_image, prs ,intensity_t=0,useWarping=True)
        #k = find_drusen_in_stacked_slices( prs , input_type=input_type)
        #k = remove_non_bright_spots( projection_image, k )
        prs_extent = binarize(sc.misc.imresize(k, (k.shape[0]*scale,k.shape[1]), interp='nearest'))
        
        prs_boundary = find_boundaries_in_binary_mask( prs_extent )
        mm = get_label_from_projection_image(k, prs, method='rpeToNormRpe')
        #pcx,pcy,parea, pheight, pvolume, plargeR, psmallR, ptheta=save_drusen_quantification(k,mm,savePath="/home/rasha/Desktop/drusenQuantification/Prediction-"+unit+".csv",unit=unit)
      #  draw_drusen_properties(gcx,gcy,garea, gheight, gvolume, glargeR, gsmallR,gtheta,\
      #          pcx,pcy,parea, pheight, pvolume, plargeR, psmallR, ptheta)
        save_array("/home/rasha/Desktop/DataFromUniClinic/test-05-04-2017/test22/",b_scans,prs,mm,useWarping=useWarping)
        if( show ):
            
            rgb_img[prs_boundary == 1.0,0] *= 0.5
            rgb_img[prs_boundary == 1.0,1] = 1.0
            rgb_img[prs_boundary == 1.0,2] = 1.0
            
            gtsPrsBoundaries = prs_boundary * gts_boundary
            rgb_img[gtsPrsBoundaries == 1.0,0] = 1.0
            rgb_img[gtsPrsBoundaries == 1.0,1] *= 0.5
            rgb_img[gtsPrsBoundaries == 1.0,2] = 1.0
            
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
    #show_image(projection.astype('float')/np.max(projection.astype('float')))
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
    
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a
    
def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])  
    
def convert_from_pixel_size_to_meter(area, height, volume, largeR, smallR, theta):    
    voxelSize = octParams['hx']*octParams['hy']*octParams['hz']
    volumeM=volume * voxelSize
    areaM=area*octParams['hx']*octParams['hz']
    heightM=height*octParams['hy']
    xM=largeR*np.cos(theta)*octParams['hx']
    zM=largeR*np.sin(theta)*octParams['hz']
    largeM=np.sqrt(xM**2+zM**2)
    thetaVer=theta+(np.pi/2.0)
    xM=smallR*np.cos(thetaVer)*octParams['hx']
    zM=smallR*np.sin(thetaVer)*octParams['hz']
    smallM=np.sqrt(xM**2+zM**2)
    return areaM, heightM, volumeM, largeM, smallM
    
def quantify_drusen(projected_labels, labels):

    realResProjLbl = sc.misc.imresize(projected_labels, size=(projected_labels.shape[0]*octParams['zRate'],projected_labels.shape[1]),interp='nearest')
    aaa=np.copy(realResProjLbl)
    #show_image(realResProjLbl)
    realResLbl = increase_resolution(((labels>0).astype('float')), factor=octParams['zRate'],interp='bilinear')
    #realResLbl=(realResLbl>0).astype('float')
    #for ii in range(realResLbl.shape[2]):
     #   if(np.sum(realResLbl[:,:,ii])>0):
      #      show_image(realResLbl[:,:,ii])
    heightImg  = np.sum(realResLbl, axis=0).T
    realResProjLbl = ((realResProjLbl*heightImg)>0.0).astype('float')
    
    #show_images([heightImg,realResProjLbl,aaa],1,3)
    cca, numL = sc.ndimage.measurements.label(realResProjLbl)
    #print "Num of drusen:", numL-1
    
    area=list()
    volume=list()
    largeR=list()
    smallR=list()
    theta=list()
    height=list()
    cx=list()
    cy=list()
    bgL = get_label_of_largest_component( cca )
    labels   = np.unique(cca)
    for l in labels:
        if( l != bgL ):
            componentL = (cca==l).astype('float')
            #componentL2=np.copy(componentL)
            componentL=((heightImg*componentL)>0.0).astype('float')
            cyL,cxL = sc.ndimage.measurements.center_of_mass(componentL)
           # print l
            
            areaL = np.sum(((heightImg*componentL)>0.0).astype('float'))
            if( areaL < 4):
                continue
            volumeL = np.sum(heightImg*componentL)
            heightL = np.max(heightImg*componentL)
            if(heightL<3):
                continue
            #boundary=sc.ndimage.morphology.binary_dilation(componentL>0,iterations=1).astype('float')-componentL
           # show_image(boundary)
           # if(l==14):
            #    show_images([componentL,boundary],1,2)
            #y,x=np.where(boundary>0) 
            #ellipseL = fitEllipse(x,y)
            #largeL,smallL = ellipse_axis_length(ellipseL)
            #thetaL=ellipse_angle_of_rotation(ellipseL)
            #if(math.isnan(largeL) or math.isnan(smallL)):
             #   show_images([componentL,componentL2,heightImg],1,3)
            largeL=0.0
            smallL=0.0
            thetaL=0.0
            props=skm.regionprops(componentL.astype('int'))
            for p in props:
                if(p.label==1):
                    areaL=p.area
                    largeL=p.major_axis_length
                    smallL=p.minor_axis_length
                    thetaL=p.orientation
                
            '''  
            print "-----------------------------"
            print "Area:",areaL
            print "Height:",heightL
            print "Volume:",volumeL
            print "Ellipse:", largeL,smallL
            
            print "theta",thetaL
            show_image(componentL)
            '''
            area.append(areaL)
            volume.append(volumeL)
            theta.append(thetaL)
            smallR.append(smallL)
            largeR.append(largeL)
            height.append(heightL)
            cx.append(cxL)
            cy.append(cyL)
            #show_images([componentL,heightImg*componentL],1,2)

    area = np.asarray(area)
    volume = np.asarray(volume)
    largeR = np.asarray(largeR)
    smallR = np.asarray(smallR)
    theta = np.asarray(theta)
    height=np.asarray(height)
    cy=np.asarray(cy)
    cx=np.asarray(cx)
    total=(area+height+volume+largeR+smallR)/5.0
    indx=np.argsort(total)
    
    area = area[indx]
    volume = volume[indx]
    largeR = largeR[indx]
    smallR = smallR[indx]
    theta = theta[indx]
    height=height[indx]
    cy=cy[indx]
    cx=cx[indx]
    
    #show_images([projected_labels,realResProjLbl],1,2)
    return cx, cy, area, height, volume, largeR, smallR, theta
   # show_images([projected_labels,realResProjLbl],1,2)
    

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


rpe_scan_list  = []
nrpe_scan_list = []
#==============================================================================
# Vitalis Wiens Method
#==============================================================================



def seg_wiens( inputImage ):
  #inputImage=inputImage/255.
  #inputImage=inputImage.astype('uint8')
 # print type(inputImage[0,0]),type(inputImage)
 # print np.unique(inputImage)
  #exit()
   # process image
  filter1=FF.FilterBilateral(inputImage)
  
  threshold=FF.computeHistogrammThreshold(filter1)
  #print threshold
 
  RNFL=FF.getTopRNFL(filter1,threshold,False)
  if( len(RNFL) ==0):
       
        return [],[]

  mask=FF.thresholdImage(filter1,threshold)
  mask2=FF.removeTopRNFL(filter1,mask,RNFL)
  FF.extractRegionOfInteresst(filter1,mask2);
  centerLine1=FF.getCenterLine(mask2)
  #centerLine1=[]
  #generate ideal rpe
  centerLine2=FF.segmentLines_Wiens(filter1,centerLine1)
 # return [],[]
  itterations=3;
  dursenDiff=5;
  idealRPE=FF.getIdealRPELine(filter1,centerLine2,itterations,dursenDiff)
  
  return centerLine2, idealRPE

def run_wiens_method_on_a_pack(b_scans):
    h, w, d = b_scans.shape
    initialize_rpe_nrpe_lists(b_scans,method='Wiens')
    masks = np.zeros(b_scans.shape)
    for i in range(b_scans.shape[2]):
        start=timeit.default_timer()
        mask = draw_lines_on_mask(rpe_scan_list[i], nrpe_scan_list[i],(h,w))
        masks[:,:,i] = find_area_btw_RPE_normal_RPE(mask)
        #print "Chen elapsig time:", timeit.default_timer()-start
        #show_images([b_scans[:,:,i],mask,masks[:,:,i]],1,3)
    #print len(rpe_scan_list), len(nrpe_scan_list)
    delete_rpe_nrpe_lists()
    #print len(rpe_scan_list), len(nrpe_scan_list)
    #exit()
    return masks
#==============================================================================
# Chen's Method
#==============================================================================

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
    
def initialize_rpe_nrpe_lists( b_scans,method='Chen' ):
    
    for i in range(b_scans.shape[2]):
        print "#######################:",i
        #show_image(b_scans[:,:,i])
        start=timeit.default_timer()
        if(method=='Chen'):
            rpe, nrpe = seg_chen(b_scans[:,:,i])
        elif(method=='Wiens'):
            
            rpe, nrpe = seg_wiens(b_scans[:,:,i])
      #  print timeit.default_timer()-start
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
        start=timeit.default_timer()
        mask = draw_lines_on_mask(rpe_scan_list[i], nrpe_scan_list[i],(h,w))
        masks[:,:,i] = find_area_btw_RPE_normal_RPE(mask)
        #print "Chen elapsig time:", timeit.default_timer()-start
        show_images([b_scans[:,:,i], draw_lines_on_mask(rpe,nrpe,b_scans[:,:,i].shape)],1,2)
    #print len(rpe_scan_list), len(nrpe_scan_list)
    delete_rpe_nrpe_lists()
    #print len(rpe_scan_list), len(nrpe_scan_list)
    #exit()
    return masks

#========================================= end of Chen related method functions                   
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
       # mPr = sc.ndimage.measurements.morphology.binary_erosion(mPr,iterations=1)
        
    elif(gtType=='drusenNrpe'):    
        mGt = remove_nrpe_to_bm_part_from_gt(gt, gtDru)
        mGt = filter_drusen_by_size(mGt)
        mPr = pr
        mPr = filter_drusen_by_size(mPr)
    elif(gtType=='chen' or gtType=='wiens'):
        mGt = remove_nrpe_to_bm_part_from_gt(gt, gtDru)
        mGt = filter_drusen_by_size(mGt)
        mPr = pr
    elif(gtType=='layer'):
        mGt = find_area_between_seg_lines(gt)
        mPr = find_area_between_seg_lines(pr)
       # show_images([gt,mGt,pr,mPr],2,2)
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
   
def total_evaluation( path, fold_type, measureName, save_path ,justEvaluateGAScans=False):
    folders = [f for f in listdir(path) if isfile(join(path, f))]
    counter = 0
    save_path += (fold_type+'/')
    measure = list()
    sourceFolder="test-evaluation"
    if(justEvaluateGAScans):
        sourceFolder="test-onGA-evaluation"
    for d1 in os.listdir(path):
        #if( fold_type == 'layer'):
          #  if( d1 == 'Fold04' or d1 =='Fold05'):
          #      continue
        fileName = path+'/'+d1+'/'+fold_type+'/'+sourceFolder+'/'+measureName+'.pkl'
        print fileName
        data = read_pickle_data(fileName)
        print len(data)
        measure.extend(data)
        
    print "===========>>",len(measure)
    #print measure
    draw_histogram_OR_ADAD(measure,100,title=fold_type,savePath=save_path+\
                    fold_type+'-'+measureName+"hist.png",logScale=True)
   # exit()
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
    if(True):
        accvl = np.asarray(accvl,dtype='float')
        losvl = np.asarray(losvl,dtype='float')
       
        accvl = sc.ndimage.filters.gaussian_filter1d(accvl,1.7)
        losvl = sc.ndimage.filters.gaussian_filter1d(losvl,1.7)
   # print np.log(np.array(losvl).astype(float))
    color_ind=0
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('test loss')
  
    modula = len(plt.rcParams['axes.color_cycle'])
    ax1.plot(losit, np.log(np.array(losvl).astype(float))-np.min(np.log(np.array(losvl).astype(float))), color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula], linewidth=2.0)
    plt.savefig(savePath+'test-loss.png')
   # plt.show(True)
    ax1.cla()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('test accuracy ')
    ax1.plot(accit, np.abs(np.log(np.abs(np.log(np.array(accvl).astype(float))))), plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula], linewidth=2.0)
    plt.savefig(savePath+'accuracy.png')
    ax1.cla()
   # plt.show(True)
    plt.close()
    #plt.show(True)
def compute_SDAD(gt, gtDru, pr, mean=-1):
    
    mGt = remove_nrpe_to_bm_part_from_gt(gt, gtDru)
    mGt = filter_drusen_by_size(mGt)
    
    gt = mGt.astype('float')
    pr = pr.astype('float')
    gt[gt>0]  = 1.0
    gt[gt<=0] = 0.0
    pr[pr>0]  = 1.0
    pr[pr<=0] = 0.0
    
   # adad      = np.sum(np.abs(gt - pr))
    sdad = np.sum(pr)-np.sum(gt)
    #print adad
    numAscans = float(np.sum(np.sum(gt, axis=0)>0)) if np.sum(gt)>0.0 else 1.0
    sdad = float(sdad)/numAscans 
    print "====================>:",sdad
  #  if(sdad>100):
  #      show_images([gt,pr,pr-gt],1,3)
    if (mean == -1):
        #print adad, numAscans
      #  show_images([gt,pr,np.abs(gt- pr)],1,3)
        return sdad
    else:
        return (sdad-mean)**2
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
    
    denom = float(np.sum((np.logical_or(gt>0, pr>0)>0).astype('float')))
    nom   = float(np.sum((np.logical_and(gt>0, pr>0)>0).astype('float')))
    numAscans = float(np.sum(np.sum(gt, axis=0)>0)) if np.sum(gt)>0.0 else 1.0
    OR      = nom/denom if denom > 0 else 1.0
 #   OR      = nom/denom 
    #OR = OR/numAscans if numAscans > 0 else OR
    print nom, denom, OR
    
    if (mean == -1):
        #print adad, numAscans
    #    show_images([gt,pr,np.logical_or(gt>0, pr>0),np.logical_and(gt>0, pr>0)],2,2,\
    #        ["gt","pr","gt || pr","gt & pr = "+str(OR)])
        return OR
    else:
        return (OR-mean)**2  
        
def read_pickle_data( data_path ):
    print "Reading Pickle data from file..."
    with open( data_path, 'rb' ) as input:
        return pickle.load( input )
    print "End of reading."
        
def write_pickle_data(data_path, data):
    with open( data_path + ".pkl",'w') as output:
  ##      print "Writing pickle data into file..."
        pickle.dump( data, output, pickle.HIGHEST_PROTOCOL )
  ##      print "Writing is done."
def remove_nrpe_to_bm_part_from_gt(lineGt, drusenGt):
    mask = np.zeros(lineGt.shape)
    y_n, x_n = normal_RPE_estimation( lineGt )
    i = 0
    for x in x_n:
        mask[:y_n[i]+1,x] = 1.0
        i += 1
    return mask * drusenGt


def get_data_from_file( path ):
    with open( path ) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    return content
    
def separate_scan_address_and_bScanId(scanList):
    scanAddress=list()
    bId  =list()
    for l in scanList:

        scanAddress.append(l.split(',')[0])
        
        bId.append(l.split(',')[1])
    return scanAddress, bId
    
def iterate_in_folders(path, predPath, fold_num, fold_type, calcMeasure='IoU',\
        justDrusen=False, justEvaluateGAScans=False,skipDrusenFreeBscan=False,label=1):
    
    target_folder = "" 
    prfolder = ""
    if( fold_type == "layer" ):
        prfolder = "test-output-shortPath"
    else:
        prfolder = "test-output"
        
    gaScans = list()
    gaBScanIds = list()
    saveFolder=""
    if( justEvaluateGAScans ):
        saveFolder="test-onGA-evaluation"
        gaScans = get_data_from_file('/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/GA-ContainingBScans.txt')
        gaScans,gaBScanIds = separate_scan_address_and_bScanId(gaScans)
    else:
        saveFolder="test-evaluation"
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
    #            print "Skip ",d2
                continue
    ##        print "Process ",d2
            #continue
            for d3 in os.listdir(path+'/'+d1+'/'+d2):
                
             #   if(counter>100):
             #       break
      ##          print "Working dir: ", path+'/'+d1+'/'+d2+'/'+d3
               # if(target_folder!="" and params['inDir']+'/'+d1+'/'+d2+'/'+d3 != target_folder):
               #     print params['inDir']+'/'+d1+'/'+d2+'/'+d3
               #     continue
              
                if( justEvaluateGAScans and not path+d1+'/'+d2+'/'+d3 in gaScans):
                     continue
                indxGARange=list()
                if( justEvaluateGAScans and skipDrusenFreeBscan):
                    indxGA=gaScans.index(path+d1+'/'+d2+'/'+d3)
                    indxGARange=(int(gaBScanIds[indxGA].split('-')[0]),int(gaBScanIds[indxGA].split('-')[1]))
       ##         print "=================================="
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
                    if(justEvaluateGAScans and skipDrusenFreeBscan and (fnum<indxGARange[0] or fnum>indxGARange[1])):
                        continue
                    ind.append(fnum)
                    print d2, d3, filename.split('/')[-1],
                    
                    drusLocPath = "/home/rasha/Desktop/DataFromUniClinic/Input-With-Drusen-Label/"
                    filenameDrusenLoc = drusLocPath + d1+'/'+d2+'/'+d3+'/'+f
                    gtDrusenLocation  = io.imread(filenameDrusenLoc)>0
                    gt = io.imread(filename)
   #                 pr = io.imread(predPath + 'Fold0' + str(fold_num) +'/'+fold_type+'/'+prfolder+'/'+d1+'/'+d2+'/'+d3+'/'+str(fnum)+'-binmask.tif')
                    f_layers = f.replace('-BinSeg.tif','-comp_layers.png')
                    pred_file_path = predPath + 'matlab_images/' + d1+'/'+d2+'/'+d3+'/'+f_layers
                    
                    if not os.path.exists(pred_file_path):
      ##                  print pred_file_path, "does not exist"
                        print
                        continue
                    
                    pr = io.imread(pred_file_path)
                    
                    gtDrusenCheck=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)
                    gtDrusenCheck = filter_drusen_by_size(gtDrusenCheck)
                  #  print "$$$$$$$$$$$$$$$ ", np.sum(gtDrusenCheck.astype(float))
                    #show_image(gtDrusenCheck.astype(float))
                    if( skipDrusenFreeBscan):
                        if(np.sum(gtDrusenCheck.astype(float))==0.0):
            #                print "Skip drusen free B-scan"
                            continue
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
                            if( fold_type == 'chen' or fold_type=='wiens'):
                                pr = filter_drusen_by_size(pr)
                            #show_images([gt,gtDrusenLocation,pr],1,3)
                            l_IoU = compute_IoU(gt, gtDrusenLocation, pr, gtType=fold_type,label=label)
                            measure.append(l_IoU)
                        
                    elif( calcMeasure == 'ADAD'):

                        if( fold_type == 'area' or fold_type=='layer'):
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
                        elif( fold_type == 'chen' or fold_type == 'wiens'):
                            gtDrusenLocation=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)
                            gtDrusenLocation=filter_drusen_by_size(gtDrusenLocation)
                            prDrusenLocation = filter_drusen_by_size(pr)
                       
                        if( fold_type=='layer' ):
                            gtDrusenLocation=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)
                        #show_images([gt,gtDrusenLocation,prN,dru],2,2)
                        l_ADAD = compute_ADAD( gt, gtDrusenLocation, prDrusenLocation )
                        measure.append(l_ADAD)
                    elif( calcMeasure == 'SDAD'):

                        if( fold_type == 'area' or fold_type=='layer'):
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
                        #    gtDrusenLocation=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)
                        elif( fold_type == 'chen' or fold_type=='wiens' ):
                            gtDrusenLocation=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)
                            prDrusenLocation = filter_drusen_by_size(pr)
                        if( fold_type=='layer' ):
                            gtDrusenLocation=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)
                      #  show_images([gt,gtDrusenLocation],1,2)
                        l_SDAD = compute_SDAD( gt, gtDrusenLocation, prDrusenLocation )
                        measure.append(l_SDAD)   
                    elif( calcMeasure == 'OR'):
                        drusLocPath = "/home/rasha/Desktop/DataFromUniClinic/Input-With-Drusen-Label/"
                        filenameDrusenLoc = drusLocPath + d1+'/'+d2+'/'+d3+'/'+f
                        gtDrusenLocation  = io.imread(filenameDrusenLoc)>0
                        if( fold_type == 'area' or fold_type=='layer'):
                            #prDrusenLocation = get_drusens_from_rpe_to_bm(pr,useWarping=True)
                            
                            prN = denoise_BM(pr, farDiff=10,max_deg = 5, it = 5)
                            #area = find_area_between_seg_lines(prN)
                            dru = compute_drusen_mask(prN)
                            start=timeit.default_timer()
                            dru = filter_drusen_by_size(dru)
                          #  print "Time Elapsed is=", timeit.default_timer()-start
                            #prDrusenLocation = get_drusens_from_rpe_to_bm(pr,useWarping=True)
                            prDrusenLocation = dru
                        elif( fold_type == 'drusen' ):
                            prDrusenLocation = pr
                        elif( fold_type == 'drusenNrpe' ):
                            prDrusenLocation = pr
                            prDrusenLocation = filter_drusen_by_size(prDrusenLocation)
                        elif( fold_type == 'chen' or fold_type=='wiens' ):
                            gtDrusenLocation=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)
                            prDrusenLocation = filter_drusen_by_size(pr)
                        if( fold_type=='layer' ):
                            gtDrusenLocation=remove_nrpe_to_bm_part_from_gt(gt, gtDrusenLocation)   
                        #show_images([gt,gtDrusenLocation,prN,dru],2,2)
                        l_OR = compute_OR(  gt,gtDrusenLocation, prDrusenLocation )
                        measure.append(l_OR)
                        
                    
                    counter += 1
                  #  show_images([gt, pr, gtDrusenLocation,prDrusenLocation],2,2)
    if( justDrusen ):
        nsave = predPath + 'Fold0' + str(fold_num) +'/'+fold_type+'/test-evaluation-just-for-drusen/'
    else:
        nsave = predPath + 'Fold0' + str(fold_num) +'/'+fold_type+'/'+saveFolder+'/'
        
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
    ff = open(nsave+calcMeasure+str(label)+".txt",'w')
    ff.write("mean:"+str(mean)+"\n")
    ff.write("std:"+str(std)+"\n")
    ff.close()

    print calcMeasure, mean, std
    return mean, std

def iterate_in_folders_for_projection_img(path, savePath, fold_num, fold_type):
    target_folder = "" 
    prfolder = ""
    if( fold_type == "layer" ):
        prfolder = "test-output-shortPath"
    else:
        prfolder = "test-output"
        
    gaScans = list()
    gaBScanIds = list()
    saveFolder=""
    saveFolder="test-evaluation"
    folders = [f for f in listdir(path) if isfile(join(path, f))]
    counter = 0
    subject_id_file="/home/rasha/Desktop/DataFromUniClinic/Multi-fold-training-data/Fold0"+str(fold_num)+"/test_subjects.txt"
    subject_ids = get_subject_ids_to_evaluate(path=subject_id_file)
    processed_subjects = set()
    measure = []
    for d1 in os.listdir(path):
        if(('Fold0'+str(fold_num))!=d1):
            continue
        for d2 in os.listdir(path+'/'+d1):
            for d3 in os.listdir(path+'/'+d1+'/'+d2+'/test-output'):
               
                print d2,d3
                #continue
                for d4 in os.listdir(path+'/'+d1+'/'+d2+'/test-output/'+d3):
                    for d5 in os.listdir(path+'/'+d1+'/'+d2+'/test-output/'+d3+'/'+d4):
                 #   if(counter>100):
                 #       break
                        print "Working dir: ", path+'/'+d1+'/'+d2+'/test-output/'+d3+'/'+d4+'/'+d5
                        print os.path.exists(path+'/'+d1+'/'+d2+'/test-output/'+d3+'/'+d4+'/'+d5)
                        v=0
                        if( len(d5.split('_'))==2):
                            v=d5.split('_')[-1]
                        elif( len(d5.split('-'))==2):
                            v=d5.split('-')[-1]
                        else: 
                            continue
                        if(int(v)<100):
                            continue
                        if( not os.path.exists(path+'/'+d1+'/'+d2+'/test-output/'+d3+'/'+d4+'/'+d5)):
                            continue
                       # if(target_folder!="" and params['inDir']+'/'+d1+'/'+d2+'/'+d3 != target_folder):
                       #     print params['inDir']+'/'+d1+'/'+d2+'/'+d3
                       #     continue
                      
                        
                         
                        print "=================================="
                        rawstack = []
                        ind = []
                        ftype=fold_type
                        fnum = fold_num
                        prfolder="test-output"
                        if(ftype=='layer'):
                            prfolder="test-output-shortPath"
                        scanname=d3+'/'+d4+'/'+d5
                        #"71-78/MOD076/170816_145"
                        b_scans    = read_b_scans( "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/Fold0"+str(fnum)+"/"+ftype+"/"+prfolder+"/"+scanname ) # For GA exp 4-18/MOD006/030214_145 / for warping
                        gt_b_scans = read_b_scans( "/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/"+scanname , "BinSeg.tif")
                        in_b_scans = read_b_scans( "/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/"+scanname,"Input.tif")
                     #   for ii in range(gt_b_scans.shape[2]):
                      #      show_image(gt_b_scans[:,:,ii])
                        k = produce_drusen_projection_image( in_b_scans, gt_b_scans, useWarping=True )
                        k /= np.max(k) if np.max(k) != 0.0 else 1.0
                        #show_image(k)
                        misc.imsave(savePath+d3+'-'+d4+'-'+d5+'.png',k)
                        '''
                        for f in os.listdir(path+'/'+d1+'/'+d2+'/'+d3):
                            filename = path+'/'+d1+'/'+d2+'/'+d3+'/'+f
                        
                            ftype = filename.split('/')[-1].split('-')[-1]
                            if( ftype != "BinSeg.tif"):
                                continue
                            
                            counter += 1
                      
                      '''
                          
        
    
def compute_begin_end_trim_mask(mask):
    trimMask = np.ones(mask.shape)
    h, w = mask.shape
    s=[[1,1,1],[1,1,1],[1,1,1]]
    cca, numLb = sc.ndimage.measurements.label(mask,structure=s)
    labels  = np.unique( cca )
    max_ws  = np.zeros( cca.shape )  
    bg_lbl  = get_label_of_largest_component( cca )
  
    for l in labels:
        if( l != bg_lbl ):
            y, x = np.where( cca == l )
            minX = np.min(x)
            maxX = np.max(x)
            
            if( maxX-minX>6 ):
                trimMask[:,minX:min(minX+4,w)]=0
                trimMask[:,max(0,maxX-3):maxX+1]=0
            else:
                length = int((maxX-minX)/3.0)
                trimMask[:,minX:min(minX+length,w)]=0
                trimMask[:,max(0,maxX-length):maxX+1]=0
    trimMask[:,:7]=1
    trimMask[:,-7:]=1
    return trimMask
def find_start_end_points(mask):
    startPointToLabelMap = dict()
    s=[[1,1,1],[1,1,1],[1,1,1]]
    cca, numLb = sc.ndimage.measurements.label(mask,structure=s)
    labels  = np.unique( cca )
    max_ws  = np.zeros( cca.shape )  
    bg_lbl  = get_label_of_largest_component( cca )
    starts = list()
    ends   = list()
    edges  = list()
    for l in labels:
        if( l != bg_lbl ):
            y, x = np.where( cca == l )
            if( y[np.argmin(x)] == y[np.argmax(x)] and np.min(x)==np.max(x)):
                continue
          #  print "=========>",l
          #  print y[np.argmin(x)],np.min(x)
          #  print y[np.argmax(x)],np.max(x)
           # show_image(cca==l)
            starts.append((y[np.argmin(x)],np.min(x)))
            ends.append((y[np.argmax(x)],np.max(x)))
            edges.append(((y[np.argmin(x)],np.min(x)),(y[np.argmax(x)],np.max(x))))
            startPointToLabelMap[(y[np.argmin(x)],np.min(x))] = l
    return starts, ends, edges, cca, startPointToLabelMap
    
def make_layer_continuous(mask):
    h, w = mask.shape
    res  = np.copy( mask.astype('float') )
    res.fill(0.)
    prev = -1
 #   show_image(mask)
    for j in range(w):
        col = mask[:, j]
        loc = np.where(col > 0)[0]
        #print loc
        if( len(loc) > 0 ):
            #print j, loc
            res[np.max(loc),j] = 1.0
            if( prev== -1 ):
                prev = np.max(loc)
                
          
            else:
                res[np.min(loc):prev+1, j] = 1.
                res[prev:np.min(loc)+1, j] = 1.
                res[prev:np.max(loc)+1, j] = 1.
                res[np.max(loc):prev+1, j] = 1.
                
            prev = np.max(loc)
    return res
    
def generate_graph(mask, n=3):
    h, w = mask.shape    
    
    # Thin the layers to 1 pixel
    mask = skmorph.skeletonize(mask)
    
    # Filter the mask and keep the lowest linesegments
    filMask = np.copy(mask)
    sumV = np.sum(mask, axis=0)
    xDble = np.where(sumV>1)[0]
    for i in xDble:
        col   = mask[:,i]
        yDble = np.where(col>0)[0]
       # print i,yDble,np.argmax(yDble)
        yDble = np.delete(yDble,np.argmax(yDble))
        for j in yDble:
            filMask[j,i] = 0.0
    
    # Find start and end auxilary points
    yE,xE = normal_RPE_estimation(filMask*255,degree=5,useWarping=False)
    estimatedLayer = np.zeros(mask.shape)
    y,x  = np.where(mask>0)
    #show_images([mask],1,1)
    
    # Auxilary points
    pStrt = Point(-1,yE[np.argmin(xE)],1)
    pEnd  = Point( w,yE[np.argmax(xE)],1)
    
    pStrt.printPoint()
    pEnd.printPoint()
    
    starts, ends, edges, cca, startPointToLabelMap = find_start_end_points( mask )
    #ends.insert(0,(pStrt.get_y(),pStrt.get_x()))
    #starts.append((pEnd.get_y(), pEnd.get_x()))
    
    pointToId = dict()
    idToPoint = dict()
    
    starts.sort(key=lambda tup: tup[1])
    ends.sort(key=lambda tup: tup[1])
    
    #print starts
    #print "========================="
    #print ends
    #print startPointToLabelMap
    nId = 0
    G=nx.Graph()
    for p in starts:
        pointToId[p] = nId
        idToPoint[nId] = p
        G.add_node(nId)
        nId += 1
    for p in ends:
        pointToId[p] = nId
        idToPoint[nId] = p
        G.add_node(nId)
        nId += 1
        
    ends.insert(0,(pStrt.get_y(),pStrt.get_x()))
    starts.append((pEnd.get_y(), pEnd.get_x()))
    sourceId = nId    
    pointToId[ends[0]] = nId
    idToPoint[nId] = ends[0]
    G.add_node(nId)
    
    nId += 1    
    targetId = nId
    pointToId[starts[-1]] = nId
    idToPoint[nId] = starts[-1]
    G.add_node(nId)
    
    for pe in ends:
        countN = 0
        for ps in starts:
            #print pe,ps
            if( pe[1] > ps[1] or countN >= n):
                continue
            
            dist = np.linalg.norm(np.asarray(ps)-np.asarray(pe))
         #   print pointToId[pe],pointToId[ps],dist
            G.add_edge(pointToId[pe],pointToId[ps],weight=dist)
            countN+=1
            
    for e in edges:
        pe = e[0]
        ps = e[1]
      #  print pointToId[pe],pointToId[ps], 0
        G.add_edge(pointToId[pe],pointToId[ps],weight=0.0)
      
    shrtPath =  nx.dijkstra_path(G,source=sourceId,target=targetId,weight='weight') 
    finalMask = np.zeros(mask.shape)
    aaaa = list()
    for s in shrtPath:
        aaaa.append(idToPoint[s])
   # show_image(mask)
    for i in range(len(shrtPath)):
       if(i==0):
           continue
       if(i==len(shrtPath)-1):
           break
       if(i%2==1):
           '''
           print "######################"
           print startPointToLabelMap
           print "----------------A"
           print shrtPath
           print aaaa
           print "----------------B"
           print idToPoint
           
           print "----------------Starts"
           print starts

           print "----------------Ends"
           print ends
           '''
           sPoint = shrtPath[i]
           #print sPoint
          # print "######################"
           
           label = startPointToLabelMap[idToPoint[sPoint]]
           finalMask[cca==label] = 1.0
           
       else:
           ps = idToPoint[shrtPath[i]]
           pe = idToPoint[shrtPath[i+1]]
           y = list([ps[0],pe[0]])
           x = list([ps[1],pe[1]])
           z = np.polyfit(x, y, 1)
           p = np.poly1d(z)
           xs= np.arange(pe[1]-ps[1])+ps[1]
           ys= p(xs)
           finalMask[ys.astype('int'),xs.astype('int')] = 1.0
    #finalMask = skmorph.skeletonize(finalMask)   
    finalMask = make_layer_continuous(finalMask) 
    finalMask = skmorph.skeletonize(finalMask)   
    finalMask = make_layer_continuous(finalMask) 
    #show_images([mask,finalMask],1,2)
    return np.where(finalMask>0)
    '''
    print "%%%%%%%%%%%%%%%%%%%%%%%%"
    print shrtPath       
    print nx.dijkstra_path_length(G,source=sourceId,target=targetId,weight='weight')
    print "#######################"
    print idToPoint
    print "-----------------"
    print G.nodes()
    print G.edges()
    nx.draw(G)
    plt.show(True)  
    #y,x  = np.where(filMask>0) 
    
    
    
    
    
    # Find dist matrix
    points = np.where(mask>0)
    ps     = zip(points[0],points[1])
    iu = np.triu_indices(len(ps),1)
    y = sc.spatial.distance.pdist(ps,metric='euclidean')
    distMat = np.zeros((len(ps),len(ps)))
    distMat[iu]=y
    distMat += distMat.T
    
    G.add_nodes_from(np.arange(len(ps)))
    # Find n nearest neighbours for each point
    for i in range(len(ps)):
        disToN = distMat[i,:]
        # Exclude the point itself
        minInd = disToN.argsort()[1:n+1]
        # Connect each node to the n nearest nodes
        for j in minInd:
            G.add_edge(i,j,weight=distMat[i,j])
        
    #print distMat[10,20],distMat[20,10]
    #print len(y)
 
    
 
    nx.draw(G)
    plt.show(True)
    print G
    #dist = sc.ndimage.distance_transform_edt(mask,)
    #show_images([mask,dist],1,2)
    
    mask = (mask==0).astype('float')*100.0
    p = skg.shortest_path(mask,reach=500,axis=1,output_indexlist=True)
    print p[0]
    path = np.zeros(mask.shape)
    a = [i[0] for i in p[0]]
    b = [i[1] for i in p[0]]
    print max(a)
    print max(b)
    path[a,b]=1.0
    show_images([mask,path],1,2)
    '''
def transform_score_image(scoreImg, gamma=1.0):
    maxPerCol = np.max(scoreImg, axis=0)
    maxPerCol[np.where(maxPerCol==0.)] = 1e-10
    maxImg = np.tile(maxPerCol, scoreImg.shape[0])
    maxImg = maxImg.reshape(scoreImg.shape)
    maxImg[np.where(maxImg==0.0)]=1e-10
    return -1.0 * gamma * np.log10(np.divide(scoreImg,maxImg))
    
def shortest_path_in_score_image(scoreImg):
    #scoreImg = scoreImg/np.max(scoreImg)
    
    mcp = skg.MCP_Geometric(scoreImg)
    starts = np.zeros((scoreImg.shape[0],2))
    starts[:,0]=np.arange(scoreImg.shape[0])
    ends = np.zeros((scoreImg.shape[0],2))
    ends.fill(scoreImg.shape[1]-1)
    ends[:,0]=np.arange(scoreImg.shape[0])
    cumCosts, trace = mcp.find_costs(starts=starts,ends=ends)
    
    yMinEnd = np.argmin(cumCosts[:,-1])
    minPath = mcp.traceback([yMinEnd,scoreImg.shape[1]-1])
    p = np.array(minPath).T
    pathImg=np.zeros(scoreImg.shape,dtype='int')
    pathImg[p[0],p[1]]=1.
    return pathImg
  #  show_images([cumCosts,scoreImg,pathImg],1,3)
    '''
    p,c = skg.shortest_path(scoreImg,reach=5,axis=1,output_indexlist=True)
    a = np.argmin(scoreImg, axis=0)
    p = np.array(p).T
    #print p
    print c
    x = np.arange(scoreImg.shape[1])    
        
#    y = np.argmin(scoreImg,axis=0)
#    x = np.arange(scoreImg.shape[1])
    pathImg=np.zeros(scoreImg.shape,dtype='int')
    pathImg[p[0],p[1]]=1.
    flScore=scoreImg*pathImg
    print flScore[flScore>0]
    #pathImg2 = shortest_path(pathImg)
    show_images([scoreImg,pathImg],1,2)
    '''
    '''
    sc.sparse.csgraph.shortest_path(scoreImg)
    p,c = skg.shortest_path(scoreImg,reach=5,axis=1)
    a = np.argmin(scoreImg, axis=0)
    
    print p
    print c
    x = np.arange(scoreImg.shape[1])
    '''
def shortest_path(segMap):
    img = segMap
    mask = img
    
    imgr = np.asarray((img>0).astype('int'))
        
        

    start = timeit.default_timer()
    yr,xr=generate_graph(imgr)
    print "Time:", timeit.default_timer()-start
    mr = np.zeros(mask.shape)

    mr[yr,xr]=1.
 
    #show_images([img,mr],1,2)
    return mr
    
def shortest_path_for_area_seg_map(segMap, debug=False):
    img = segMap
    mask = img
    
    trimMask = compute_begin_end_trim_mask(mask)
    
    img = extract_seg_layers2(img)*trimMask
    if(len(np.unique(img))>3):
        print "First case",len(np.unique(img))
        imgr = np.asarray((img>=170).astype('int'))
        imgb = np.asarray((img==85).astype('int'))
    else:
        print "Second case",len(np.unique(img))
        imgr = np.asarray((img==255).astype('int'))
        imgb = np.asarray((img==127).astype('int'))
        
    if(debug==True):
        show_images([segMap,trimMask,imgr,imgb],2,2)
    start = timeit.default_timer()
    yr,xr=generate_graph(imgr)
    yb,xb=generate_graph(imgb)
    print "Time:", timeit.default_timer()-start
    mr = np.zeros(mask.shape)
    mb = np.zeros(mask.shape)
    mr[yr,xr]=2.
    mb[yb,xb]=1.
    mr += mb
    #show_images([img,mr],1,2)
    return mr
    
def shortest_path_for_layer_seg_map(segMap, debug=False):
    img = segMap
   
    
    if(len(np.unique(img))>3):

        imgr = np.asarray((img>=2).astype('int'))
        imgb = np.asarray((img==1).astype('int'))
    else:

        imgr = np.asarray((img==2).astype('int'))
        imgb = np.asarray((img==1).astype('int'))
        
    if(debug==True):
        show_images([segMap,imgr,imgb],1,3)
    start = timeit.default_timer()
    yr,xr=generate_graph(imgr)
    yb,xb=generate_graph(imgb)
    print "Time:", timeit.default_timer()-start
    mr = np.zeros(img.shape)
    mb = np.zeros(img.shape)
    mr[yr,xr]=2.
    mb[yb,xb]=1.

    mr += mb
    #show_images([img,mr],1,2)
    return mr   
def get_subject_ids_to_evaluate(path):
    subjects = set()
    with open( path ) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    for c in content:
        subjects.add(int(c.split(',')[0]))
    return subjects
def im2double(img):
    return (img.astype('float64') ) / 255.0

def permute(A, dim):
    return np.transpose( A , dim )
    
def imread(filename):
  #  return io.imread(filename)[0,:,:]
    return io.imread(filename)

def compute_sen_spc_TP_FP_TN_FN( gt, pr):
    dnom=1.0
    #gt[0,0] = 1.0
   # pr[0,0] = 1.0
    #dnom = float(gt.shape[0]*gt.shape[1])
    mGt = gt.astype('float')
    mPr = pr.astype('float')
    mGt[mGt>0]  = 1.0
    mGt[mGt<=0] = 0.0
    mPr[mPr>0]  = 1.0
    mPr[mPr<=0] = 0.0
  
    sim = np.logical_and(mGt , mPr).astype('int')    
    TP  = float(np.sum(sim))/dnom
    sub = ((mPr-mGt)>0.0).astype('int')
    FP  = float(np.sum(sub))/dnom
    
    nGt = 1.0-mGt
    nPr = 1.0-mPr
    
    nsim = np.logical_and(nGt , nPr).astype('int')    
    TN  = float(np.sum(nsim))/dnom
    nsub = ((nPr-nGt)>0.0).astype('int')
    FN  = float(np.sum(nsub))/dnom
    
    sen=TP/float(np.sum((mGt>0.0).astype('int'))) if float(np.sum((mGt>0.0).astype('int')))!=0.0 else TP
    spc  =TN/float(np.sum((mGt==0.0).astype('int'))) if float(np.sum((mGt==0.0).astype('int')))!=0.0 else 1.0-TN
    #show_images([gt,pr,sim,sub,nsim,nsub],3,2)
    return sen, spc, TP, FP, TN, FN
def draw_histogram(data):
    n, bins, patches = plt.hist(data, 50, normed=1, facecolor='blue', alpha=0.5)

    # add a 'best fit' line
    y = mlab.normpdf( bins, 0.5, 1.5)
   # l = plt.plot(bins, y, 'r--', linewidth=1)
   # plt.yscale('log', nonposy='clip')
    plt.xlabel('Probability')
    plt.ylabel('Normalized frequency')
    plt.title('Probability histogram')
    plt.axis([0, 1, 0, 1.0])
    #plt.grid(True)
    
    plt.show(True)
    
def draw_histogram_OR_ADAD(data,nbins,title="",savePath="",logScale=True):
    n, bins, patches = plt.hist(data, nbins, normed=1, facecolor='blue', alpha=0.5)

    # add a 'best fit' line
    y = mlab.normpdf( bins, 0.5, 1.5)
   # l = plt.plot(bins, y, 'r--', linewidth=1)
    if(logScale):
        plt.yscale('log', nonposy='clip')
        
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(title)
    #plt.axis([0, 1, 0, 1.0])
    #plt.grid(True)
    if( savePath != "" ):
        plt.savefig(savePath)
        plt.close()
        return
    plt.show(True)
def compute_ROC(replaceGA=False,useShortestPath=False,useMultiThresholds=True,\
                fold_num =5,fold_type = 'drusenNrpe',computeTestOutput=False):
    
    params = dict()
    
    if( useShortestPath ):
        saveToDir='test-output-shortPath'
    else:
        saveToDir='test-output'
    
    thresholds=dict()
    # drusen or area or chen or drusenNrpe or layer
    params['inDir']         = '/home/rasha/Desktop/DataFromUniClinic/Input-With-Label';
    params['inDir2']        = '/home/rasha/Desktop/DataFromUniClinic/Input-With-Drusen-Label';
    params['outDir']        = "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/Fold0"+str(fold_num)+"/"+fold_type+"/"+saveToDir+"/";
    params['outDir2']        = "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/Fold0"+str(fold_num)+"/"+fold_type+"/test-score-output/";
    params['outDir3']        = "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/Fold0"+str(fold_num)+"/"+fold_type+"/test-multi-threshold-output/";
   
    params['netname']       = 'phseg_v5';
    params['normImage']     = False;
    params['scaleImage']    = 1;
    params['zeroCenter']    = False
    params['nTiles']        = 2;
    params['gpu_or_cpu']    = 'gpu';
    params['useFillHoles']  = 1;
    params['minSegmAreaPx'] = 500;
    params['FOI_E']         = 50; 
    params['version']       = 'new' # old, new
    if( fold_type=='drusenNrpe' or fold_type=='drusen'):
        params['type']          = 'drusen'#area_seg or drusen or layer
    elif( fold_type=='area'):
        params['type']          = 'area_seg'#area_seg or drusen or layer
    elif(fold_type=='layer'):
        params['type']          = 'layer'
    params['thresholds']    = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,\
               0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    for t in params['thresholds']:
        thresholds[t]=dict({'Sen':list(),'Spc':list(),'TP':list(),'FP':list(),'TN':list(),'FN':list()})
        
    target_folder = "/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/4-18/MOD006/030214_145" 

    folders = [f for f in listdir(params['inDir']) if isfile(join(params['inDir'], f))]


    subject_id_file="/home/rasha/Desktop/DataFromUniClinic/Multi-fold-training-data/Fold0"+str(fold_num)+"/test_subjects.txt"

    subject_ids = get_subject_ids_to_evaluate(path=subject_id_file)
    processed_subjects = set()


    for d1 in os.listdir(params['inDir']):
        for d2 in os.listdir(params['inDir']+'/'+d1):
            s_id = int(d2[3:6])
            if( True ):
                if( not s_id in subject_ids or s_id in processed_subjects):
                    print "Skip ",d2
                    continue
            else:
                if( not s_id in processed_subjects):
                    print "Skip ",d2
                    continue
            #print "Process ",d2
            #continue
            tt = 0
            for d3 in os.listdir(params['inDir']+'/'+d1+'/'+d2):
                print "Working dir: ", params['inDir']+'/'+d1+'/'+d2+'/'+d3
             #   print target_folder
             #   print params['inDir']+'/'+d1+'/'+d2+'/'+d3 
                if( replaceGA ):
                    if(not params['inDir']+'/'+d1+'/'+d2+'/'+d3+'/' in gaTargetFolders ):
                        print '/'+d1+'/'+d2+'/'+d3+'/'
                        continue
                if(target_folder!="" and params['inDir']+'/'+d1+'/'+d2+'/'+d3 != target_folder):
               #     print params['inDir']+'/'+d1+'/'+d2+'/'+d3
                    continue
              ##  
                print "=================================="
                rawstack = []
                ind = []
                rawstackchen = []
                
                rawstackDrusen=[]
                for f in os.listdir(params['inDir']+'/'+d1+'/'+d2+'/'+d3):
                    filename = params['inDir']+'/'+d1+'/'+d2+'/'+d3+'/'+f
                 #   if( target_folder!="" and params['inDir']+'/'+d1+'/'+d2+'/'+d3 != target_folder):
                 #       break
                    
                    #filename = filename.split('/')[-1]
                    ftype = filename.split('/')[-1].split('-')[-1]
                    
                    if( ftype != "Input.tif"):
                        continue
                    fnum = int(filename.split('/')[-1].split('-')[0])
                   # print f
                    filename=params['inDir']+'/'+d1+'/'+d2+'/'+d3+'/'+str(fnum)+'-'+f.split('-')[1]+'-BinSeg.tif'
                    #if( fold_type=='drusenNrpe'):
                    filenameDrusen=params['inDir2']+'/'+d1+'/'+d2+'/'+d3+'/'+str(fnum)+'-'+f.split('-')[1]+'-BinSeg.tif'
                    drusenRaw=im2double(imread(filenameDrusen))
                    rawstackDrusen.append(permute(drusenRaw, (1,0)))
                 #   if(fnum < 55 or fnum > 70):
                 #       continue
                    ind.append(fnum)
                    #print fnum, filename.split('/')[-1]
                    raw = permute(im2double(imread(filename)), (1,0))
                    
                  #  show_image(raw)
                  #  print tt,fnum
                  #  tt+=1
                    rawSize = raw.shape
                    #if( rawstack == [] ):
                       #  rawstack = raw
                         
                    #else:
                         #rawstack = np.dstack((rawstack, raw))
                    rawstack.append(raw)
                    rawstackchen.append(permute(raw, (1,0)))
                   # print "##########",len(rawstack)
              #  if( len(rawstack.shape) < 3 ):
              #      rawstack = rawstack.reshape((rawstack.shape[0], rawstack.shape[1], 1))
                    
                #print ind   
                ds = len(rawstack)
                print rawSize
    
                #if( target_folder!="" and params['inDir']+'/'+d1+'/'+d2+'/'+d3 == target_folder):
                if( len(rawstack)>0):
                    rawstack = np.asarray(rawstack).transpose((1,2,0))
                    print rawstack.shape
                    
                   # if( fold_type=='drusenNrpe'):
                    rawstackDrusen=np.asarray(rawstackDrusen).transpose((1,2,0))
                else:
                    continue
                
                if( len(rawstackchen)>0):
                    rawstackchen = np.asarray(rawstackchen).transpose((1,2,0))
                    print rawstackchen.shape
                else:
                    continue
      
                data = np.reshape( rawstack.astype('float32'),\
                				[rawstack.shape[0], rawstack.shape[1], 1, rawstack.shape[2]])
                dataDrusen=list()
               # if(fold_type=='drusenNrpe'):
                dataDrusen= np.reshape( rawstackDrusen.astype('float32'),\
                    [rawstackDrusen.shape[0], rawstackDrusen.shape[1], 1, rawstackDrusen.shape[2]])
                    
                scores = read_pickle_data(params['outDir2']+'/'+d1+'/'+d2+'/'+d3+'/scores.pkl')
                scores = np.exp(scores)
                for jj in range(scores.shape[3]):
                    meanImg = np.sum(scores[:,:,:,jj], axis=2)
                    for kk in range(scores.shape[2]):
                        scores[:,:,kk,jj] = scores[:,:,kk,jj]/meanImg
                        #show_image(scores[:,:,1,jj])
               # draw_histogram(scores[:,:,1,:].ravel())
             #  scores[0,0,1,:]      =1.0
                # Compute True/False Psitive/Negative
                for ii in range(scores.shape[3]):
                  #  if( ii !=55):
                  #      continue
                 #   show_images([scores[:,:,1,ii].T,scores[:,:,2,ii].T,scores[:,:,3,ii].T],1,3)
                 #   continue
                    if( computeTestOutput ):
                        gtLines=np.round(data[:,:,0,ii].T*255.0)
                        gt=find_area_between_seg_lines(gtLines)
                        pr=np.argmax(scores[:,:,:,ii],axis=2).T
                       # pr=sc.ndimage.measurements.morphology.binary_erosion(pr,iterations=1)
                        show_images([scores[:,:,1,ii].T,pr,gt],1,3)
                    else:
                        for t in params['thresholds']:
                            #if( t!=0.05):
                            #    continue
                            pr=(scores[:,:,1,ii].T>=t).astype('float')
                            gtDru=dataDrusen[:,:,0,ii].T
                            if(fold_type=='drusenNrpe'):
                                gt=dataDrusen[:,:,0,ii].T
                                gtLines=np.round(data[:,:,0,ii].T*255.0)
                                gt = remove_nrpe_to_bm_part_from_gt(lineGt=gtLines, drusenGt=gt)
                                gtDru=gt
                                #show_images([gt,gtLines,pr],1,3)
                                
                            elif(fold_type=='area'):
                                gtLines=np.round(data[:,:,0,ii].T*255.0)
                                gt = find_area_between_seg_lines(gtLines)
                                
                                gtDru=dataDrusen[:,:,0,ii].T
                               # gtLines=np.round(data[:,:,0,ii].T*255.0)
                                gtDru = remove_nrpe_to_bm_part_from_gt(lineGt=gtLines, drusenGt=gtDru)
                                
                                    
                          #  print "-------t=",t
                      #      show_images([gt,pr],1,2,["GT","Pr (t="+str(t)+")"])
                            if(np.sum(gtDru.astype(float))==0.0):
                                continue
                            sen,spc,TP,FP,TN,FN=compute_sen_spc_TP_FP_TN_FN(gt,pr)
                                
                            thresholds[t]['Sen'].append(sen)
                            thresholds[t]['Spc'].append(spc)
                            thresholds[t]['TP'].append(TP)
                            thresholds[t]['FP'].append(FP)
                            thresholds[t]['TN'].append(TN)
                            thresholds[t]['FN'].append(FN)   
                          #  print "Sensitivity=", sen
                          #  print "1-Specifity=",1.0-spc
                        if(False):
                            tpr=list()
                            fpr=list()
                            for t in params['thresholds']:
                                tmpSen=np.mean(np.asarray(thresholds[t]['Sen']))
                                tmpSpc=np.mean(np.asarray(thresholds[t]['Spc']))
                                tpr.append(tmpSen)
                                fpr.append(1.0-tmpSpc)
                            auc=sklearnm.auc(fpr,tpr)
                            
                            fig = plt.figure(figsize=(8.0, 8.0))
                           
                            ax = fig.add_subplot(1,1,1)
                            ax.plot(fpr, tpr, color='r', label='(area = %0.4f)' % auc)
                            plt.xlim([-0.01, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Different cluster number comparison')
                            plt.legend(loc="lower right")
                            plt.show(True)
                          #  show_images([pr,gt],1,2)
    if( fold_type=='drusenNrpe'):
        outfilename4 = "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/drusenFold0"+str(fold_num)
    else:
        outfilename4 = "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/areaFold0"+str(fold_num)
    write_pickle_data(outfilename4,thresholds)
                        #break
                    #show_images([scores[:,:,1,ii].T,data[:,:,0,ii].T],1,2)
    '''
    tpr=list()
    fpr=list()
    for t in params['thresholds']:
        tmpSen=np.mean(np.asarray(thresholds[t]['Sen']))
        tmpSpc=np.mean(np.asarray(thresholds[t]['Spc']))
        tpr.append(tmpSen)
        fpr.append(1.0-tmpSpc)
    auc=sklearnm.auc(fpr,tpr)
    
    fig = plt.figure(figsize=(8.0, 8.0))
   
    ax = fig.add_subplot(1,1,1)
    ax.plot(fpr, tpr, color='r', label='(area = %0.2f)' % auc)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Different cluster number comparison')
    plt.legend(loc="lower right")
    plt.show(True)
    '''
def draw_ROC_curve(pathT, thresholds,fold_type='drusen'):
    #thresholds=thresholds[::-1]
    types=['area','drusen']
    color=['b','#FFCC00']
    aucs=list()
    tprs=list()
    fprs=list()
    for fold_type in types:
        path=pathT+"/"+fold_type
        tpr=list()
        fpr=list()
        multiData=list()
        for f in os.listdir(path):
            fileName=path+'/'+f
            print fileName
            data=read_pickle_data(fileName)
            multiData.append(data)
      
        numFolds=float(len(multiData))
        print "################", fold_type
        for t in thresholds:
            tprS=0.0
            fprS=0.0
            print "---------t:", t
           # if( t==1.0):
            #    continue
            
            for i in range(int(numFolds)):
                print "+++Fold:", i
                senAvg=np.mean(np.asarray(multiData[i][t]['Sen']))                
                spcAvg=np.mean(np.asarray(multiData[i][t]['Spc']))
                print np.asarray(multiData[i][t]['Sen'])
                tprS+=senAvg
                fprS=fprS+(1.0-spcAvg)
            print t,"    -------    ", tprS/numFolds, fprS/numFolds
            
            tpr.append(tprS/numFolds)
            fpr.append(fprS/numFolds)
        tpr.append(0.0)
        fpr.append(0.0)
      #  tpr.insert(0,1.0)
      #  fpr.insert(0,1.0)
        auc=sklearnm.auc(fpr,tpr)
        aucs.append(auc)
        fprs.append(fpr)
        tprs.append(tpr)
        
    fig = plt.figure(figsize=(8.0, 8.0))
   
    ax = fig.add_subplot(1,1,1)
    i=0
    ax.plot((0,1.0), (0.0,1.0), color='#ff5900', linestyle='--')
    for fold_type in types:
        ax.plot(fprs[i], tprs[i], color=color[i], label='('+fold_type+' segmentation = %0.3f)' % aucs[i])
        plt.plot(fprs[i], tprs[i], color=color[i],marker='o')
        for label, x, y in zip(thresholds, fprs[i], tprs[i]):
            plt.annotate(label, xy=(x, y), xytext=(-50, 0),\
                textcoords='offset points', ha='right', va='bottom',fontsize=8.5,\
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        for t in range(len(thresholds)):
            if(thresholds[t]==0.5):
                print "<<<<<<<<<<<<<<<<<HERE>>>>>>>>>>>>>>>>>>>"
                plt.plot(fprs[i][t], tprs[i][t], color='r',marker='*',markersize=12.0)
        i+=1
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Segmentation map comparison')
    plt.legend(loc="lower right")
    plt.show(True)

def warp_layer_flat(bScans,denoise=False):
    for i in range(bScans.shape[2]):
        if(denoise):
            bScans[:,:,i] = denoise_BM(bScans[:,:,i],\
                        farDiff=10,max_deg = 5, it = 5)
        cornerFilter=np.ones((bScans.shape[0],bScans.shape[1]))
        cornerFilter[:,:5]=0.0
        cornerFilter[:,-5:]=0.0
        bScans[:,:,i] = np.flipud(warp_BM(bScans[:,:,i],\
                         returnWarpedImg=True))*cornerFilter
    return bScans

def compute_measures_in_full_scan(gt,pr):
    gtDru=find_drusen_in_full_scans(gt)
    prDru=find_drusen_in_full_scans(pr)
    
    projImg=np.max(gtDru, axis=0)    
    
    adadM=0.0
    orM  =0.0
    
    # Compute ADAD    
    numProj=np.sum(projImg)
    adadM=np.abs(np.sum(gtDru)-np.sum(prDru))
    adadM=adadM/numProj if numProj>0 else adadM
    
    # Compute OR
    intersect=np.sum(gtDru*prDru)
    union=np.sum(((gtDru+prDru)>0.0).astype(float))
    orM=intersect/union if union>0.0 else 1.0
    
    return adadM,orM

def find_drusen_in_full_scans(scans):
    drusen=np.zeros(scans.shape)
    for i in range(scans.shape[2]):
        dru = compute_drusen_mask(scans[:,:,i])
        dru = filter_drusen_by_size(dru)
        y,x=np.where(dru>0)
        drusen[y,x,i]=1.0
    return drusen
  
def draw_3d_and_measure_all(path, predPath, foldNum, foldType,copy=False):
    
    target_folder = path+"34-45/MOD035/170516_19" 
    prfolder = ""
    if( foldType == "layer" ):
        prfolder = "test-output-shortPath"
    else:
        prfolder = "test-output"
    if(not copy):
        saveFile=open(predPath+'/Fold0'+str(foldNum)+'/'+foldType+'/volumeMeasure.txt','w')
        saveFile.close()
    saveFolder="3d-view"
    folders = [f for f in listdir(path) if isfile(join(path, f))]
    counter = 0
    subject_id_file="/home/rasha/Desktop/DataFromUniClinic/Multi-fold-training-data/Fold0"+str(foldNum)+"/test_subjects.txt"
    subject_ids = get_subject_ids_to_evaluate(path=subject_id_file)
    processed_subjects = set()
    measure = []
    for d1 in os.listdir(path):
        
        for d2 in os.listdir(path+'/'+d1):
            s_id = int(d2[3:6])
            
            if( not s_id in subject_ids or s_id in processed_subjects):
                print "Skip ",d2
                continue
            print "Process ",d2
   
            for d3 in os.listdir(path+'/'+d1+'/'+d2):
                lend4 =len(os.listdir(path+'/'+d1+'/'+d2+'/'+d3))
                if(lend4==0):
                    print "Working dir: ", path+'/'+d1+'/'+d2+'/'+d3
                    continue
                if(not copy):
                    saveFile=open(predPath+'/Fold0'+str(foldNum)+'/'+foldType+'/volumeMeasure.txt','a')
               # if(target_folder!="" and path+d1+'/'+d2+'/'+d3 != target_folder):
               #     print ">>>"+path+'/'+d1+'/'+d2+'/'+d3
               #     print "***",target_folder
               #     continue
                
                    b_scans    = read_b_scans( predPath +'/Fold0'+str(foldNum)+'/'+\
                                 foldType+'/'+prfolder +'/'+d1+'/'+d2+'/'+d3)             
                    gt_b_scans = read_b_scans( path+'/'+d1+'/'+d2+'/'+d3, "BinSeg.tif")
                 #   in_b_scans = read_b_scans( "/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/"+scanname,"Input.tif")
                    mGt=np.copy(gt_b_scans)
                    mPr=np.copy(b_scans)
                    b_scans=warp_layer_flat(b_scans, denoise=True)
                    gt_b_scans=warp_layer_flat(gt_b_scans)
                    if not os.path.exists(predPath+'/Fold0'+str(foldNum)+'/'+\
                                 foldType+'/'+saveFolder+'/'+d1+'/'+d2+'/'+d3+'/'):
                        os.makedirs(predPath+'/Fold0'+str(foldNum)+'/'+\
                                 foldType+'/'+saveFolder+'/'+d1+'/'+d2+'/'+d3+'/')
                    adadM,orM=compute_measures_in_full_scan(mGt, mPr)
                    
                    saveFile.write(predPath+'/Fold0'+str(foldNum)+'/'+\
                                     foldType+'/'+saveFolder+'/'+d1+'/'+d2+'/'+d3+\
                                     '/'+'-'+str(adadM)+','+str(orM)+'\n')
                    saveFile.close()
                    show_PED_volume( [gt_b_scans,b_scans],[ 0.1,0.1],titles=["GT","Pr(ADAD,OR)=("+str(round(adadM,2))+","+str(round(orM,2))+")"],\
                        savePath=predPath+'/Fold0'+str(foldNum)+'/'+\
                                 foldType+'/'+saveFolder+'/'+d1+'/'+d2+'/'+d3+"/3d") 
                else:
                    threeDView=misc.imread(predPath+'/Fold0'+str(foldNum)+'/'+\
                                 foldType+'/'+saveFolder+'/'+d1+'/'+d2+'/'+d3+"/3d.png")
                    misc.imsave("/home/rasha/Desktop/3dViews/"+d1+'-'+d2+'-'+d3+"-3d.png",threeDView)
                #show_PED_volume( [b_scans], [0.1] )
    
def main_old():
    
    if( False ):
        fold_num  = 5
        fold_type = 'area'
        draw_validation_diagram("/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/Fold0"+str(fold_num)+"/"+fold_type+"/validation/",\
        "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/Fold0"+str(fold_num)+"/"+fold_type+"/")
        exit()
   # evaluate_net_outputs()
   # exit()
   # path = "/home/rasha/Desktop/OCT-Project/"+\
   # "sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/"+\
   # "tools/test3/MOD026-020514_145-HighRes/"
 #   is_bm_always_below_rpe( '/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/71-78/MOD077/150813_145/26-18F81E90-BinSeg.tif' )
 #   exit()
    if( False ):
        compute_ROC(replaceGA=False,useShortestPath=False,useMultiThresholds=True,\
        fold_num =1,fold_type = 'layer')
    if( False ):
        if( True ):
            draw_ROC_curve(pathT="/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/total-evaluation/ROC-Data",\
               thresholds=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,\
               0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],fold_type='drusen')
        else:
            draw_ROC_curve(pathT="/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/total-evaluation/ROC-Data2",\
               thresholds=[0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,\
                                    0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                                    0.2,0.25,0.3,0.35,0.4,0.45,\
                                    0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.81,0.82,0.83,0.84,\
                                   0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,\
                                   0.95,0.96,0.97,0.98,0.99,1.0],fold_type='drusen')
    if( False ):
       # mmm=['IoU','ADAD','OR']
     
        a = ""#-Drusen
        path = "/home/rasha/Desktop/DataFromUniClinic/Input-With"+a+"-Label/"
        predPath = "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/"
        i=5
        iterate_in_folders(path, predPath, fold_num=i, fold_type='wiens',\
               calcMeasure='OR',justDrusen=False, justEvaluateGAScans=False,\
               skipDrusenFreeBscan=False,label = 1)
        exit()
        
    if( True ): 
        mmm=['OR1','ADAD1']
        mmm=['ADAD1','OR1']
        mmm=['OR','ADAD']
        fff=['chen']
        for m in mmm:
            for f in fff:
                saveFolder="/"
                justEvaluateGAScans=False
                if(justEvaluateGAScans):
                    saveFolder=saveFolder+"justGAEvaluation/"
                
                path="/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/"
                save_path = "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/total-evaluation"+saveFolder
                total_evaluation( path=path, fold_type=f, measureName=m, save_path =save_path, justEvaluateGAScans=justEvaluateGAScans)
        exit()
    
    if( False ):
        iterate_in_folders_for_projection_img(path="/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/",\
        savePath="/home/rasha/Desktop/projectionImages/", fold_num=5, fold_type='area')
        exit()
    ftype="layer"
    fnum = 1
    prfolder="test-output"
    if(ftype=='layer'):
        prfolder="test-output-shortPath"
        
    if( False ):
        path = "/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/"
        predPath = "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/"
        draw_3d_and_measure_all(path, predPath, foldNum=5, foldType='layer', copy=True)
        exit()
    scanname="4-18/MOD014/010414_145"#4-18/MOD014/010311_19    4-18/MOD014/010414_145
    b_scans    = read_b_scans( "/home/rasha/Desktop/OCT-Project/sh-net/u-net-release/NetWorks/Net_BN_ML_No_ZC/multi-fold-training/Fold0"+str(fnum)+"/"+ftype+"/"+prfolder+"/"+scanname ) # For GA exp 4-18/MOD006/030214_145 / for warping
    gt_b_scans = read_b_scans( "/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/"+scanname , "BinSeg.tif")
    in_b_scans = read_b_scans( "/home/rasha/Desktop/DataFromUniClinic/Input-With-Label/"+scanname,"Input.tif")
 #   for ii in range(gt_b_scans.shape[2]):
  #      show_image(gt_b_scans[:,:,ii])
 #   k = produce_drusen_projection_image( b_scans, gt_b_scans, useWarping=True )
 #   k /= np.max(k) if np.max(k) != 0.0 else 1.0
 #   show_image(k)
    if( False ):
        img = b_scans[:,:,17]        
        shortest_path_for_area_seg_map(img)
        exit()
        
    if( False ):
        for i in range(b_scans.shape[2]):
            print i
            if( i+1 == 8 ):
                b_scans[:,:,i] = filter_drusen_by_size(b_scans[:,:,i])
                img = np.dstack((in_b_scans[:,:,i],in_b_scans[:,:,i],in_b_scans[:,:,i]))
                y, x = np.where( b_scans[:,:,i] > 0.0 )
                img[y, x,0] = 255.0
                img[y, x,1] = 0.0
                img[y, x,2] = 0.0
                
                misc.imsave("/home/rasha/Desktop/pr-druNrpe.png",img)
        exit()
    if( False):
        for i in range(b_scans.shape[2]):
            if( i+1 == 25 ):
                area = find_area_between_seg_lines(gt_b_scans[:,:,i])
                dru = compute_drusen_mask(gt_b_scans[:,:,i])
                dru = filter_drusen_by_size(dru)
                misc.imsave("/home/rasha/Desktop/miccai-Images/b-scan.png",in_b_scans[:,:,i])
                misc.imsave("/home/rasha/Desktop/miccai-Images/gt-lineSeg.png",gt_b_scans[:,:,i])            
                misc.imsave("/home/rasha/Desktop/miccai-Images/gt-areaSeg.png",area.astype('int'))  
                misc.imsave("/home/rasha/Desktop/miccai-Images/gt-druSeg.png",dru)
                show_images([in_b_scans[:,:,i],gt_b_scans[:,:,i],area,dru],2,2)
        exit()
    
    for i in range(b_scans.shape[2]):
         #if(i>33):
         #    pre = np.copy(b_scans[:,:,i])
         b_scans[:,:,i] = denoise_BM(b_scans[:,:,i], farDiff=10,max_deg = 5, it = 5)
         #if(i>33):
         #    show_images([pre,b_scans[:,:,i]],1,2)
    if(True):
       # b_scans = gt_b_scans
        for i in range(b_scans.shape[2]):
             cornerFilter=np.ones((b_scans.shape[0],b_scans.shape[1]))
             cornerFilter[:,:5]=0.0
             cornerFilter[:,-5:]=0.0
             b_scans[:,:,i] = np.flipud(warp_BM(b_scans[:,:,i], returnWarpedImg=True))*cornerFilter
            # show_image(b_scans[:,:,i])
        show_PED_volume( b_scans, 0.1 )
    #run_chen_method_on_a_pack(in_b_scans)
   # rpe, nrpe = seg_chen( in_b_scans[:,:,2] )

  # for x in rpe[:,0]:
   #    print x
  #  mask = draw_lines_on_mask(rpe, nrpe,in_b_scans[:,:,2].shape)
  #  area_mask = find_area_btw_RPE_normal_RPE(mask)
  #  show_images([in_b_scans[:,:,2], mask, area_mask], 1, 3)
    
    #evaluate_net_outputs()
    draw_drusen_boundary_over_projection_image( in_b_scans,gt_b_scans,b_scans, show=True, scale= 1, input_type='line_segments')
   # exit()
    
    path = "/home/rasha/Desktop/DataFromUniClinic/test-05-04-2017/"
            
    #b_scans = read_b_scans( path + "predictions" )
    #gt_b_scans = read_b_scans( path + "GT" )
    #in_b_scans = read_b_scans( path + "b-scans" )
    
    for d in range(b_scans.shape[2]):
        print "B_scan:"+ str(d)
        
        
        img_gt  = mark_drusen_on_b_scan( in_b_scans[:,:,d], gt_b_scans[:,:,d] )
        gt_info = get_drusen_quantification_info( gt_b_scans[:,:,d] )
        show_image_rgb(img_gt, gt_info, d, path, "-gt")
        
        img_pr  = mark_drusen_on_b_scan( in_b_scans[:,:,d], b_scans[:,:,d] )
        pr_info = get_drusen_quantification_info( b_scans[:,:,d] )
        show_image_rgb(img_pr, pr_info, d, path, "-pr")
        
        overlayed = make_overlay_of_drusen_masks( in_b_scans[:,:,d], gt_b_scans[:,:,d], b_scans[:,:,d] )
        true_pos, false_pos, or_value, adad,iou = compute_true_and_false_positives( gt_b_scans[:,:,d], b_scans[:,:,d] )
        info = "OR = "+ str(or_value)+", ADAD = "+ str(adad) +"\nTrue Positive = " + str(true_pos) + ", False Positive = " + str( false_pos )
        show_image_rgb(overlayed, info, d, path, "-mix")
        
        avg_dist_rpe = compute_curve_distance( gt_b_scans[:,:,d], b_scans[:,:,d], layer_type='RPE')
        avg_dist_bm  = compute_curve_distance( gt_b_scans[:,:,d], b_scans[:,:,d], layer_type='BM')
        
        
        avg_dist_rpe_on_d = compute_curve_distance_on_drusens( gt_b_scans[:,:,d], b_scans[:,:,d], layer_type='RPE')
        avg_dist_bm_on_d  = compute_curve_distance_on_drusens( gt_b_scans[:,:,d], b_scans[:,:,d], layer_type='BM')
        
        seg_dist_info = "Average dist : RPE = "+ str(avg_dist_rpe) + ", BM = "+str(avg_dist_bm)    
        
        seg_dist_info_on_drusen = "Average dist on drusen : RPE = "+ str(avg_dist_rpe_on_d) + ", BM = "+str(avg_dist_bm_on_d)
        seg_di
        st_info = seg_dist_info + '\n' + seg_dist_info_on_drusen
        
        overlay_img   = overlay_prediction_on_gt(in_b_scans[:,:,d], gt_b_scans[:,:,d], b_scans[:,:,d])
        show_image_rgb(overlay_img, seg_dist_info, d, path)
    
    
    b_scans2 = np.copy(b_scans)                       
    for d in range(b_scans.shape[2]):  
      
        b_scans2[:,:,d] = compute_distance_image( b_scans[:,:,d] )
        
        
        
    #b_scans2 = increase_resolution(b_scans2,  1)
 #   kk = 3
#    show_image(b_scans2[:,:,kk])
  #  b_scans2 = sc.ndimage.filters.gaussian_filter1d(b_scans2, sigma=2.5, axis=1)
   # b_scans2 = sc.ndimage.filters.gaussian_filter1d(b_scans2, sigma=2.5, axis=2)
    #show_image(b_scans2[:,:,kk])
    #print ">>>>>>>",b_scans2.shape
    #show_PED_volume( b_scans2, 0.1 )
        
        
def main():

    if( True ):
        a = ""#-Drusen
        path = "/home/rasha/Desktop/DataFromUniClinic/Input-With"+a+"-Label/"
 #       predPath = "/home/rasha/Desktop/resnet_results/"
 #       predPath = '/home/rasha/work/deeplab-public-ver2_diff_weights/drusen_area/features/RESNET-50/val/small/snapshot_latest/'
 #       predPath = '/home/rasha/work/deeplab-public-ver2/drusen_area/features/RESNET-50/val/small/snapshot_/post_densecrf/several_settings/'
        predPath = '/home/rasha/work/deeplab-public-ver2/drusen_area/features/RESNET-50/val/small/snapshot_mon_results'
        
 #       sett = ['2-1-3', '4-1-16', '4-2-1-16', '8-2-1-16', 'no-crf']
        sett = ['']
        for s in sett:
            predPath2 = predPath + s + '/'
            ma = iterate_in_folders(path, predPath2, fold_num=1, fold_type='area',calcMeasure='ADAD',justDrusen=False, justEvaluateGAScans=False, skipDrusenFreeBscan=False, label = 1)
#            mb = iterate_in_folders(path, predPath2, fold_num=1, fold_type='area',calcMeasure='OR',justDrusen=False, justEvaluateGAScans=False, skipDrusenFreeBscan=False, label = 1)
#            mc = iterate_in_folders(path, predPath2, fold_num=1, fold_type='area',calcMeasure='IoU',justDrusen=False, justEvaluateGAScans=False, skipDrusenFreeBscan=False, label = 1)
            print
            print
            print s        
            print 'ADAD', ma
#            print 'OR', mb
#            print 'IoU', mc
            
#        exit()
        

main()
        