#@author zjh

import numpy as np 
import cv2 
from PIL import Image
import random
import os
import datetime
class Nonrigid_deformation():
    def __init__(self,show_point=True):
        self.show_point = show_point
    def deformation(self,label_path=None,save_path=None):
        
        label_img = Image.open(label_path)
        palette = label_img.getpalette()
        gt_arr = np.array(label_img)


        # let us do non-rigid deformation
        N = 5
        Delta = 0.05
        #get the target boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary = cv2.dilate(gt_arr, kernel)-gt_arr
        boundindex = np.where(boundary==1)
        num_index = boundindex[0].shape[0]
        if num_index>N:
            maxH,minH = max(boundindex[0]),min(boundindex[0])
            tarH = maxH - minH
            maxW,minW = max(boundindex[1]),min(boundindex[1])
            tarW = maxW - minW

            # thin plate spline coord num    
            randindex = [random.randint(0,num_index-1) for _ in range(N)]
            sourcepoints=[]
            targetpoints = []
            for i in range(N):
                sourcepoints.append((boundindex[1][randindex[i]],boundindex[0][randindex[i]]))
                x = boundindex[1][randindex[i]]+int(random.uniform(-Delta,Delta)*tarW)
                y = boundindex[0][randindex[i]]+int(random.uniform(-Delta,Delta)*tarH)
                targetpoints.append((x,y))
        
            sourceshape = np.array(sourcepoints,np.int32)
            sourceshape=sourceshape.reshape(1,-1,2)
            targetshape = np.array(targetpoints,np.int32)
            targetshape=targetshape.reshape(1,-1,2)

            matches =[]
            for i in range(0,N):
                matches.append(cv2.DMatch(i,i,0))
            tps= cv2.createThinPlateSplineShapeTransformer()
            tps.estimateTransformation(targetshape, sourceshape,matches)
            no_grid_img=tps.warpImage(gt_arr)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            no_grid_img = cv2.dilate(no_grid_img,kernel)
            if self.show_point:
                for p in sourcepoints:
                    cv2.circle(gt_arr,p,4,(2),2)   
                for p in targetpoints:
                    cv2.circle(no_grid_img,p,4,(2),2)
            no_grid_out = Image.fromarray(no_grid_img)
            gt_out = Image.fromarray(gt_arr)
        else:
            no_grid_out = Image.fromarray(gt_arr)
            gt_out = Image.fromarray(gt_arr)

        #save gt and none_rigid_deform and affine_tran
        no_grid_out.putpalette(palette)
        no_grid_out.save(save_path)
        if self.show_point:
            gt_out.putpalette(palette)
            gt_out.save(label_path[:-4]+'_gt'+'.png')
        no_grid_out.show('Deformation result')
        gt_out.show('source label')
        

    def script(self):
        label_path = '2.png'
        save_path = 'res_2.png'
        self.deformation(label_path,save_path)

        
        
if __name__ == '__main__':

    Nonrigid = Nonrigid_deformation(show_point=True)
    Nonrigid.script()

