# -*- coding: utf-8 -*-
import time
from datetime import datetime, timedelta
import os
import math
import argparse
import warnings
import numpy as np
from skimage.io import imread, imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import onnx
import onnxruntime as rt
import cv2
import pydicom as pyd
from pydicom.dataset import Dataset, FileDataset
from pydicom.encaps import encapsulate
import pandas as pd
import archs
from utils import str2bool, count_params
import losses
import joblib
import matplotlib.pyplot as plt
import copy
import shutil
from glob import glob

arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())

def parse_args_keypoint():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default='True', type=str2bool)
    parser.add_argument('--dataset', default='OAIKneeAP',
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='dcm',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='dcm',
                        help='mask file extension')
    parser.add_argument('--aug', default='True', type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default= 50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--m_class', default=2, type=float,
                        help='4')

    args = parser.parse_args()

    return args

def parse_args_line():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default='True', type=str2bool)
    parser.add_argument('--dataset', default='KneeLineBase',
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='bmp',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='dcm',
                        help='mask file extension')
    parser.add_argument('--aug', default='True', type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default= 50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--m_class', default=8, type=float,
                        help='4')

    args = parser.parse_args()

    return args

def get_max_preds(batch_heatmaps):
    # 在heatmap上面求最大點的座標
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def knee_location_detection(image,pred_img,modelkeypoint,osi,LcMmw,LcMmh):
  
    #keypoint sech knee location
    train_started = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():       
                
            input = torch.from_numpy(image)   
            # compute output            
            outputkeypoint = modelkeypoint(input)[-1]
            
            
            # inference的結果
            outputkeypoint = outputkeypoint.cpu().detach().numpy() 
            
            for ij in range(outputkeypoint.shape[0]): 
                    
                    preds_tmp = outputkeypoint[ij,:,:,:]                    
                    preds_tmp  = preds_tmp[:,:,np.newaxis]
                    preds_tmp = preds_tmp.transpose((2, 0, 1, 3))
                    
                    out_preds, out_maxvals = get_max_preds(preds_tmp)                    
                    out_preds = out_preds.squeeze()
                    out_maxvals = out_maxvals.squeeze()
                    
                    # 
                    for Poi in range(2):
                        out_preds[Poi][0] = out_preds[Poi][0]/512*osi[1]
                        out_preds[Poi][1] = out_preds[Poi][1]/512*osi[0]                

                    
                
        torch.cuda.empty_cache()

    # crop knee image    
    if (out_preds[0][1]-LcMmh) < 0:
        wsx1 = 0
    else:
        wsx1 = out_preds[0][1]-LcMmh

    if (out_preds[0][1]+LcMmh) > osi[0]:
        wsx2 =  osi[0]
    else:
        wsx2 = out_preds[0][1]+LcMmh 

    if (out_preds[0][0]-LcMmw) < 0:
        wsy1 = 0
    else:
        wsy1 = out_preds[0][0]-LcMmw

    if (out_preds[0][0]+LcMmw) > osi[0]:
        wsy2 =  osi[1]
    else:
        wsy2 = out_preds[0][0]+LcMmw
    #print(osi)
    #print(out_preds)
    crop_img = pred_img[int(wsx1):int(wsx2),int(wsy1):int(wsy2),:]
    print('knee location detection time:', time.time()-train_started, 'seconds')
    #print(crop_img)
    
    return out_preds, crop_img

def OA_classification(crop_img,sess):
    try:
        train_started = time.time()
        imknee = cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_LINEAR)
        imknee = imknee[:,:,np.newaxis].astype('float32')
        imknee = imknee.transpose((2, 3, 0, 1))

        #print(imknee.shape)
        input_data = np.ones((1, 3, 224, 224), dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        print("input name", input_name)

        input_shape = sess.get_inputs()[0].shape
        print("input shape", input_shape)

        output_name= sess.get_outputs()[0].name
        print("output name", output_name)
        output_shape = sess.get_outputs()[0].shape
        print("output shape", output_shape)

        #forward model
        res = sess.run([output_name], {input_name: imknee})
        probs = np.array(res)
        probs = probs.squeeze()    
        print(np.argmax(probs))
        print('knee KL classification time:', time.time()-train_started, 'seconds')
        run_index = 1
    except:
        run_index = 0
    return probs,run_index

def OA_detection(image,modelLine,osi,ps):   
    
       

    #init
    LS = np.round(5/ps)

    
    # line processing
    train_started = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with torch.no_grad():       

            input = torch.from_numpy(image)    

            outputLine = modelLine(input)[-1]
            
            
        outputLine = outputLine.squeeze()  
        Fmap = np.zeros((outputLine.shape[1],outputLine.shape[2]))
        FLP = np.zeros((2,1))
        FRP = np.zeros((2,1))
        Fx = []
        Fy = []
        checkI = 0
        values, indices  = torch.max(outputLine[0,:,:],0)
        values_c, indices_c  = torch.max(outputLine[0,:,:]>0.5,0)
        for i in range(outputLine.shape[2]):
            if indices_c[i]:
                Fx.append(i)
                Fy.append(indices[i].cpu().detach().numpy())
                Fmap[indices[i],i] = 255
                if checkI == 0:
                    checkI = 1
                    FRP[0] = indices[i].cpu().detach().numpy()  
                    FRP[1] = i
                else:
                    FLP[0] = indices[i].cpu().detach().numpy()  
                    FLP[1] = i
        
        tmpFC = np.round((FLP[1]+FRP[1])/2)
        FRPC = tmpFC - LS; #右側分界線
        FLPC = tmpFC + LS; #左側分界線

        FRPC41 = np.round((FRP[1] + FRPC) / 2)
        FLPC41 = np.round((FLP[1] + FLPC) / 2)

        plt.imsave('test1F.png',Fmap.astype('uint8'))

        
        Tmap = np.zeros((outputLine.shape[1],outputLine.shape[2]))
        TLP = np.zeros((2,1))
        TRP = np.zeros((2,1))
        Tx = []
        Ty = []
        checkI = 0
        values, indices  = torch.max(outputLine[1,:,:],0)
        values_c, indices_c  = torch.max(outputLine[1,:,:]>0.5,0)
        for i in range(outputLine.shape[2]):
            if indices_c[i]:
                Tx.append(i)
                Ty.append(indices[i].cpu().detach().numpy())
                Tmap[indices[i],i] = 255
                if checkI == 0:
                    checkI = 1
                    TRP[0] = indices[i].cpu().detach().numpy()  
                    TRP[1] = i
                else:
                    TLP[0] = indices[i].cpu().detach().numpy()  
                    TLP[1] = i

        tmpTC = np.round((TLP[1]+TRP[1])/2)
        TRPC = tmpTC - LS
        TLPC = tmpTC + LS

        TRPC41 = np.round((TRP[1] + TRPC) / 2)
        TLPC41 = np.round((TLP[1] + TLPC) / 2)

        plt.imsave('test1t.png',Fmap.astype('uint8'))

        Fx = np.array(Fx)
        Fy = np.array(Fy)
        Tx = np.array(Tx)
        Ty = np.array(Ty)
        locFt =  np.where(Fx <= FRPC)
        locTt =  np.where(Tx <= TRPC)

        Rmx = np.concatenate((Fx[locFt],Tx[np.fliplr(locTt)].squeeze()),axis=0)  
        Rmy = np.concatenate((Fy[locFt],Ty[np.fliplr(locTt)].squeeze()),axis=0)


        locFt = np.where(Fx >= FLPC)
        locTt = np.where(Tx >= TLPC)

        Lmx = np.concatenate((Fx[locFt],Tx[np.fliplr(locTt)].squeeze()),axis=0)  
        Lmy = np.concatenate((Fy[locFt],Ty[np.fliplr(locTt)].squeeze()),axis=0)

        checkMap = ((Fmap + Tmap)>120)*255
        plt.imsave('test1ft.png',checkMap.astype('uint8'))
        checkRap = np.zeros((outputLine.shape[1],outputLine.shape[2]), np.uint8)
        area1 = np.concatenate((Rmx[:,np.newaxis],Rmy[:,np.newaxis]), axis=1)
        area2 = np.concatenate((Lmx[:,np.newaxis],Lmy[:,np.newaxis]), axis=1)

        cv2.fillPoly(checkRap, [area1], 128)
        cv2.fillPoly(checkRap, [area2], 175)
        plt.imsave('test1tcheckRap.png',checkRap.astype('uint8'))


        outputIM = checkMap+checkRap


        
    torch.cuda.empty_cache() 

    # Femur
    FemurFRLocDP = np.zeros((2,1))
    FemurTRLocDP = np.zeros((2,1))
    
    loctmpFromFR = np.array(np.where(outputIM[:,int(FRPC41)]!=0)).squeeze() 
    
    FemurFRLocDP[0] = (FRPC41+5)/512*osi[1]
    FemurFRLocDP[1] = (loctmpFromFR[0]+5)/512*osi[0]
    
    FemurTRLocDP[0] = (FRPC41+5)/512*osi[1]
    FemurTRLocDP[1] = (loctmpFromFR[-1]+5)/512*osi[0]
    
    FemurRLocSeDmin = np.sqrt(np.sum(pow((FemurFRLocDP - FemurTRLocDP),2)))*ps
    #print(FemurRLocSeDmin)
    
    FemurFLLocDP = np.zeros((2,1))
    FemurTLLocDP = np.zeros((2,1))
    
    loctmpFromFL = np.array(np.where(outputIM[:,int(FLPC41)]!=0)).squeeze() 


    FemurFLLocDP[0] = (FLPC41+5)/512*osi[1]
    FemurFLLocDP[1] = (loctmpFromFL[0]+5)/512*osi[0]

    FemurTLLocDP[0] = (FLPC41+5)/512*osi[1]
    FemurTLLocDP[1] = (loctmpFromFL[-1]+5)/512*osi[0]

    FemurLLocSeDmin = np.sqrt(np.sum(pow((FemurFLLocDP - FemurTLLocDP),2)))*ps
    
    #print(FemurLLocSeDmin)

    
    ## Tibia
    TibiaFRLocDP = np.zeros((2,1))
    TibiaTRLocDP = np.zeros((2,1))
    
    loctmpFromTR = np.array(np.where(outputIM[:,int(TRPC41)]!=0)).squeeze() 
    
    
    TibiaFRLocDP[0] = (TRPC41+5)/512*osi[1]
    TibiaFRLocDP[1] = (loctmpFromTR[0]+5)/512*osi[0]
    
    TibiaTRLocDP[0] = (TRPC41+5)/512*osi[1]
    TibiaTRLocDP[1] = (loctmpFromTR[-1]+5)/512*osi[0]
    
    TibiaRLocSeDmin = np.sqrt(np.sum(pow((TibiaFRLocDP - TibiaTRLocDP),2)))*ps
    
    
    TibiaFLLocDP = np.zeros((2,1))
    TibiaTLLocDP = np.zeros((2,1))

    
    loctmpFromTL = np.array(np.where(outputIM[:,int(TLPC41)]!=0)).squeeze() 
    

    TibiaFLLocDP[0] = (TLPC41+5)/512*osi[1]
    TibiaFLLocDP[1] = (loctmpFromTL[0]+5)/512*osi[0]
    
    TibiaTLLocDP[0] = (TLPC41+5)/512*osi[1]
    TibiaTLLocDP[1] = (loctmpFromTL[-1]+5)/512*osi[0]
    
    TibiaLLocSeDmin = np.sqrt(np.sum(pow((TibiaFLLocDP - TibiaTLLocDP),2)))*ps

    #print(TibiaRLocSeDmin)
    #print(TibiaLLocSeDmin)
    
    print('knee line infer time:', time.time()-train_started, 'seconds')
    

    outputIM[5:,5:] = outputIM[0:-5,0:-5]
    tmpImMap = cv2.resize(outputIM,(osi[1],osi[0]),interpolation=cv2.INTER_NEAREST).astype('uint8')                      
    
    
    RPL = round(np.array(np.where(outputIM ==128)).shape[1]*ps*ps/100,4)
    LPL = round(np.array(np.where(outputIM ==175)).shape[1]*ps*ps/100,4)              


    FemurRLocSeDmin = round(FemurRLocSeDmin,2)
    FemurLLocSeDmin = round(FemurLLocSeDmin,2)
    TibiaRLocSeDmin = round(TibiaRLocSeDmin,2)
    TibiaLLocSeDmin = round(TibiaLLocSeDmin,2)    

    
    return tmpImMap,FemurFRLocDP,FemurTRLocDP,FemurFLLocDP,FemurTLLocDP,TibiaFRLocDP,TibiaTRLocDP,TibiaFLLocDP,TibiaTLLocDP,RPL,LPL,FemurRLocSeDmin,FemurLLocSeDmin,TibiaRLocSeDmin,TibiaLLocSeDmin

def save_dicom_process(img_file_name,ToOutput,result_str,dcm_ds,osi,Result_path):
    #ds.some_function_that_modifies_arr

    dsn = Dataset()
    dsn.file_meta = copy.deepcopy(dcm_ds.file_meta) 
    dsn.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
    dsn.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.1'
    dsn.file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    dsn.file_meta.ImplementationClassUID = "1.2.3.4"

    for tagIndex in dcm_ds.keys():
        if tagIndex == 'PixelData':
            dsn.PixelData = ToOutput.tobytes()
        elif tagIndex == 'WindowCenter' or tagIndex == 'WindowWidth':
            continue
        else:
            dsn[tagIndex] = dcm_ds[tagIndex]

    dsn.PatientName = 'Created'

    dsn.Rows = osi[0]
    dsn.Columns = osi[1]
    dsn.PhotometricInterpretation = "RGB"
    dsn.SamplesPerPixel = 3
    dsn.BitsStored = 8
    dsn.BitsAllocated = 8
    dsn.HighBit = 7
    dsn.PixelRepresentation = 0
    dsn.PlanarConfiguration = 0 
    dsn.NumberOfFrames = 1    
    dsn.SOPClassUID = dcm_ds.SOPClassUID
    dsn.SOPInstanceUID = dcm_ds.SOPInstanceUID
    dsn.StudyInstanceUID = dcm_ds.StudyInstanceUID
    dsn.SeriesInstanceUID = dcm_ds.SeriesInstanceUID     
    dsn.ImageComments = result_str
  
    #dsn.add_new(0x002040000,'LT', str(FemurRLocSeDmin))
    dsn.is_little_endian = True
    dsn.is_implicit_VR = True
    print(os.path.join(Result_path,img_file_name))
    dsn.save_as(os.path.join(Result_path,img_file_name), write_like_original=False)

def infer_process(sess,modelkeypoint,modelLine,infer_path,Result_path,Final_path,buffer_size):
    # init     
    mmCw = 65
    mmCh = 80    
    #load Q data
    #   
    QDataFile = glob(infer_path + '/*')
    #load dicom data
    for img_path in QDataFile:    
        # Data loading code
        
        try:
            print("=> loading dicom image " )
            dicom_image = pyd.dcmread(img_path)
            print(img_path)
            # 16 bit to bit
            ds = pyd.read_file(img_path, force=True)
            ds.decompress()
            ansds = ds.PhotometricInterpretation

            if ansds == 'MONOCHROME1':
                image = ((2**ds.get('BitsStored'))-1) - dicom_image.pixel_array
            else:
                image = dicom_image.pixel_array

            image = image.astype('float32') / ((2**ds.get('BitsStored'))-1) 
            image = np.repeat(image[:,:,np.newaxis],3,axis=2)

            pred_img = copy.deepcopy(image)
            pred_img = (pred_img*255).astype('uint8')   
            
            # 原圖處理
            if 'PixelSpacing' in ds:
                ps = ds.get('PixelSpacing')[0]
            elif 'ImagerPixelSpacing' in ds:   
                ps = ds.get('ImagerPixelSpacing')[0]
            else:
                ps = 0.15

            LcMmw = mmCw / ps
            LcMmh = mmCh / ps 

            osi = image.shape
            
            image = cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR) 
            image = image[:,:,:,np.newaxis]            
            
            image = image.transpose((3, 2, 0, 1))        
            image = np.asarray(image)       
            

            # finding knee location and croping knee image
            print("=> finding knee location and croping knee image" )
            try:
                out_preds, crop_img = knee_location_detection(image,pred_img,modelkeypoint,osi,LcMmw,LcMmh)
            except:
                result_str = 'knee location model cannot infer'
                save_dicom_process(filename,pred_img,result_str,ds,osi,Result_path)
                continue

            print("=> Knee KL grading infer..." )    
            try:
                probs,OA_classification_run_index = OA_classification(crop_img,sess)
            except:
                result_str = 'Knee KL grading model cannot infer'
                save_dicom_process(filename,pred_img,result_str,ds,osi,Result_path)
                continue
            # create result  
            print("=> create Knee KL grading result" )

            plt.figure()   
            plt.subplot(1,2,1)
            plt.imshow(crop_img, cmap='gray')
            plt.title('KL-grading: ' + str(np.argmax(probs)))
            plt.axis('off')

            plt.subplot(1,2,2) 
            
            for kl in range(5):
                print(np.round(probs[kl], 2))
                plt.text(kl - 0.3, 0.35, "%.2f" % np.round(probs[kl], 2), fontsize=6)
                plt.bar(np.array([0, 1, 2, 3, 4]), probs, color='red', align='center',
                        tick_label=['KL0', 'KL1', 'KL2', 'KL3', 'KL4'], alpha=0.3)
                plt.ylim(0, 1)
                plt.yticks([])
            
            plt.savefig('./Result/Knee_classifcation_result.png', dpi=300)
            plt.close()
            try:
                tmpImMap,FemurFRLocDP,FemurTRLocDP,FemurFLLocDP,FemurTLLocDP,TibiaFRLocDP,TibiaTRLocDP,TibiaFLLocDP,TibiaTLLocDP,RPL,LPL,FemurRLocSeDmin,FemurLLocSeDmin,TibiaRLocSeDmin,TibiaLLocSeDmin = OA_detection(image,modelLine,osi,ps) 
            except:
                result_str = 'Knee line model cannot infer'
                save_dicom_process(filename,pred_img,result_str,ds,osi,Result_path)
                continue
            ##
            for i in range(pred_img.shape[0]):
                for j in range(pred_img.shape[1]):
                    if tmpImMap[i,j] == 174 or tmpImMap[i,j] == 127 or tmpImMap[i,j] == 255:
                        pred_img[i,j,:] = [255,255,255]
                    elif tmpImMap[i,j] == 128:
                        pred_img[i,j,:] = pred_img[i,j,:] *0.6 + [0,100,0]
                    elif tmpImMap[i,j] == 175:
                        pred_img[i,j,:] = pred_img[i,j,:] *0.6 + [100,0,0] 
                    
            ##

            # crop knee image    
            if (out_preds[0][1]-LcMmh) < 0:
                wsx1 = 0
            else:
                wsx1 = out_preds[0][1]-LcMmh

            if (out_preds[0][1]+LcMmh) > osi[0]:
                wsx2 =  osi[0]
            else:
                wsx2 = out_preds[0][1]+LcMmh 

            if (out_preds[0][0]-LcMmw) < 0:
                wsy1 = 0
            else:
                wsy1 = out_preds[0][0]-LcMmw

            if (out_preds[0][0]+LcMmw) > osi[0]:
                wsy2 =  osi[1]
            else:
                wsy2 = out_preds[0][0]+LcMmw
            
            crop_img_with_line = pred_img[int(wsx1):int(wsx2),int(wsy1):int(wsy2),:]
           
            train_started = time.time()
            
            plt.figure() 
            plt.subplot(1,3,1)
            plt.imshow(pred_img,cmap = ('gray')) 
            plt.title('JSW area')   
            plt.text(100,100,'Area:%.4f'%RPL,ha = 'center', color = "g",fontsize=12)
            plt.text(100,300,'Area:%.4f'%LPL,ha = 'center', color = "r",fontsize=12)     
            plt.axis('off')   
            ##line
            ToOutput = copy.deepcopy(pred_img)
            Femurpred_img = copy.deepcopy(pred_img)
            ToOutput = cv2.line(ToOutput, (int(FemurFRLocDP[0]), int(FemurFRLocDP[1])),(int(FemurTRLocDP[0]), int(FemurTRLocDP[1])), (255,0,0),1)
            ToOutput = cv2.line(ToOutput, (int(FemurFLLocDP[0]), int(FemurFLLocDP[1])), (int(FemurTLLocDP[0]), int(FemurTLLocDP[1])), (0,255,0),1) 
            ToOutput = cv2.line(ToOutput, (int(TibiaFRLocDP[0]), int(TibiaFRLocDP[1])),(int(TibiaTRLocDP[0]), int(TibiaTRLocDP[1])), (0,0,255),1)
            ToOutput = cv2.line(ToOutput, (int(TibiaFLLocDP[0]), int(TibiaFLLocDP[1])), (int(TibiaTLLocDP[0]), int(TibiaTLLocDP[1])), (255,255,0),1)

            Femur_line_img = cv2.line(Femurpred_img, (int(FemurFRLocDP[0]), int(FemurFRLocDP[1])),(int(FemurTRLocDP[0]), int(FemurTRLocDP[1])), (255,0,0),5)
            Femur_line_img = cv2.line(Femur_line_img, (int(FemurFLLocDP[0]), int(FemurFLLocDP[1])), (int(FemurTLLocDP[0]), int(FemurTLLocDP[1])), (0,255,0),5) 

            Tibiapred_img = copy.deepcopy(pred_img)
            Tibia_line_img = cv2.line(Tibiapred_img, (int(TibiaFRLocDP[0]), int(TibiaFRLocDP[1])),(int(TibiaTRLocDP[0]), int(TibiaTRLocDP[1])), (255,0,0),5)
            Tibia_line_img = cv2.line(Tibia_line_img, (int(TibiaFLLocDP[0]), int(TibiaFLLocDP[1])), (int(TibiaTLLocDP[0]), int(TibiaTLLocDP[1])), (0,255,0),5)
            
            Femur_line_img_crop = Femur_line_img[int(wsx1):int(wsx2),int(wsy1):int(wsy2),:]
        
            Tibia_line_img_crop = Tibia_line_img[int(wsx1):int(wsx2),int(wsy1):int(wsy2),:]
            plt.imsave('Femur_line_img.png',Femur_line_img.astype('uint8'))
            plt.imsave('Tibia_line_img.png',Tibia_line_img.astype('uint8'))
            plt.subplot(1,3,2)
            plt.imshow(Femur_line_img_crop,cmap = ('gray'))
            plt.title('Based on Femur')  
            plt.text(100,100,'Dist:%.2f'%FemurRLocSeDmin,ha = 'center', color = "g",fontsize=12)
            plt.text(100,300,'Dist:%.2f'%FemurLLocSeDmin,ha = 'center', color = "r",fontsize=12)          
            plt.axis('off')
            

            plt.subplot(1,3,3)
            plt.imshow(Tibia_line_img_crop,cmap = ('gray'))
            plt.title('Based on Tibia')  
            plt.text(100,100,'Dist:%.2f'%TibiaRLocSeDmin,ha = 'center', color = "g",fontsize=12)
            plt.text(100,300,'Dist:%.2f'%TibiaLLocSeDmin,ha = 'center', color = "r",fontsize=12)         
            plt.axis('off')

            plt.savefig('./Result/Knee_JSW_result.png', dpi=300)
            print('knee line draw time:', time.time()-train_started, 'seconds')
            plt.close()
            result_str = 'OA KL Grading: %s (%.2f) \n Right Area: %.4f \n Left Area: %.4f \n Femur based Right Dist (Red line): %.2f \n Femur based Left Dist (Green line): %.2f \n Tibia based Right Dist (Blue line): %.2f \n Tibia based Left Dist (Yellow line): %.2f ' %(str(np.argmax(probs)),np.round(probs[np.argmax(probs)], 2),RPL,LPL,FemurRLocSeDmin,FemurLLocSeDmin,TibiaRLocSeDmin,TibiaLLocSeDmin)
            dirname,filename=os.path.split(img_path)
            save_dicom_process(filename,ToOutput,result_str,ds,osi,Result_path)
        except:
            result_str = 'model cannot infer'
            save_dicom_process(filename,pred_img,result_str,ds,osi,Result_path)
    move_file_to_inferFolder(buffer_size,infer_path,Final_path)

def move_file_to_inferFolder(buffer_size,data_path,infer_path):
    img_paths = glob(data_path + '/*')
    count_index = 0
    for ftmp in img_paths:        
        dirname,filename=os.path.split(ftmp)
        shutil.move(os.path.join(data_path ,filename), os.path.join(infer_path,filename))
        #os.remove(os.path.join(data_path,filename))
        count_index += 1
        if count_index == buffer_size:
            break
        


def main():
    buffer_size = 4
    count_file_size = 0
    data_path = './Data'
    infer_path = './QData'
    Final_path = './EData'
    Result_path = './Result'

    argsKeypoint = parse_args_keypoint()

    argsLine = parse_args_line()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # loading line model
    print("=> loading line model " )
    modelLine = archs.NestedUNet(argsLine)
    modelLine = nn.DataParallel(modelLine)
    modelLine.to(device)
    modelLine.load_state_dict(torch.load('model_best_line.pth'))   

    # loading classification model
    print("=> loading classification model " )
    sess = rt.InferenceSession("DeepKnee_classification.onnx")

    print("=> loading keypoint model " )
    modelkeypoint = archs.NestedUNet(argsKeypoint)
    modelkeypoint = nn.DataParallel(modelkeypoint)
    modelkeypoint.to(device)
    modelkeypoint.load_state_dict(torch.load('model_best_keypoint.pth'))  

    while True:
        try:
            time.sleep(5)
            count_file_size = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
            print('File number:', count_file_size) 
            if count_file_size >= buffer_size:
                print('moving file....')
                move_file_to_inferFolder(buffer_size,data_path,infer_path)
                time.sleep(10)
                print('infer QData file....')
                infer_process(sess,modelkeypoint,modelLine,infer_path,Result_path,Final_path,buffer_size)
            else:
                print('wait file buffer is not enough....')
        except:  
            print('infer too quick....')  


    

    


if __name__ == '__main__':
    main()
