
"""

@author: sheng he 
@email: heshengxgd@gmail.com

This code is used for writer identification based on deep learning

"""

import os
import pickle
import numpy as np
import imageio
from PIL import Image
import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import random

class DatasetFromFolder(data.Dataset):
        def __init__(self,dataset,foldername,labelfolder,imgtype='jpg',scale_size=(64,128),
                     is_training=True):
                super(DatasetFromFolder,self).__init__()
                
                self.is_training = is_training
                
                self.imgtype = imgtype
                self.scale_size = scale_size
                self.folder = foldername
                self.dataset = dataset
                
                if self.dataset == 'CERUG-RU':
                    self.cerug = True
                else:
                    self.cerug = False
                
                self.labelidx_name = os.path.join(labelfolder, f"{dataset}_writer_index_table.pickle")
                print(self.labelidx_name)
                
                self.imglist = self._get_image_list(self.folder)
                
                self.idlist = self._get_all_identity()
                
                self.idx_tab = self._convert_identity2index(self.labelidx_name)
                
                self.num_writer = len(self.idx_tab)
                
                #------------ print info.
                print('-'*10)
                print('loading dataset %s with images: %d'%(dataset,len(self.imglist)))
                print('number of writer is: %d'%len(self.idx_tab))
                print('-*'*10)
                
                #self.trans = True
                
                
        
        # convert to idx for neural network
        def _convert_identity2index(self,savename):
                if os.path.exists(savename):
                        with open(savename,'rb') as fp:
                                identity_idx = pickle.load(fp)
                else:
                        #'''
                        identity_idx = {}
                        for idx,ids in enumerate(self.idlist):
                                identity_idx[ids] = idx
                        
                        with open(savename,'wb') as fp:
                                pickle.dump(identity_idx,fp)
                        #'''
                print(identity_idx)        
                return identity_idx
                                
        # get all writer identity
        def _get_all_identity(self):
                writer_list = []
                for img in self.imglist:
                        writerId = self._get_identity(img)
                        writer_list.append(writerId)
                writer_list=list(set(writer_list))
                return writer_list
        
        def _get_identity(self,fname):
                if self.cerug:
                        return fname.split('_')[2]
                else: return fname.split('-')[0]
        
        # get all image list 
        def _get_image_list(self, folder):
                imgs = []
                for fn in os.listdir(folder):
                        ext = fn.lower().rsplit('.',1)[-1]
                        if ext in ('png','jpg','jpeg'):
                                imgs.append(fn)
                return imgs
        
        def transform(self):
                return Compose([ToTensor(),])
        
        def resize(self,image):
                h,w = image.shape[:2]
                ratio_h = float(self.scale_size[0])/float(h)
                ratio_w = float(self.scale_size[1])/float(w)
                
                if ratio_h < ratio_w:
                        ratio = ratio_h
                        hfirst = False
                else:
                        ratio = ratio_w
                        hfirst = True
                        
                nh = int(ratio * h)
                nw = int(ratio * w)
                
                # создаём PIL-объект, масштабируем и возвращаем numpy-массив
                im_pil = Image.fromarray(image)
                imresized = im_pil.resize((nw, nh), resample=Image.BICUBIC)
                imre = np.array(imresized)

                
                imre = 255 - imre
                ch,cw = imre.shape[:2]
                if self.is_training:
                    new_img = np.zeros(self.scale_size)
                    dy = int((self.scale_size[0]-ch))
                    dx = int((self.scale_size[1]-cw))
                    dy = random.randint(0,dy)
                    dx = random.randint(0,dx)
                else:
                    new_img = np.zeros(self.scale_size)
                    dy = int((self.scale_size[0]-ch)/2.0)
                    dx = int((self.scale_size[1]-cw)/2.0)
                
                #new_img = np.zeros(self.scale_size)
                #dy = int((self.scale_size[0]-ch)/2.0)
                #dx = int((self.scale_size[1]-cw)/2.0)

                imre = imre.astype('float')
                
                new_img[dy:dy+ch,dx:dx+cw] = imre
                #new_img /= 256.0
                #print(new_img.shape)
                
                return new_img,hfirst

        
        def __getitem__(self,index):             
                imgfile = self.imglist[index]
                writer = self.idx_tab[self._get_identity(imgfile)]
                
                image = imageio.imread(os.path.join(self.folder, imgfile))
                if image.ndim == 3:
                       image = np.array(Image.fromarray(image).convert('L'))
                image,hfirst = self.resize(image)
                image = image / 255.0

                image = self.transform()(image)
                writer = torch.from_numpy(np.array(writer))
                
                return image,writer,imgfile
        
        def __len__(self):
                return len(self.imglist)
        
