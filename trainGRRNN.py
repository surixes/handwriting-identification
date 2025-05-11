"""
    This code is for the following paper:
    
    Sheng He and Lambert Schomaker
    GR-RNN: Global-Context Residual Recurrent Neural Networks for Writer Identification
    Pattern Recognition
    
    @email: heshengxgd@gmail.com
    @author: Sheng He
    @Github: https://github.com/shengfly/writer-identification
    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import dataloader as dset
import GRRNN as net
import numpy as np
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

class LabelSomCE(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,x,target,smoothing=0.1):
		confidence = 1.0 - smoothing
		logprobs = F.log_softmax(x,dim=-1)
		nll_loss = - logprobs.gather(dim=-1,index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = confidence * nll_loss + smoothing * smooth_loss

		return loss.mean()
    
class DeepWriter_Train:
    def __init__(self,dataset='CERUG-RU',imgtype='png',mode='vertical'):
    
        self.dataset = dataset
        self.folder = dataset
        self.train_folder = os.path.join(self.folder, 'train')
        self.test_folder  = os.path.join(self.folder, 'test')
        
        for d in (self.train_folder, self.test_folder):
            os.makedirs(d, exist_ok=True)

        # скачиваем и распаковываем через Kaggle API
        if not os.listdir(self.train_folder) or not os.listdir(self.test_folder):
            api = KaggleApi()
            api.authenticate()

            # скачиваем и распаковываем весь датасет сразу
            print(f"[1/2] Скачать и распаковать датасет в '{self.folder}' …")
            api.dataset_download_files(
                'constantinwerner/cyrillic-handwriting-dataset',
                path=self.folder,
                unzip=True
            )
            print("[1/2] Скачивание и распаковка завершены.")

            # Проверяем, что в папках train/ и test/ что-то появилось
            train_count = len(os.listdir(self.train_folder))
            test_count  = len(os.listdir(self.test_folder))
            print(f"[2/2] В '{self.train_folder}' — {train_count} файлов.")
            print(f"[2/2] В '{self.test_folder}' — {test_count} файлов.")
        
        self.labelfolder = self.folder
        self.train_folder = self.folder+'/train/'
        self.test_folder = self.folder+'/test/'
        
        self.imgtype=imgtype
        self.mode = mode
        self.device = torch.device('cpu')
        self.scale_size=(64,128)
        
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        if self.dataset == 'CVL':
            self.imgtype = 'tif'
        
        self.model_dir = 'model'
        if not os.path.exists(self.model_dir):
            #raise ValueError('Model directory: %s does not existed'%self.model_dir)
            os.mkdir(self.model_dir)#raise ValueError('Model directory: %s does not existed'%self.model_dir)
        
        basedir = 'GRRNN_WriterIdentification_dataset_'+self.dataset+'_model_'+self.mode+'_aug_16'
        self.logfile= basedir + '.log'
        self.modelfile = basedir
        self.batch_size = 16
        
        train_set = dset.DatasetFromFolder(dataset=self.dataset,
        				labelfolder = self.labelfolder,
                        foldername=self.train_folder,
                        imgtype=self.imgtype,
                        scale_size=self.scale_size,
                        is_training = True)
        
        self.training_data_loader = DataLoader(dataset=train_set, num_workers=0, 
                           batch_size=self.batch_size, shuffle=True)
        
        test_set = dset.DatasetFromFolder(dataset=self.dataset,
        				labelfolder = self.labelfolder,
                        foldername=self.test_folder,imgtype=self.imgtype,
                        scale_size=self.scale_size,
                        is_training = False)
        
        self.testing_data_loader = DataLoader(dataset=test_set, num_workers=0, 
                           batch_size=self.batch_size, shuffle=False)
        
        num_class = train_set.num_writer
        self.model = net.GrnnNet(1,num_classes=train_set.num_writer,mode=self.mode).to(self.device)
        
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = LabelSomCE()
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.0001,weight_decay=1e-4) 
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.5)
                
    def train(self,epoch):
        self.model.train()
        losstotal = []
        
        for iteration,batch in enumerate(self.training_data_loader,1):
            inputs = batch[0].to(self.device).float()
            target = batch[1].type(torch.long).to(self.device)
        
        
            self.optimizer.zero_grad()
   
            logits = self.model(inputs)
         
            train_loss= self.criterion(logits,target)

            losstotal.append(train_loss.item())
            train_loss.backward()
            self.optimizer.step()
        
        
        with open(self.logfile,'a') as fp:
            fp.write('Training epoch %d avg loss is: %.6f\n'%(epoch,np.mean(losstotal)))
        print('Traing epoch:',epoch,'  avg loss is:',np.mean(losstotal))

   
    def test(self,epoch,during_train=True):
        self.model.eval()
        
        if not during_train:
            self.load_model(epoch)

        top1 = 0
        top5 = 0
        ntotal=0
        
        for iteration,batch in enumerate(self.testing_data_loader,1):
            inputs = batch[0].to(self.device).float()
            target = batch[1].to(self.device).long()
           
            logits = self.model(inputs)
            
            res = self.accuracy(logits,target,topk=(1,5))
            top1 += res[0]
            top5 += res[1]
            
            ntotal += inputs.size(0)
        

        top1 /= float(ntotal)
        top5 /= float(ntotal)
    
        print('Testing on epoch: %d has accuracy: top1: %.2f top5: %.2f'%(epoch,top1*100,top5*100))
        with open(self.logfile,'a') as fp:
            fp.write('Testing epoch %d accuracy is: top1: %.2f top5: %.2f\n'%(epoch,top1*100,top5*100))

    def check_exists(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        return os.path.exists(model_out_path)
    
    def checkpoint(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        torch.save(self.model.state_dict(),model_out_path)
    
    
    def load_model(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        self.model.load_state_dict(torch.load(model_out_path,map_location=self.device))
        print('Load model successful')
                
    def train_loops(self,start_epoch,num_epoch):
        #if self.check_exists(num_epoch): return
        if start_epoch > 0:
            self.load_model(start_epoch-1)
        
        for epoch in range(start_epoch,num_epoch):
             
            self.train(epoch)
            self.checkpoint(epoch)
            self.test(epoch)
            self.scheduler.step()
        
    def accuracy(self,output,target,topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            _,pred = output.topk(maxk,1,True,True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.data.cpu().numpy())
        
        return res
                        
                        
                
                
if __name__ == '__main__':
	
    modelist = ['vertical','horzontal']
    mode = modelist[0]
    
    mod = DeepWriter_Train(dataset='CERUG-RU',mode=mode)
    mod.train_loops(0,50)
