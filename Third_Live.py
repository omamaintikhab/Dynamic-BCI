
#Load username.pkl model to predict live eeg data output will be A or B then we
#can send that action to any application through socket

import os
import ctypes
import sys
from ctypes import *
from numpy import *
import time
from ctypes.util import find_library
print (ctypes.util.find_library('edk.dll'))  
print (os.path.exists('.\\edk.dll'))
libEDK = cdll.LoadLibrary("edk.dll")
import numpy as np
from sklearn.externals import joblib
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cbook

import serial




class LivePrediction:
    def __init__(self,self_kivy,profilepath,profilename):  
        
        self.profile_name = profilename
        self.profile_path = profilepath

        self.self_kivy = self_kivy
        self.action_type = 'no_action'
        self.is_reference_data_taken = False
        self.time = 0
        self.choice = ''
        self.reference_data = np.zeros((14,2560))########################################3#####################################################33
        self.reference_data_features = np.zeros((1,70))#############################
        self.live_data =np.zeros((14,640))############################

        self.mlp = joblib.load((self.profile_path+self.profile_name+'_model.pkl'))
        
        #self.clf = joblib.load((self.profile_path+self.profile_name+'_model.pkl'))
        print(self.profile_path+self.profile_name+'_model.pkl')
        self.ED_COUNTER = 0
        self.ED_INTERPOLATED=1
        self.ED_RAW_CQ=2
        self.ED_AF3=3
        self.ED_F7=4
        self.ED_F3=5
        self.ED_FC5=6
        self.ED_T7=7
        self.ED_P7=8
        self.ED_O1=9
        self.ED_O2=10
        self.ED_P8=11
        self.ED_T8=12
        self.ED_FC6=13
        self.ED_F4=14
        self.ED_F8=15
        self.ED_AF4=16
        self.ED_GYROX=17
        self.ED_GYROY=18
        self.ED_TIMESTAMP=19
        self.ED_ES_TIMESTAMP=20
        self.ED_FUNC_ID=21
        self.ED_FUNC_VALUE=22
        self.ED_MARKER=23
        self.ED_SYNC_SIGNAL=24

        self.targetChannelList = [self.ED_RAW_CQ,self.ED_AF3, self.ED_F7, self.ED_F3, self.ED_FC5, self.ED_T7,self.ED_P7, self.ED_O1, self.ED_O2, self.ED_P8, self.ED_T8,self.ED_FC6, self.ED_F4, self.ED_F8, self.ED_AF4, self.ED_GYROX, self.ED_GYROY, self.ED_TIMESTAMP, self.ED_FUNC_ID, self.ED_FUNC_VALUE, self.ED_MARKER, self.ED_SYNC_SIGNAL]
        self.eEvent      = libEDK.EE_EmoEngineEventCreate()
        self.eState      = libEDK.EE_EmoStateCreate()
        self.userID            = c_uint(0)
        self.nSamples   = c_uint(0)
        self.nSam       = c_uint(0)
        self.nSamplesTaken  = pointer(self.nSamples)
        self.data     = pointer(c_double(0))
        self.user     = pointer(self.userID)
        self.composerPort          = c_uint(1726)
        self.secs      = c_float(1)
        self.datarate    = c_uint(0)
        self.readytocollect    = False
        self.option      = c_int(0)
        self.state     = c_int(0)
        
        print (libEDK.EE_EngineConnect("Emotiv Systems-5"))
        if libEDK.EE_EngineConnect("Emotiv Systems-5") != 0:
            print ("Emotiv Engine start up failed.")

        print ("Start receiving EEG Data! Press any key to stop logging...\n")

        self.hData = libEDK.EE_DataCreate()
        libEDK.EE_DataSetBufferSizeInSec(self.secs)

        self.j=0

        print ("Buffer size in secs:")
        

    def get_frequency(self,all_channel_data): 
                """
                Get frequency from computed fft for all channels. 
                Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
                Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
                """
                #Length data channel
                L = len(all_channel_data[0])

                #Sampling frequency
                Fs = 64
                wL = Fs*2
                mean_all_channel_data = np.mean(all_channel_data, axis=1)

                ######################## for DC Offset #########################
                raw_data = np.zeros((14,640),dtype=np.double)
                for m in range (0,14):
                        for n in range (0,640):
                                raw_data[m][n] = all_channel_data[m][n] - mean_all_channel_data[m]

                ################################################################
                

                ############## Applying welch function for fft, window, and overlapping ################

                freq_ampl = np.zeros((14,65),dtype=np.double)

                for i in range(0,14):
                        
                        pwelch = signal.welch(raw_data[i], fs=Fs, window='hanning', nperseg=wL, noverlap=wL/2, nfft=wL)
                        freq_ampl[i] = pwelch[1]

                ########################################################################################

                '''
                #print np.shape(freq_ampl)

                #Get fft data
                #data_fft = self.do_fft(all_channel_data)
                #print np.shape(data_fft)

                #Compute frequency
                #frequency = map(lambda x: abs(x/L),data_fft)
                #frequency = map(lambda x: x[: L/2+1]*2,frequency)
                #print np.shape(frequency)
                '''
                
                #List frequency
                delta = map(lambda x: x[wL*1/Fs-1: wL*4/Fs],freq_ampl)    #pick sample1--sample8(means 1Hz to 4Hz with freq resolution of 0.5)  
                theta = map(lambda x: x[wL*4/Fs-1: wL*8/Fs],freq_ampl)    #pick sample7--sample16(means 3.5Hz to 8Hz with freq resolution of 0.5Hz)
                alpha = map(lambda x: x[wL*8/Fs-1: wL*13/Fs],freq_ampl)   #pick sample15--sample26(means 7.5Hz to 13Hz with freq resolution of 0.5)
                beta = map(lambda x: x[wL*13/Fs-1: wL*30/Fs],freq_ampl)   #pick sample25--sample60(means 12.5Hz to 30Hz with freq resolution of 0.5)
                gamma = map(lambda x: x[wL*30/Fs-1: wL*50/Fs],freq_ampl)  #pick sample59--sample100(means 29.5Hz to 50Hz with freq resolution of 0.5)

                avg_freq_ampl = np.mean(freq_ampl)
                
                return delta,theta,alpha,beta,gamma,avg_freq_ampl


    def get_feature(self,all_channel_data):
                #Get frequency data
                (delta,theta,alpha,beta,gamma,avg_freq_ampl) = self.get_frequency(all_channel_data)

                #Compute feature std
                delta_std = np.std(delta, axis=1)
                theta_std = np.std(theta, axis=1)
                alpha_std = np.std(alpha, axis=1)
                beta_std = np.std(beta, axis=1)
                gamma_std = np.std(gamma, axis=1)

                #Compute feature mean
                delta_m = np.mean(delta, axis=1)
                theta_m = np.mean(theta, axis=1)
                alpha_m = np.mean(alpha, axis=1)
                beta_m = np.mean(beta, axis=1)
                gamma_m = np.mean(gamma, axis=1)

                ############################################## for mean only #############################################################

                #Concate feature
                feature = np.array([delta_m,theta_m,alpha_m,beta_m,gamma_m])
                feature = feature.T
                feature = feature.ravel()

                ##########################################################################################################################
                
                return feature







    def get_reference_frequency(self,all_channel_data): 
                """
                Get frequency from computed fft for all channels. 
                Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
                Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
                """
                #Length data channel
                L = len(all_channel_data[0])

                #Sampling frequency
                Fs = 128
                wL = Fs*2
                ###################### for DC Offset #####################
                mean_all_channel_reference_data = np.mean(all_channel_data, axis=1)
                raw_reference_data = np.zeros((14,1920),dtype=np.double)
                for m in range (0,14):
                        for n in range (0,1920):
                                raw_reference_data[m][n] = all_channel_data[m][n] - mean_all_channel_reference_data[m]

                ##########################################################
                
                ref_freq_ampl = np.zeros((14,129),dtype=np.double)

                ############## Applying welch function for fft, window, and overlapping ################
                
                for j in range(0,14):
                        
                        ref_pwelch = signal.welch(raw_reference_data[j], fs=128, window='hanning', nperseg=wL, noverlap=wL/2, nfft=wL)
                        ref_freq_ampl[j] = ref_pwelch[1]

                ########################################################################################

                '''
                #print np.shape(freq_ampl)

                #Get fft data
                #data_fft = self.do_fft(all_channel_data)
                #print np.shape(data_fft)

                #Compute frequency
                #frequency = map(lambda x: abs(x/L),data_fft)
                #frequency = map(lambda x: x[: L/2+1]*2,frequency)
                #print np.shape(frequency)
                '''
                
                #List frequency
                ref_delta = map(lambda x: x[wL*1/Fs-1: wL*4/Fs],ref_freq_ampl)   #pick sample1--sample8(means 1Hz to 4Hz with freq resolution of 0.5)
                ref_theta = map(lambda x: x[wL*4/Fs-1: wL*8/Fs],ref_freq_ampl)   #pick sample7--sample16(means 3.5Hz to 8Hz with freq resolution of 0.5Hz)
                ref_alpha = map(lambda x: x[wL*8/Fs-1: wL*13/Fs],ref_freq_ampl)  #pick sample15--sample26(means 7.5Hz to 13Hz with freq resolution of 0.5)
                ref_beta = map(lambda x: x[wL*13/Fs-1: wL*30/Fs],ref_freq_ampl)  #pick sample25--sample60(means 12.5Hz to 30Hz with freq resolution of 0.5)
                ref_gamma = map(lambda x: x[wL*30/Fs-1: wL*50/Fs],ref_freq_ampl) #pick sample59--sample100(means 29.5Hz to 50Hz with freq resolution of 0.5)

                ref_avg_freq_ampl = np.mean(ref_freq_ampl)
                
                return ref_delta,ref_theta,ref_alpha,ref_beta,ref_gamma,ref_avg_freq_ampl


    def get_reference_feature(self,all_channel_data):
                #Get frequency data
                (ref_delta,ref_theta,ref_alpha,ref_beta,ref_gamma,ref_avg_freq_ampl) = self.get_reference_frequency(all_channel_data)

                #Compute feature std
                ref_delta_std = np.std(ref_delta, axis=1)
                ref_theta_std = np.std(ref_theta, axis=1)
                ref_alpha_std = np.std(ref_alpha, axis=1)
                ref_beta_std = np.std(ref_beta, axis=1)
                ref_gamma_std = np.std(ref_gamma, axis=1)

                #Compute feature mean
                ref_delta_m = np.mean(ref_delta, axis=1)
                ref_theta_m = np.mean(ref_theta, axis=1)
                ref_alpha_m = np.mean(ref_alpha, axis=1)
                ref_beta_m = np.mean(ref_beta, axis=1)
                ref_gamma_m = np.mean(ref_gamma, axis=1)

                ############################################## for mean only #############################################################

                #Concate feature
                ref_feature = np.array([ref_delta_m,ref_theta_m,ref_alpha_m,ref_beta_m,ref_gamma_m])#not using std
                ref_feature = ref_feature.T
                ref_feature = ref_feature.ravel()

                ##########################################################################################################################
                
                return ref_feature









    
    def data_acq(self):
        a=1
        
        tscore = 0
        fscore = 0
        score_sum = 0
        #ser = serial.Serial('COM1', 9600, timeout=0)
        
        
        while (1):
            state = libEDK.EE_EngineGetNextEvent(self.eEvent)
            if state == 0:
                eventType = libEDK.EE_EmoEngineEventGetType(self.eEvent)
                libEDK.EE_EmoEngineEventGetUserId(self.eEvent, self.user)
                if eventType == 16:
                    libEDK.EE_DataAcquisitionEnable(self.userID,True)
                    self.readytocollect = True
            
            if self.readytocollect==True:
                libEDK.EE_DataUpdateHandle(0, self.hData)
                libEDK.EE_DataGetNumberOfSample(self.hData,self.nSamplesTaken)
                if self.nSamplesTaken[0] == 128:
                    self.nSam=self.nSamplesTaken[0]
                    arr=(ctypes.c_double*self.nSamplesTaken[0])()
                    ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))                         
                    data = array('d')
                    y = np.zeros((128,14))
                    for sampleIdx in range(self.nSamplesTaken[0]):
                        x = np.zeros(14)
                        for i in range(1,15):
                            libEDK.EE_DataGet(self.hData,self.targetChannelList[i],byref(arr), self.nSam)
                            x[i-1] = arr[sampleIdx]
                            
                        y[sampleIdx] = x
                    y = np.transpose(y)
                    if self.is_reference_data_taken == False:
                                
                        t = self.time
                        self.self_kivy.lbl4.text = 'Please First provide Neutral Thought Data for 15 seconds'
                        print ('Reference data time = ', str(t))
                        if t !=-1:
                            self.reference_data[:,(t*128):((t*128)+127)] = y[:,0:127]
                        self.time = self.time + 1
                            #print self.time
                        #print np.shape(self.reference_data)

                           #####for end 

                        if self.time>=19:
                            self.is_reference_data_taken = True
                            self.reference_data_features[0,:] = self.get_reference_feature(self.reference_data)
                            self.time = 0
                            
######### if end
                    #a = 2

                    if self.is_reference_data_taken == True:
                        t = self.time
                        #print ("time ="+str(t))
                        '''
                        if t>=0 and t<=4:
                            self.self_kivy.lbl4.text = 'Take rest & feel free for 5 seconds'
                            print ("feel free time ="+str(t))
                            self.time +=1

                        elif t>4 and t<=6:
                            self.self_kivy.lbl4.text = ''
                            self.self_kivy.text = os.getcwd()+'\\cue.png'
                            print ("cue time ="+str(t))
                            self.time +=1

                        elif t>6 and t<=11:
                            self.live_data[:,((t-7)*128):(((t-7)*128)+127)] = y[:,0:127]
                            print ("time for real time data extraction ="+str(t))
                            if t==7:
                                self.choice = random.choice(['1','2'])
                                self.self_kivy.lbl4.text = 'Lets think about the showing object ' + self.choice
                            if self.choice == '1':
                                self.self_kivy.text = os.getcwd()+'\\A.png'
                            elif self.choice == '2':
                                self.self_kivy.text = os.getcwd()+'\\B.png'
                            
                            self.time = self.time + 1
                            
                        elif t>11: 
                            #print np.shape(self.live_data)
                            original_data_features = self.get_feature(self.live_data)
                            #print np.shape(original_data_features)
                            X=np.zeros((1,70))
                            
                            x = original_data_features
                            #print np.shape(x)
                            #print self.reference_data_features[0][1]
                            #print x[0]
                    
                            #X[0]= x[0]
                            #y = []
                            #print np.shape(X)
                            ERS = np.zeros((1,70), dtype = np.double)
                            for r in range(0,70):
                                
                                ERS[0][r] = ((self.reference_data_features[0][r] - original_data_features[r]) / original_data_features[r])*100
                                
                            X[0]= ERS[0]
                            q=0
                            real_time_data = np.zeros((14,5),dtype=np.double) 
                            for row in range(0,14):
                                for col in range (0,5):
                                    real_time_data[row,col] = ERS[0][q]
                                    q +=1
                                    if(q==70):
                                        q=0
                            
                            print ("Size of live feature data: " +str(np.shape(ERS)))
                            
                           
                            
                            #print (ERS[0])
                            #plt.plot(ERS[0])
                            #plt.show()
                            y = []
                       
                            #y = self.mlp.predict(real_time_data)
                            y = self.mlp.predict(ERS)
                            
                            print ("final prediction: "+str(self.mlp.predict(ERS)))
                            self.time = 0

                            if str(y[0])==self.choice:
                                #print('hy')
                                self.self_kivy.lbl4.text='True'
                                tscore += 1
                                #self.self_kivy.lbl5.text='Score: '+ str(score)
                                print("Device is on for 5 seconds")
                                #ser.write('1')
                                time.sleep(5)
                                #self.self_kivy.text = os.getcwd()+'\\A.png'
                                #ser.write('')
                                #self.self_kivy.text = os.getcwd()+'\\machine-learning.jpeg'
                                #print('score is ',self.clf.score(X,y))
                            else:
                                self.self_kivy.lbl4.text='False'
                                fscore += 1

                            score_sum = tscore + fscore
                            self.self_kivy.lbl5.text='Score: '+ str(tscore)+'/'+str(score_sum)
                        '''
                        #if t==1:
                        self.live_data[:,(t*128):((t*128)+127)] = y[:,0:127]
                        original_data_features = self.get_feature(self.live_data)
                        x = original_data_features
                        #print np.shape(x)
                        #print self.reference_data_features[0][1]
                        #print x[0]
                    
                        #X[0]= x[0]
                        #y = []
                        #print np.shape(X)
                        X=np.zeros((1,70))
                        ERS = np.zeros((1,70), dtype = np.double)
                        for r in range(0,70):
                                
                            ERS[0][r] = ((self.reference_data_features[0][r] - original_data_features[r]) / original_data_features[r])*100
                                
                        X[0]= ERS[0]
                        q=0
                        real_time_data = np.zeros((14,5),dtype=np.double) 
                        for row in range(0,14):
                            for col in range (0,5):
                                real_time_data[row,col] = ERS[0][q]
                                q +=1
                                if(q==70):
                                    q=0
                        y = []
                       
                        #y = self.mlp.predict(real_time_data)
                        y = self.mlp.predict(ERS)
                            
                        print ("final prediction: "+str(self.mlp.predict(ERS)))
                        f= open("my_file.txt","w")
                        if y[0]==1:
                            #print('hy')
                            self.self_kivy.lbl4.text=''
                            self.self_kivy.text = os.getcwd()+'\\A.png'
                            f.write('1')#for on
                            f.close()
                            #print('score is ',self.clf.score(X,y))
                            #ser.write('1')

                        elif y[0]==2:
                            self.self_kivy.lbl4.text=''
                            #print(y[0])
                            #print(y[0],'for B')
                            self.self_kivy.text = os.getcwd()+'\\B.png'
                            f.write('2')#for off
                            f.close()
                            #print('score is ',self.clf.score(X,y))
                            #ser.write('2')
                        
                            '''
                            if str(y[0])==self.choice:
                                #print('hy')
                                self.self_kivy.lbl4.text='True'
                                tscore += 1
                                #self.self_kivy.lbl5.text='Score: '+ str(score)
                                print("Device is on for 5 seconds")
                                #ser.write('1')
                                time.sleep(5)
                                #self.self_kivy.text = os.getcwd()+'\\A.png'
                                #ser.write('')
                                #self.self_kivy.text = os.getcwd()+'\\machine-learning.jpeg'
                                #print('score is ',self.clf.score(X,y))
                            else:
                                self.self_kivy.lbl4.text='False'
                                fscore += 1

                            score_sum = tscore + fscore
                            self.self_kivy.lbl5.text='Score: '+ str(tscore)+'/'+str(score_sum) 
                            self.time = 0
                             
                            elif stry[0]==2:
                                self.self_kivy.lbl4.text=''
                                #print(y[0])
                                #print(y[0],'for B')
                                self.self_kivy.text = os.getcwd()+'\\B.png'
                                #ser.write('0')
                                #print('score is ',self.clf.score(X,y))
                            






                            if y[0]==1:
                                #print('hy')
                                self.self_kivy.lbl4.text=''
                                self.self_kivy.text = os.getcwd()+'\\A.png'
                                #ser.write('1')
                                #print('score is ',self.clf.score(X,y))

                            elif y[0]==2:
                                self.self_kivy.lbl4.text=''
                                #print(y[0])
                                #print(y[0],'for B')
                                self.self_kivy.text = os.getcwd()+'\\B.png'
                                #ser.write('0')
                                #print('score is ',self.clf.score(X,y))

                            #ser.write(y[0])
                            '''
                        time.sleep(0.2)
                        #self.time = 1
                        self.self_kivy.text = os.getcwd()+'\\machine-learning.jpeg'
                        
                           
                else:
                    print('No')
                    self.x = 'no_action'
                
            
            
            time.sleep(1.1)
            #self.self_kivy.text = ''
            if self.j>=5:
                print(' ')
                self.disconnect_engine()
                print("Engine Disconnected")
                break
           


    def disconnect_engine(self):
        libEDK.EE_DataFree(self.hData)
        libEDK.EE_EngineDisconnect()
        libEDK.EE_EmoStateFree(self.eState)
        libEDK.EE_EmoEngineEventFree(self.eEvent)






