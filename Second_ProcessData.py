
#This program load data of a user and convert time-series data into frequency domain
#and then mean and standard deviation of delta, theta, alpha, beta and gamma is calculated and provide these features of each trial to svm grid-search cv to
#make a model with best C and gamma parameters then the .pkl model is stored for a user which will be used for real time data prediction


import csv
import numpy as np # suuport multidimensional array and high level mathematical functions
import os # assign its attributes to its specific path module
from sklearn.svm import SVC #
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd 
from sklearn.metrics import accuracy_score,classification_report
import time
from sklearn import model_selection,preprocessing,neighbors
from sklearn.pipeline import Pipeline, FeatureUnion
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
class ProcessData:
        
        def __init__(self,profilepath,profilename):
                self.profile_name = profilename
                self.profile_path = profilepath
                self.action_features = []
                self.rest_features = []
                self.ready_features = []
                self.ERS_features = []
                #self.Final = [][]
                self.labels = []
                self.label1 = []
                self.feature_names = ['Delta_Mean','Theta_Mean','Alpha_Mean','Beta_Mean','Gamma_Mean']######  for mean only
                self.column_names = ['Features','Label','Sensor1','Sensor2','Sensor3','Sensor4','Sensor5','Sensor6','Sensor7','Sensor8','Sensor9','Sensor10','Sensor11','Sensor12','Sensor13','Sensor14']

        def get_frequency(self,all_channel_data): 
                """
                Get frequency from computed fft for all channels. 
                Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
                Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
                """
                #Length data channel
                L = len(all_channel_data[0])
                
                #print L

                #Sampling frequency
                Fs = 128
                wL = Fs*2
                mean_all_channel_data = np.mean(all_channel_data, axis=1)
                #print np.shape(mean_all_channel_data)
                #print mean_all_channel_data[14]

                ######################## for DC Offset #########################
                raw_data = np.zeros((14,640),dtype=np.double)
                for m in range (0,14):
                        for n in range (0,640):
                                raw_data[m][n] = all_channel_data[m][n] - mean_all_channel_data[m]

                ################################################################
                
                freq_ampl = np.zeros((14,129),dtype=np.double)

                '''
                #pwelch = signal.welch(all_channel_data[0], fs=128, window='hanning', nperseg=384, noverlap=64, nfft=L)#for10%
                #plt.plot(pwelch[0],pwelch[1])
                #plt.show
                '''
                ############## Applying welch function for fft, window, and overlapping ################
                for i in range(0,14):
                        
                        pwelch = signal.welch(raw_data[i], fs=128, window='hanning', nperseg=wL, noverlap=wL/2, nfft=wL)
                        freq_ampl[i] = pwelch[1]
                ########################################################################################
                '''
                print pwelch[0]
                print np.shape(freq_ampl)

                Get fft data
                data_fft = self.do_fft(all_channel_data)
                print np.shape(data_fft)

                #Compute frequency
                frequency = map(lambda x: abs(x/L),data_fft)
                frequency = map(lambda x: x[: L/2+1]*2,frequency)
                print np.shape(frequency)
                '''
                #print pwelch[0]
                #List frequency
                delta = map(lambda x: x[wL*1/Fs-1: wL*4/Fs],freq_ampl)   #pick sample1--sample8(means 1Hz to 4Hz with freq resolution of 0.5)
                theta = map(lambda x: x[wL*4/Fs-1: wL*8/Fs],freq_ampl)   #pick sample7--sample16(means 3.5Hz to 8Hz with freq resolution of 0.5Hz)
                alpha = map(lambda x: x[wL*8/Fs-1: wL*13/Fs],freq_ampl)  #pick sample15--sample26(means 7.5Hz to 13Hz with freq resolution of 0.5)
                beta = map(lambda x: x[wL*13/Fs-1: wL*30/Fs],freq_ampl)  #pick sample25--sample60(means 12.5Hz to 30Hz with freq resolution of 0.5)
                gamma = map(lambda x: x[wL*30/Fs-1: wL*50/Fs],freq_ampl) #pics sample59--sample100(means 29.5Hz to 50Hz with freq resolution of 0.5)

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
                
                return feature,avg_freq_ampl

        
        
        def grid_searchcv(self,feature_with_labels,y):
                #print(y)
                ###For Equal Weiht


                #df = pd.DataFrame(feature_with_labels)
                #df = df.sort_values([70], ascending=[True])

                X = feature_with_labels
                y = y

                #scaler = preprocessing.StandardScaler()
                #X = scaler.fit_transform(X)
                #print(X)

                print ("Applying Neural Networks")
                print ("------------------------")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test =scaler.transform(X_test)
                mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),max_iter=800)#optimization of stochastic optimizer converged at 356 iterations
                mlp.fit(X_train,y_train)
                predictions = mlp.predict(X_test)
                c_matrix = confusion_matrix(y_test,predictions)
                print "Confusion Matrix: " 
                print c_matrix
                Sum = c_matrix[0][0] + c_matrix[0][1] + c_matrix[1][0] + c_matrix[1][1]
                result = (float(c_matrix[0][0] + c_matrix[1][1])/Sum)*100
                print ("Neural networks result ==> " + str(result) + " %")

                joblib.dump(mlp, os.path.join(self.profile_path,self.profile_name+'_model.pkl'))
                
                '''
                
                X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
                
                #print X_train
                min_c = -5
                max_c = 15
                C_range = [2**i for i in range(min_c,max_c+1)]

                min_gamma = -10
                max_gamma = 5
                gamma_range = [2**i for i in range(min_gamma,max_gamma+1)]
                
                print("# Tuning hyper-parameters")

                param_grid = {'C' : C_range, 'gamma' : gamma_range} #kernel='rbf'
                cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
                clf = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)  #####see page 1195 for SVM rbf gridsearch cv from http://scikit-learn.org/dev/_downloads/scikit-learn-docs.pdf
                clf.fit(X_train, y_train)

                print('Best score for X:', clf.best_score_)
                print('Best C:',clf.best_estimator_.C) 
                #print('Best Kernel:',clf.best_estimator_.kernel)
                print('Best Gamma:',clf.best_estimator_.gamma)
                
                print('svm result ==> ',clf.score(X_test, y_test))
                
                
                print('#############################################')
                k_range = range(1,31)
                k_scores = []
                for k in k_range:
                        knn = neighbors.KNeighborsClassifier(n_neighbors = k, weights='uniform')
                        scores = model_selection.cross_val_score(knn, X_train, y_train, cv = 10, scoring='accuracy')
                        k_scores.append(scores.mean())
                #print(k_scores)

                #plt.plot(k_range, k_scores)
                #plt.xlabel('value of k')
                #plt.ylabel('cross validated accuracy')
                #plt.show()
                trained_model = knn.fit(X_train,y_train)
                print('knn result ==> ',trained_model.score(X_test,y_test))
               
                
                joblib.dump(clf, os.path.join(self.profile_path,self.profile_name+'_model.pkl'))
                '''
                
                
        def main_process(self):
                self.original_df = pd.read_csv(os.path.join(self.profile_path,self.profile_name+'.csv'))

                Trials = len(self.original_df['Trial_No'])/1664
                
                selected = 0
                rejected = 0
                
                for trial in range(1,Trials+1):
                        
                        loc = self.original_df['Trial_No']==trial
                        #print 'loc = ',loc
                        trial_df = self.original_df[loc]
                        
                        rest_df = trial_df[:640]  #first 640
                        self.action_df = trial_df[-640:] #last 640
                        self.ready_df = trial_df[640:1024] #middle 384
                        

                        rest_all_channels_data = np.zeros((14,640),dtype=np.double)
                        action_all_channels_data = np.zeros((14,640),dtype=np.double)
                        ready_all_channels_data = np.zeros((14,384),dtype=np.double)

                        for i in range(0,14):
                               
                            sensor_rest_data = np.array((rest_df['Sensor'+str(i+1)]),dtype=np.double)
                            rest_all_channels_data[i] = sensor_rest_data
                            
                            sensor_action_data = np.array((self.action_df['Sensor'+str(i+1)]),dtype=np.double)
                            action_all_channels_data[i] = sensor_action_data

                            sensor_ready_data = np.array((self.ready_df['Sensor'+str(i+1)]),dtype=np.double)
                            ready_all_channels_data[i] = sensor_ready_data
                        
                        rest_features = np.zeros((1,70),dtype=np.double)###############  for mean only
                        action_features = np.zeros((1,70),dtype=np.double)##############
                        ready_features = np.zeros((1,70),dtype=np.double)##############

                        rest_features[0],rest_avg_frequency     = self.get_feature(rest_all_channels_data)
                        action_features[0],action_avg_frequency = self.get_feature(action_all_channels_data)
                        ready_features[0],action_avg_frequency = self.get_feature(action_all_channels_data)

                        #print action_features[0][0]
                        #ready_features[0],ready_avg_frequency     = self.get_feature(ready_all_channels_data)
                        
                        if action_avg_frequency < rest_avg_frequency:
                                print('Trial==>'+str(trial)+'  ##accepted' + 'with frequency '+str(action_avg_frequency) + 'and rest frequency is=>'+str(rest_avg_frequency))
                                selected = selected + 1
                                if (self.action_df['Label'].values)[0]=='A':
                                        
                                        
                                        for w in range(0,14):
                                                self.label1.append(1)
                                        self.labels.append(1)
                                        self.action_features.append(action_features[0])
                                        #print np.shape(self.action_features)
                                        #selected = selected +1
                                        
                                elif (self.action_df['Label'].values)[0]=='B':
                                        for u in range(0,14):
                                                self.label1.append(2)
                                        self.labels.append(2)
                                        self.action_features.append(action_features[0])
                                        #selected = selected + 1



                        
                                if (rest_df['Label'].values)[0]=='A':
                                        
                                        #self.labels.append(1)
                                        self.rest_features.append(rest_features[0])
                                        #selected = selected +1
                                        
                                elif (rest_df['Label'].values)[0]=='B':
                                        #self.labels.append(2)
                                        self.rest_features.append(rest_features[0])
                                        #selected = selected + 1
                                


                                if (self.ready_df['Label'].values)[0]=='A':
                                        
                                        #self.labels.append(1)
                                        self.ready_features.append(ready_features[0])
                                        #selected = selected +1
                                        
                                elif (self.ready_df['Label'].values)[0]=='B':
                                        #self.labels.append(2)
                                        self.ready_features.append(ready_features[0])
                                        #selected = selected + 1
                                
                                

                                        
                        else:
                                print('Trial==>'+str(trial)+'  ##rejected'+ 'with frequency '+str(action_avg_frequency) + 'and rest frequency is=>'+str(rest_avg_frequency))
                                rejected = rejected + 1
                                        
                print('selected = ',selected, ' and rejected =',rejected)
                
                #print self.rest_features[0][0]
                #print np.shape(self.rest_features)
                e = len(self.rest_features)
                ERS = np.zeros((e,70), dtype = np.double)
                
                for r in range(0,e):
                        for c in range (0,70):
                                
                                ERS[r][c] = ((self.action_features[r][c] - self.rest_features[r][c]) / self.rest_features[r][c])*100
                        self.ERS_features.append(ERS[r])
                print ("Size of ERD fearure: "+str(np.shape(self.ERS_features)))
                q = 0
                x=0
                y=14
                Final = np.zeros((e*14,5),dtype=np.double)
                for k in range(0,e):
                        
                        for l in range(x,y):
                                for m in range(0,5):
                                        Final[l,m] = self.ERS_features[k][q]
                                        q = q + 1
                                        if (q == 70):
                                                q = 0
                        x = x + 14
                        y = y + 14
                
                #print self.ERS_features[87][69]
                #Final = Final.reshape(e,14,5)
                print ("Size of Final features: "+str(np.shape(Final)))
                '''
                print (np.shape(self.labels))
                print self.labels                
                
                labels = np.transpose(self.labels)
                
                print (np.shape(labels))
                print labels
                #print (labels.reshape(34,1))
                '''



                                

                '''
                print self.labels[87]       
                print self.labels
                print len(self.labels)
                print np.shape(self.labels)
                print len(self.action_features)
                print np.shape(self.action_features)
                '''
                Store_action_Features = pd.DataFrame(columns = self.column_names)
                Store_rest_Features = pd.DataFrame(columns = self.column_names)
                Store_ready_Features = pd.DataFrame(columns = self.column_names)
                Store_ERS_Features = pd.DataFrame(columns = self.column_names)

                lab = 0
                for f in self.action_features:
                        features_action_dict = {
                            'Features' : self.feature_names,
                            'Label'    : [self.labels[lab] for i in range(0,5)],
                            'Sensor1'  : f[0:5],
                            'Sensor2'  : f[5:10],
                            'Sensor3'  : f[10:15],
                            'Sensor4'  : f[15:20],
                            'Sensor5'  : f[20:25],
                            'Sensor6'  : f[25:30],
                            'Sensor7'  : f[30:35],
                            'Sensor8'  : f[35:40],
                            'Sensor9'  : f[40:45],
                            'Sensor10' : f[45:50],
                            'Sensor11' : f[50:55],
                            'Sensor12' : f[55:60],
                            'Sensor13' : f[60:65],
                            'Sensor14' : f[65:70],  
                                        }
                        df_action_features = pd.DataFrame(features_action_dict,columns = self.column_names)
                        Store_action_Features = Store_action_Features.append(df_action_features)
                        lab = lab+1

                Store_action_Features.to_csv(os.path.join(self.profile_path,self.profile_name+'_action_Features'+'.csv'))


                lab = 0
                for f in self.rest_features:
                        features_rest_dict = {
                            'Features' : self.feature_names,
                            'Label'    : [self.labels[lab] for j in range(0,5)],
                            'Sensor1'  : f[0:5],
                            'Sensor2'  : f[5:10],
                            'Sensor3'  : f[10:15],
                            'Sensor4'  : f[15:20],
                            'Sensor5'  : f[20:25],
                            'Sensor6'  : f[25:30],
                            'Sensor7'  : f[30:35],
                            'Sensor8'  : f[35:40],
                            'Sensor9'  : f[40:45],
                            'Sensor10' : f[45:50],
                            'Sensor11' : f[50:55],
                            'Sensor12' : f[55:60],
                            'Sensor13' : f[60:65],
                            'Sensor14' : f[65:70],   
                                        }
                        df_rest_features = pd.DataFrame(features_rest_dict,columns = self.column_names)
                        Store_rest_Features = Store_rest_Features.append(df_rest_features)
                        lab = lab+1

                Store_rest_Features.to_csv(os.path.join(self.profile_path,self.profile_name+'_rest_Features'+'.csv'))
                



                lab = 0
                for f in self.ready_features:
                        features_ready_dict = {
                            'Features' : self.feature_names,
                            'Label'    : [self.labels[lab] for k in range(0,5)],
                            'Sensor1'  : f[0:5],
                            'Sensor2'  : f[5:10],
                            'Sensor3'  : f[10:15],
                            'Sensor4'  : f[15:20],
                            'Sensor5'  : f[20:25],
                            'Sensor6'  : f[25:30],
                            'Sensor7'  : f[30:35],
                            'Sensor8'  : f[35:40],
                            'Sensor9'  : f[40:45],
                            'Sensor10' : f[45:50],
                            'Sensor11' : f[50:55],
                            'Sensor12' : f[55:60],
                            'Sensor13' : f[60:65],
                            'Sensor14' : f[65:70],
                                        }
                        df_ready_features = pd.DataFrame(features_ready_dict,columns = self.column_names)
                        Store_ready_Features = Store_ready_Features.append(df_ready_features)
                        lab = lab+1

                Store_ready_Features.to_csv(os.path.join(self.profile_path,self.profile_name+'_ready_Features'+'.csv'))
                


                lab = 0
                for f in self.ERS_features:
                        features_ERS_dict = {
                            'Features' : self.feature_names,
                            'Label'    : [self.labels[lab] for i in range(0,5)],
                            'Sensor1'  : f[0:5],
                            'Sensor2'  : f[5:10],
                            'Sensor3'  : f[10:15],
                            'Sensor4'  : f[15:20],
                            'Sensor5'  : f[20:25],
                            'Sensor6'  : f[25:30],
                            'Sensor7'  : f[30:35],
                            'Sensor8'  : f[35:40],
                            'Sensor9'  : f[40:45],
                            'Sensor10' : f[45:50],
                            'Sensor11' : f[50:55],
                            'Sensor12' : f[55:60],
                            'Sensor13' : f[60:65],
                            'Sensor14' : f[65:70],   
                                        }
                        df_ERS_features = pd.DataFrame(features_ERS_dict,columns = self.column_names)
                        Store_ERS_Features = Store_ERS_Features.append(df_ERS_features)
                        lab = lab+1

                Store_ERS_Features.to_csv(os.path.join(self.profile_path,self.profile_name+'_ERS_Features'+'.csv'))
                
                
                
                self.grid_searchcv(np.array(self.ERS_features),np.array(self.labels))
                


#prd = ProcessData('F:\NCAI-Neurocomputation Lab\Project1 -\UsersData','Daniyal Azhar')
#prd.main_process()
