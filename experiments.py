#!/usr/bin/env python
# Copyright (C) 2023  José Miguel Ramírez-Sanz <jmrsanz@ubu.es>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#imports
import numpy as np
import pandas as pd
import utils
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sslearn.wrapper import TriTraining, SelfTraining, CoTrainingByCommittee, Rasco, RelRasco, CoTraining, DeTriTraining, DemocraticCoLearning, CoForest, Setred
from sslearn.model_selection import artificial_ssl_dataset

#load data
df = pd.read_csv("/home/jmrsanz/Data/Salento/tsfresh_minimal.csv")
df = df.drop("id",axis=1)
df = df.drop("y_valuations",axis=1)
df = df.drop("y_users",axis=1)

#map data into stressful 1 - not stressful
df["y_experiences"] = df["y_experiences"].map({1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 1, 9: 0, 10: 1, 11: 0})

#drop others target not used in this experiment
X = df.drop("y_experiences",axis=1)
y = df["y_experiences"]

#random seed
random_state = np.random.RandomState(33)

#SSL models, except Co-Training
ssl_models = {"SelfTraining(kNN)":SelfTraining(base_estimator=KNeighborsClassifier(n_neighbors=3)),
              "SelfTraining(NB)":SelfTraining(base_estimator=GaussianNB()), 
              "SelfTraining(DT)":SelfTraining(base_estimator=DecisionTreeClassifier(random_state=random_state)), 
              "TriTraining":TriTraining(random_state=random_state), 
              "DemocraticCo":DemocraticCoLearning(random_state=random_state),
              "CoForest":CoForest(random_state=random_state)}

#SL models
sl_models = {"NB":GaussianNB(),
             "kNN":KNeighborsClassifier(n_neighbors=3),
             "DT":DecisionTreeClassifier(random_state=random_state)}

#Base estimators for Co-Training, apart because Co-Training view creation
base_estimator_co = {"NB":GaussianNB(),
                     "kNN":KNeighborsClassifier(n_neighbors=3),
                     "DT":DecisionTreeClassifier(random_state=random_state)}

#Generate labeled proportion from 0.1 to 0.9 by 0.1
label_prop = np.arange(0.1,0.91,0.1)

file_name = "experiment.csv"

#Create results file
row = ["model", "rep", "kfold", "lp", "acc"]
utils.create_metrics(file_name,row)

#5 repetitions experiment
for r in range(5):
    print("-> REP: ",r, flush=True)

    #Create the stratified kfold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    k=0
    
    #For each fold
    for train_index, test_index in skf.split(X, y):
        print("--> KFOLD nº: ", k, flush=True)

        #Separeta train/test data
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        #For each labeled proportion
        for lp in label_prop:
            print("---> Label percentage: ", lp, flush=True)

            #Remove label from data in order to create the SSL dataset
            X_train_ssl, y_train_ssl, X_unlabel, y_unlabel = artificial_ssl_dataset(X_train, y_train, lp, random_state=random_state)

            #For SL models delete unlabeled data from the dataset
            X_train_supervised, y_train_supervised = utils.get_labeled_data(X_train_ssl, y_train_ssl)

            #Training and predictions of SSL models except Co-Training
            for ssl_m in ssl_models:
                print("----> SSL model: ", ssl_m, flush=True)
                model = ssl_models[ssl_m]
                model.fit(X_train_ssl, y_train_ssl)
                y_pred = model.predict(X_test)
                utils.save_metrics(ssl_m,r,k,lp,y_test,y_pred, file_name)

            #Co-Training, generate views, then train and predict
            indices_columnas = list(range(X_train_ssl.shape[1]))
            columnas_X1 = random.sample(indices_columnas, X_train_ssl.shape[1]//2)
            columnas_X2 = [col for col in indices_columnas if col not in columnas_X1]
            for bs in base_estimator_co:
                print("----> SSL model: CoTraining + ", bs, flush=True)
                base_estimator = base_estimator_co[bs]
                model = CoTraining(base_estimator = base_estimator, random_state=random_state)
                model.fit(X_train_ssl, y_train_ssl, features = [columnas_X1, columnas_X2])
                y_pred = model.predict(X_test)
                utils.save_metrics("CoTraining("+bs+")",r,k,lp,y_test,y_pred, file_name)
            
            #SL models, train and predict
            for sl_m in sl_models:
                print("----> SL model: ", sl_m, flush=True)
                model = sl_models[sl_m]
                model.fit(X_train_supervised, y_train_supervised)
                y_pred = model.predict(X_test)
                utils.save_metrics(sl_m,r,k,lp,y_test,y_pred, file_name)
            
        print("\n-----------------------------------------------------------------\n\n", flush=True)
        k+=1
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n",flush=True)
print("\n\n\nFINISH", flush=True)
