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
import pandas as pd
import csv
from sklearn.metrics import accuracy_score

def get_labeled_data(X,y):
    """
    Function that returns the labeled data
    Parameters
    ---------- 
    X: array-like, shape (n_samples, n_features)
        The input samples.
    y: array-like, shape (n_samples,)
        The target values.
    Returns
    -------
    X: array-like, shape (n_samples, n_features)
        The input samples.
    y: array-like, shape (n_samples,)
        The target values.
    """
    df = pd.DataFrame(X)
    df['y'] = pd.Series(y)

    #Remove -1 values on y
    df = df[df['y'] != -1]

    X = df.drop(columns=['y']).values
    y = df['y'].values

    return X,y

def create_metrics(file_name, row):
    """
    Function that creates the metrics file
    Parameters
    ----------
    file_name: string
        Name of the file where the metrics will be saved
    row: list
        List with the name of the columns
    Returns
    -------
    None
    """
    with open("./Results/"+file_name, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(row)



def save_metrics(model, rep, kfold, lp, y_test, y_pred, file_name):
    """
    Function that calculates and saves results in the file
    Parameters
    ----------
    model: string
        Name of the model
    rep: int
        Repetition number
    kfold: int
        Kfold number
    lp: int
        Label proportion
    y_test: array-like, shape (n_samples,)
        The target values.
    y_pred: array-like, shape (n_samples,)
        The predicted values.
    file_name: string
        Name of the file where the metrics will be saved
    Returns
    -------
    None
    """
    acc = accuracy_score(y_test, y_pred)

    with open("./Results/"+file_name,"a") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([model, rep, fold, lp, acc])
        