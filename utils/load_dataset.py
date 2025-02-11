import pandas as pd
import os, sys
import numpy as np

def carica_dataset(percorso_dataset):

    Y = []
    for sim_id in range(10):  # Itera sulle cartelle SIM0, SIM1, ..., SIM9
        sim_folder = os.path.join(percorso_dataset, f"SIM{sim_id}")
        for file_id in range(10):  # Itera sui file data0.csv, data1.csv, ..., data9.csv
            file_path = os.path.join(sim_folder, f"data{file_id}.csv")
            df = pd.read_csv(file_path, header=None)
            Y.append(df.values[:, [2, 3]])  # Aggiungi la matrice traiettoria alla lista

    # Crea il vettore di tempo (comune a tutte le simulazioni)
    t = np.arange(0, len(Y[0]) * 0.02, 0.02)  # Assumiamo che tutte le simulazioni abbiano la stessa lunghezza

    return Y, t
    
#set the directory file of the dataset
percorso_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'H-B-D', 'Generate_Dataset_HBD', 'DATASET'))
Y, t = carica_dataset(percorso_dataset)
