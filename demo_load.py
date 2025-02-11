import os, time
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False


import tensorflow as tf
tf.get_logger().setLevel('ERROR')
sess = tf.compat.v1.InteractiveSession()
FLAG= True
import numpy as np
from npODEeSDE.npde_helper import load_model



if FLAG:

    npde = load_model('npODEeSDE/npde_state_sde.pkl',sess)
    #plot_model(npde,t,Y)

    x0 = [0,0]
    t = np.linspace(0,20,100)
    Nw = 3 # number of samples

    print(t)
    start_time = time.time()
    samples = npde.sample(x0,t,Nw)
    samples = sess.run(samples)
    end_time = time.time()  
    tempo_esecuzione = end_time - start_time
    print("Tempo di esecuzione: %f secondi" % tempo_esecuzione)

    plt.figure(figsize=(12,5))
    for i in range(Nw):
        plt.plot(t,samples[i,:,0],'-k',linewidth=0.25)
        plt.plot(t,samples[i,:,1],'-r',linewidth=0.25)
    plt.xlabel('time',fontsize=12)
    plt.ylabel('states',fontsize=12)
    plt.title('npSDE samples',fontsize=16)
    plt.savefig('samples.png', dpi=200)
    plt.show()
else:
    npde = load_model('npODEeSDE/npde_state.pkl',sess)
    x0 = [0,-1] # initial value
    t = np.linspace(0,20,100) # time points 
    path = npde.predict(x0,t)
    path = sess.run(path)

    plt.figure(figsize=(12,5))
    plt.plot(t,path)
    plt.xlabel('time',fontsize=12)
    plt.ylabel('states',fontsize=12)
    plt.title('npSDE mean future predictions',fontsize=16)
    plt.show()


