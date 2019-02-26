import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    prec = np.load('logs/bilstm_crf_test_precisions.npy')
    rec = np.load('logs/bilstm_crf_test_recalls.npy')
    f1 = [2.0/(1/prec[i]+1/rec[i]) for i in range(len(prec))]
    print(prec)
    print(rec)
    print(f1)
    plt.figure(figsize=(6,4))
    plt.plot(prec,'b')
    plt.plot(rec,'r')
    plt.plot(f1,'k')
    plt.grid()
    plt.show()