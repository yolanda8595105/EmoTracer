from cgitb import reset
import imp
from operator import imod
from turtle import color

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

original_path = "D:/My_Document/EmoTracer/Wearable/P30/Observer/Integration/Grove.txt"
original_data = np.loadtxt(original_path) #转换为npy
data = original_data[:,6]

#均值滤波
def mean_filter(kernel_size, data):
    if kernel_size%2==0 or kernel_size<=1:
        print('kernel_size滤波核的需为大于1的奇数')
        return
    else:
        padding_data = []
        mid = kernel_size//2
        for i in range(mid):
            padding_data.append(0)
        padding_data.extend(data.tolist())
        for i in range(mid):
            padding_data.append(0)
    result = []
    for i in range(0, len(padding_data)-2*mid, 1):
        temp = 0
        for j in range(kernel_size):
            temp += padding_data[i+j]
        temp = temp / kernel_size
        result.append(temp)
    return result

# Before fft the data should remove the average data
def removemean(data):
    meandata = np.mean(data)
    print(meandata)
    result = data - meandata
    return result

def fft_filter(signal,fs,fc=[],type = 'bandpass'):
    '''
    this function is a filter using fft method.

    Band-pass or band-stop filtering is performed 
    according to the type instructions

        Parameters
        ----------
            signal: Signal
            fs: Sampling frequency
            fc: [fc1,fc2...] Cut-off frequency 
            type: bandpass | bandstop
        Returns
        -------
            result
    '''
    k = []
    N=len(signal)#get N

    for i in range(len(fc)):
        k.append(int(fc[i]*N/fs))
    #FFT
    signal_fft=scipy.fftpack.fft(signal)
    #Frequency truncation

    if type == 'bandpass':
        a = np.zeros(N)
        for i in range(int(len(fc)/2)):
            a[k[2*i]:k[2*i+1]] = 1
            a[N-k[2*i+1]:N-k[2*i]] = 1
    elif type == 'bandstop':
        a = np.ones(N)
        for i in range(int(len(fc)/2)):
            a[k[2*i]:k[2*i+1]] = 0
            a[N-k[2*i+1]:N-k[2*i]] = 0
    signal_fft = a*signal_fft
    signal_ifft=scipy.fftpack.ifft(signal_fft)
    result = signal_ifft.real
    return result

def main():
    

    afterremovemean = removemean(data)
    # gsr_fft = fft(gsr_afterremovemean)

    afterfft = fft_filter(afterremovemean, 200, [49, 51, 149, 151, 249, 251],type='bandstop')



    plt.plot(data[10000:10100], color = 'g') 
    plt.show()
    plt.plot(afterfft[10000:10100],color ='b')


    plt.show()

main()