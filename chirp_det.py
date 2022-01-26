#!/usr/bin/env python
#
# data format agnostic generic chirp detector
# juha vierinen 2020
#

# 我现在就觉得 这个算法就TM纯粹扯淡 我甚至都怀疑作者放出来的是假代码
# 自己写一个试试 分析用thor.py 接收到的数据 首先做小波 看是不是接收到了阶梯状的频率信号
# 然后在试试用我之前写的filter去识别 看能不能识别出来

import numpy as n
import argparse
import scipy.signal as ss
import matplotlib.pyplot as plt
import time
import glob
import re
import os
import scipy.fftpack
fftw=False
try:
    import pyfftw
    fftw=True
    print("using pyfftw")
except:
    print("couldn't load pyfftw, reverting to scipy. performance will suffer")
    fftw=False
    
import h5py
import scipy.constants as c
import datetime

def power(x):
    return(x.real**2.0 + x.imag**2.0)

def fft(x):
    if fftw:
        return(pyfftw.interfaces.numpy_fft.fft(x))#,planner_effort='FFTW_ESTIMATE'))
    else:
        return(scipy.fftpack.fft(x))

def ifft(x):
    if fftw:
        return(pyfftw.interfaces.numpy_fft.ifft(x))#,planner_effort='FFTW_ESTIMATE'))
    else:
        return(scipy.fftpack.ifft(x))

debug_out0=False
def debug0(msg):
    if debug_out0:
        print(msg)
debug_out1=True
def debug1(msg):
    if debug_out1:
        print(msg)

def unix2date(x):
    return datetime.datetime.utcfromtimestamp(float(x))  # ATC bug-fix for python3

def unix2datestr(x):
    return(unix2date(x).strftime('%Y-%m-%d %H:%M:%S'))

def unix2dirname(x):
    return(unix2date(x).strftime('%Y-%m-%d'))

class chirp_matched_filter_bank:
    def __init__(self,conf):
        self.conf=conf

        # create chirp signal vectors
        # centered around zero frequency
        self.chirps=[]
        # 一个单峰的分布
        self.wf=n.array(ss.hann(self.conf.n_samples_per_block),dtype=n.float32)
        for cr in self.conf.chirp_rates:
            print("creating filter with chirp-rate %1.2f kHz/s"%(cr/1e3))
            # 这里乘的是共轭 相当于把扫频带来的相位去掉 得到一个信号开始时频率的单频信号
            chirp_vec=n.array(self.wf*n.conj(self.chirpf(cr=cr)))
            self.chirps.append(chirp_vec)
        self.n_chirps=len(self.chirps)

    def chirpf(self,cr=160e3):
        """
        Generate a chirp. This is used for matched filtering
        """
        L=self.conf.n_samples_per_block
        sr=self.conf.sample_rate
        f0=0.0
        tv=n.arange(L,dtype=n.float64)/float(sr)
        # 从数据头开始的时间：dt
        # 我一直觉得这个有问题
        # phi = (w0 + alpha * dt) * (t0 + dt)
        # dphi = phi - w0*(t0+dt) = alpha*dt^2+alpha*dt*t0
        # 而这里是             dphi = alpha*dt^2/2
        # 这里不仅少一项，有一项还除了2 那么他最后识别出来的到底是什么？ 为什么还能识别出来？
        dphase=0.5*tv**2*cr*2*n.pi
        chirp=n.exp(1j*n.mod(dphase,2*n.pi))*n.exp(1j*2*n.pi*f0*tv)
        return(n.array(chirp,dtype=n.complex64))

    def seek(self,z,i0):
        """
        Look for chirps in data vector
        z data vector
        i0 time of the leading edge of the vector
        """
        cput0=time.time()
        n_samps=len(z)
        
        t0=i0/self.conf.sample_rate
        
        if n_samps != self.conf.n_samples_per_block:
            print("wrong number of samples given to matched filter")
            exit(0)
        
        # whiten noise with a regularized filter
        # 这不行吧 归一化了整个频谱都变了啊 这。。。
        Z=fft(self.wf*z)
        z=ifft(Z/(n.abs(Z)+1e-9))
        
        # matched filter output
        # store the best matching chirp-rate and
        # normalized SNR (we pre-whiten the signal)
        mf_p = n.zeros(n_samps,dtype=n.float32)
        mf_chirp_rate_idx = n.zeros(n_samps,dtype=n.int32)

        # filter output for all chirps, for storing ionograms
        mf = n.zeros([self.n_chirps,n_samps],dtype=n.float32)
        
        for cri in range(self.n_chirps):
            mf[cri,:]=power(n.fft.fftshift(fft(self.wf*self.chirps[cri]*z)))
            # combined max SNR for all chirps
            idx=n.where(mf[cri,:] > mf_p)[0]
            # find peak match function at each point
            mf_p[idx]=mf[cri,idx]
            # record chirp-rate that produces the highest matched filter output
#            mf_cr[idx]=self.conf.chirp_rates[cri]
            mf_chirp_rate_idx[idx]=cri

            # store snippet of the spectrum
            

        # detect peaks
        snrs=[]
        chirp_rates=[]
        frequencies=[]
        for i in range(self.conf.max_simultaneous_detections):
            mi=n.argmax(mf_p)
            # CLEAN detect peaks
            snr_max=mf_p[mi]
            # this is the center frequency of the dechirped signal
            # corresponds to the instantaneous
            # chirp frequency at the leading edge of the signal
            # 这不对吧 fvec可是加了个fs/2的 这样就把原来0-fs/2变成fs/2-fs去了
            # 这怎么行呢 fs/2以上的都是不靠谱的吧
            f0=self.conf.fvec[mi]
            # clear region around detection
            # 这里清不清空又怎么样 后面又没用这个变量了 扯淡
            mf_p[n.max([0,mi-self.conf.mfsi]):n.min([mi+self.conf.mfsi,n_samps-1])]=0.0
            # this is the chirp rate we've detected
            detected_chirp_rate=self.conf.chirp_rates[mf_chirp_rate_idx[mi]]

            # did we find a chirp?
            if snr_max > self.conf.threshold_snr:
                # the virtual start time
                chirp_time = t0 - f0/detected_chirp_rate
                debug1("found chirp snr %1.2f chirp-rate %1.2f f0 %1.2f chirp_time %1.4f %s"%(snr_max,detected_chirp_rate/1e3,f0/1e6,chirp_time,unix2datestr(chirp_time)))
                snrs.append(snr_max)
                chirp_rates.append(detected_chirp_rate)
                frequencies.append(f0)

                dname="%s/%s"%(self.conf.output_dir,unix2dirname(float(i0)/self.conf.sample_rate))
                
                if not os.path.exists(dname):
                    print("creating %s"%(dname))
                    os.mkdir(dname)
                    

                # tbd: make an hour directory
                ofname = "%s/chirp-%d.h5"%(dname,
                                           i0)
                ho=h5py.File(ofname,"w")
                ho["f0"]=f0
                ho["i0"]=i0
                ho["sample_rate"]=self.conf.sample_rate
                ho["n_samples"]=n_samps
                ho["chirp_time"]=chirp_time
                ho["chirp_rate"]=detected_chirp_rate
                ho["snr"]=snr_max
                debug1("saving %s"%(ofname))
                ho.close()
            
        cput1=time.time()

        data_dt=(n_samps/float(self.conf.sample_rate))

        return(snrs,chirp_rates,frequencies)
        
