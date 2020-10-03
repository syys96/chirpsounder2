import numpy as n
import os
try:
    import configparser
except ImportError as e:
    import configparser2 as configparser
    
import json

class chirp_config:
    def __init__(self,fname=None):
        c=configparser.ConfigParser()
        # initialize with default values
        c["config"]={"channel":'"cha"',
                     "sample_rate":"25000000.0",
                     "center_freq":"12.5e6",
                     "data_dir":'"/mnt/data/juha/hf25"',
                     "threshold_snr":"13.0",
                     "max_simultaneous_detections":"5",
                     "step":"1",
                     "n_samples_per_block":"5000000",
                     "minimum_frequency_spacing":"0.2e6",
                     "chirp_rates":"[50e3,100e3,125e3,500.0084e3]",
                     "output_dir":'"./chirp2"',
                     "range_resolution":"2e3",
                     "frequency_resolution":"50e3",
                     "maximum_analysis_frequency":"25e6",
                     "max_range_extent":"2000e3",
                     "plot_timings":"false",
                     "realtime":"false",
                     "n_downconversion_threads":"4"}

        if fname != None:
            if os.path.exists(fname):
                print("reading %s"%(fname))
                c.read(fname)
            else:
                print("configuration file %s doesn't exist. using default values"%(fname))
        self.fname=fname
        self.plot_timings=json.loads(c["config"]["plot_timings"])
        self.realtime=json.loads(c["config"]["realtime"])
        self.data_dir=json.loads(c["config"]["data_dir"])
        self.n_downconversion_threads=json.loads(c["config"]["n_downconversion_threads"])
        self.max_range_extent=json.loads(c["config"]["max_range_extent"])
        self.n_samples_per_block=json.loads(c["config"]["n_samples_per_block"])
        self.sample_rate=json.loads(c["config"]["sample_rate"])
        self.center_freq=json.loads(c["config"]["center_freq"])
        self.chirp_rates=json.loads(c["config"]["chirp_rates"])
        self.range_resolution=json.loads(c["config"]["range_resolution"])
        self.frequency_resolution=json.loads(c["config"]["frequency_resolution"])
        self.channel=json.loads(c["config"]["channel"])
        self.step=json.loads(c["config"]["step"])
        self.maximum_analysis_frequency=json.loads(c["config"]["maximum_analysis_frequency"])
        self.output_dir=json.loads(c["config"]["output_dir"])

        try:
            os.mkdir(self.output_dir)
        except:
            pass
        
        if not os.path.exists(self.output_dir):
            print("Output directory %s doesn't exists and cannot be created"%(self.output_dir))
            exit(0)
            
        # the minimum distance in frequency between detections
        # (avoid multiple detections of the same chirp)
        self.minimum_frequency_spacing=json.loads(c["config"]["minimum_frequency_spacing"])
        self.df=(float(self.sample_rate)/float(self.n_samples_per_block))
        self.mfsi=int(self.minimum_frequency_spacing/self.df) # minimum spacing of detections in fft bins
        
        # how many chirps can we detect simultaneously
        self.max_simultaneous_detections=json.loads(c["config"]["max_simultaneous_detections"])
        # the smallest normalized snr that is detected
        self.threshold_snr=json.loads(c["config"]["threshold_snr"])
        
        self.fvec=n.fft.fftshift(n.fft.fftfreq(self.n_samples_per_block,
                                               d=1.0/float(self.sample_rate)))+self.center_freq
    def __str__(self):
        out="Configuration\n"
        for e in dir(self):
            if not callable(getattr(self,e)) and not e.startswith("__"):
                out+="%s = %s\n"%(e,getattr(self,e))
        return(out)
    
if __name__ == "__main__":
    cc=chirp_config()
    print(cc)
