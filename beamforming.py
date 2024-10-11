import numpy as np

class beamform_alg:
    
    def __init__(self,f0=1.5e9,theta_in=[],N_array=32,L1=10000,snr=20,angle_step=0.1,sample_rate=1e6,signal_type=0):
        self.c = 3e8                     #光速
        self.t = np.arange(L1)/sample_rate          #时间向量
        self.N = np.arange(0,N_array,1).reshape([-1,1])   #阵元数目矩阵
        self.theta_t = np.array(theta_in)       #信源入射角度
        self.num_t = len(self.theta_t)          #信源数目
        self.lambda0 = self.c/f0              #波长
        self.d = self.lambda0*0.5               #阵元间隔
        self.theta_scan = np.arange(-90, 90, angle_step).reshape([1,-1])   #角度
        self.sample_rate = sample_rate          #信号采样率
        self.steer_vector = np.exp(-2j*np.pi * self.N * self.d/self.lambda0 * np.sin(np.deg2rad(self.theta_t)))
        self.sin_signal = np.tile(np.exp(2j*np.pi*f0*self.t),(self.num_t,1))  #单频信号
        self.rand_signal = np.random.randn(self.num_t,L1)  #随机信号
        self.signal = self.sin_signal if signal_type == 1 else self.rand_signal 
        self.signal[-1,:] = self.rand_signal[-1,:]*1.414   #最后一个信号不同
        self.noise = np.random.randn(self.N.shape[0],L1) + 1j*np.random.randn(self.N.shape[0],L1)
        self.snr_ratio = (1/(10**(snr/10)))**0.5
        self.X = (self.steer_vector @ self.signal) + self.snr_ratio * self.noise
        self.R = self.X @ self.X.T.conj()/self.X.shape[1]

    def CBF(self):
        yy = []
        for th in self.theta_scan[0]:
            steervec = np.exp(-2j * np.pi / self.lambda0 * self.N * self.d * np.sin(np.deg2rad(th)))  # 每个角度上的方向矢量
            result = np.abs(steervec.T.conj() @ self.R @ steervec)
            yy.append(result)
        yy = yy/np.max(yy)
        y = 10*np.log10(np.array(yy).reshape([-1,1]))
        return y
    
    def MVDR(self):
        Rinv = np.linalg.inv(self.R)
        yy = []
        for th in self.theta_scan[0]:
            steervec = np.exp(-2j * np.pi * self.d/self.lambda0 * self.N * np.sin(np.deg2rad(th))) # 每个角度上的方向矢量
            result = np.abs(1/(steervec.T.conj() @ Rinv @ steervec))
            yy.append(result)
        yy = yy/np.max(yy)
        y = 10*np.log10(np.array(yy).reshape([-1,1]))        
        return y
    
    def MUSIC(self,expected_sig_num = 3):
        '''Adjusting num_expected_signals between 1 and 7, 
           underestimating the number will lead to missing signal(s)
           while overestimating will only slightly hurt performance.'''
        
        w, v = np.linalg.eig(self.R) # 特征值分解，v[:,i]是特征值w[i]对应的特征向量
        eig_order = np.argsort(np.abs(w)) # 升序排列，eig_order是对应的索引序号
        v = v[:,eig_order] # 重新排序v
        V = v[:,0:len(self.N)-expected_sig_num]
        
        yy = []  
        for th in self.theta_scan[0]:
            steervec = np.exp(-2j * np.pi * self.d/self.lambda0 * self.N * np.sin(np.deg2rad(th))) # 每个角度上的方向矢量
            value = np.abs((1/(steervec.T.conj() @ V @ V.T.conj() @ steervec)).squeeze()) # MUSIC
            yy.append(value)
        yy = yy/np.max(yy)
        y = 10*np.log10(np.array(yy).reshape([-1,1]))
        return y