import numpy as np

class beamform_alg:

    def __init__(
        self,
        f0=1.5e9,
        theta_in=[],
        N_array=32,
        L1=10000,
        snr=20,
        angle_step=0.1,
        sample_rate=1e6,
        signal_type=0,
        space_smooth=True,
        eq_array=8
    ):
        self.L1 = L1
        self.c = 3e8                                                       # 光速
        self.t = np.arange(L1) / sample_rate                               # 时间向量
        self.N_array = N_array                                             # 阵元数目
        self.N = np.arange(0, N_array, 1).reshape([-1, 1])                 # 阵元数目矩阵
        self.theta_t = np.array(theta_in)                                  # 信源入射角度
        self.num_t = len(self.theta_t)                                     # 信源数目
        self.lambda0 = self.c / f0                                         # 波长
        self.d = self.lambda0 * 0.5                                        # 阵元间隔
        self.theta_scan = np.arange(-90, 90, angle_step).reshape([1, -1])  # 角度
        self.sample_rate = sample_rate                                     # 信号采样率
        self.eq_array = eq_array                                           # 等效阵元数
        self.steer_vector = np.exp(                                        # 导向矢量
            -2j
            * np.pi
            * self.N
            * self.d
            / self.lambda0
            * np.sin(np.deg2rad(self.theta_t))
        )
        self.sin_signal = np.tile(
            np.exp(2j * np.pi * f0 * self.t), (self.num_t, 1)
        )                                                                           # 单频信号
        self.rand_signal = np.random.randn(self.num_t, L1)                          # 随机信号
        self.signal = self.rand_signal if signal_type == 1 else self.sin_signal
        self.signal[-1, :] = self.rand_signal[-1, :]
        self.noise = np.random.randn(self.N.shape[0], L1) + 1j * np.random.randn(
            self.N.shape[0], L1
        )
        self.snr_ratio = (1 / (10 ** (snr / 10))) ** 0.5
        self.steer_sig = self.steer_vector @ self.signal
        self.X = self.steer_sig + self.snr_ratio * self.noise                        # X为含噪声信号
        self.R0 = self.X @ self.X.T.conj() / self.X.shape[1]                         # 自相关矩阵
        self.R = self._space_smooth_alg() if space_smooth else self.R0
        # print('1')

    def _space_smooth_alg(self,):
        smooth_array_num = self.N_array - self.eq_array + 1
        R_sig = np.zeros([self.eq_array,self.eq_array])
        for i in range(smooth_array_num):
            sub = self.steer_sig[i:self.eq_array+i,:]
            R_sig = R_sig + (sub @ sub.T.conj() / sub.shape[1])
        R_sig = R_sig / smooth_array_num

        self.N_array = self.eq_array
        self.N = np.arange(0, self.N_array, 1).reshape([-1, 1])
        self.steer_vector = np.exp(                                        # 导向矢量
            -2j
            * np.pi
            * self.N
            * self.d
            / self.lambda0
            * np.sin(np.deg2rad(self.theta_t))
        )
        self.noise = np.random.randn(self.N.shape[0], self.L1) + 1j * np.random.randn(
            self.N.shape[0], self.L1
        )
        self.steer_sig = self.steer_vector @ self.signal
        self.X = self.steer_sig + self.snr_ratio * self.noise    # X为含噪声信号
        return R_sig

    def CBF(self):
        yy = []
        for th in self.theta_scan[0]:
            steervec = np.exp(
                -2j * np.pi / self.lambda0 * self.N * self.d * np.sin(np.deg2rad(th))
            )  # 每个角度上的方向矢量
            result = np.abs(steervec.T.conj() @ self.R @ steervec)
            yy.append(result)
        yy = yy / np.max(yy)
        y = 10 * np.log10(np.array(yy).reshape([-1, 1]))
        return y

    def MVDR(self):
        Rinv = np.linalg.inv(self.R)
        yy = []
        for th in self.theta_scan[0]:
            steervec = np.exp(
                -2j * np.pi * self.d / self.lambda0 * self.N * np.sin(np.deg2rad(th))
            )  # 每个角度上的方向矢量
            result = np.abs(1 / (steervec.T.conj() @ Rinv @ steervec))
            yy.append(result)
        yy = yy / np.max(yy)
        y = 10 * np.log10(np.array(yy).reshape([-1, 1]))
        return y

    def MUSIC(self, expected_sig_num=3):
        """Adjusting num_expected_signals between 1 and 7,
        underestimating the number will lead to missing signal(s)
        while overestimating will only slightly hurt performance."""

        w, v = np.linalg.eig(self.R)            # 特征值分解，v[:,i]是特征值w[i]对应的特征向量
        eig_order = np.argsort(np.abs(w))       # 升序排列，eig_order是对应的索引序号
        v = v[:, eig_order]                     # 重新排序v
        V = np.zeros((len(self.N), len(self.N) - expected_sig_num), dtype=np.complex64)
        for i in range(len(self.N) - expected_sig_num):
            V[:, i] = v[:, i]

        yy = []

        for th in self.theta_scan[0]:
            steervec = np.exp(
                -2j * np.pi * self.d / self.lambda0 * self.N * np.sin(np.deg2rad(th))
            )  # 每个角度上的方向矢量
            value = np.abs(
                (1 / (steervec.T.conj() @ V @ V.T.conj() @ steervec)).squeeze()
            )  # MUSIC
            yy.append(value)
        yy = yy / np.max(yy)
        y = 10 * np.log10(np.array(yy).reshape([-1, 1]))
        return y

    def Espirit(self,):
        """Espirit算法，旋转子空间不变算法，返回估计的角度，有相干源时效果很差

        Returns:
            ndarray: shape[num_of_signals,1], 返回的直接是估计的角度
        """
        X1 = self.X[:self.N_array-1,:]
        X2 = self.X[1:self.N_array,:]
        XX = np.vstack((X1,X2))
        RX = XX @ XX.T.conj() / XX.shape[1]
        w , v = np.linalg.eig(RX)
        Us = v[:,:self.num_t]
        Us1 = Us[:self.N_array-1,:]
        Us2 = Us[self.N_array-1:,:]
        PSI = np.linalg.inv(Us1.T.conj() @ Us1) @ Us1.T.conj() @ Us2    # 求LS下的PSI矩阵并求其特征值
        W , V =np.linalg.eig(PSI)
        ephi=np.arctan2(np.imag(W),np.real(W))
        P = np.rad2deg(-np.arcsin(ephi/np.pi))
        return P

    def DML(self,):
        """DML算法，曲线极小值处为估计角度

        Returns:
            y: ndarray[1800,1]
        """
        steervec = np.exp(-2j * np.pi / self.lambda0 * self.N * self.d * np.sin(np.deg2rad(self.theta_scan)))
        scan_len = steervec.shape[1]
        y = np.zeros([scan_len,1])
        for i in range(0,scan_len):
            scan = steervec[:,i].reshape([-1,1])
            Pa = scan / (scan.T.conj() @ scan) @ scan.T.conj()
            oPa = np.eye(Pa.shape[0]) - Pa
            y[i] = np.abs(np.trace( oPa @ self.R) / self.N_array)
        y = y / np.max(y)
        y = 10 * np.log10(y)

        # steervec = np.exp(-2j * np.pi / self.lambda0 * self.N * self.d * np.sin(np.deg2rad(self.theta_scan)))
        # yy = []
        # for index in range(steervec.shape[1]):
        #     vec = steervec[:,index].reshape([-1,1])
        #     Pi_A_theta = vec @ np.linalg.pinv(vec)
        #     oPi_A_theta = np.eye(Pi_A_theta.shape[0]) - Pi_A_theta
        #     yy.append(np.abs(np.trace(oPi_A_theta @ self.R) / self.N_array))
        # yy = yy / np.max(yy)
        # y = 10 * np.log10(np.array(yy).reshape([-1, 1]))
        return y

    def MSINR(self,theta_inter=[],mix=False):
        inter_sig = (
                    np.random.randn(len(theta_inter), self.L1) +
                    1j * np.random.randn(len(theta_inter), self.L1)
        )
        noise = (
                    np.random.randn(self.N.shape[0], self.L1) + 
                    1j * np.random.randn(self.N.shape[0], self.L1)
        )

        inter_steer_vector = np.exp(  # 干扰信号导向矢量
            -2j
            * np.pi
            * self.N
            * self.d
            / self.lambda0
            * np.sin(np.deg2rad(np.array(theta_inter)))
        )

        inter_steer_sig = inter_steer_vector @ inter_sig + 0.05 * noise

        inter_sig_R = (
            (inter_steer_sig @ inter_steer_sig.T.conj() / inter_steer_sig.shape[0])
            if mix == False
            else (
                inter_steer_sig @ inter_steer_sig.T.conj() / inter_steer_sig.shape[0]
                + self.R
            )
        )  # inter_sig_R为干扰信号的自相关矩阵

        W_opt = np.linalg.inv(inter_sig_R) @ self.steer_vector  # MSINR准则下最优权矢量
        # self.R为目标信号的自相关矩阵
        yy = []
        for th in self.theta_scan[0]:
            steervect = np.exp(
                -2j * np.pi / self.lambda0 * self.N * self.d * np.sin(np.deg2rad(th))
            )
            yy.append(abs(W_opt.T.conj() @ steervect))
        yy = np.array(yy)
        yy = yy / np.max(yy, 0)
        yy = np.sum(yy, 1)
        yy = yy / np.max(yy)
        y = 10 * np.log10(yy.reshape([-1, 1]))
        return y
