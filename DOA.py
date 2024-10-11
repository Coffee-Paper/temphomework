import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from beamforming import beamform_alg

# 生成功率-角度数据
def generate_y(angle_in,SNR,N_array,snaps):
    ########## 此处调节信噪比等参数
    instanceWF = beamform_alg(f0=1.5e9,theta_in=angle_in,snr=SNR,signal_type=1,N_array=N_array,L1=snaps)
    ########## signal_type=1使得前两个信号为相关信号，取0则全是非相干信号
    return  instanceWF.CBF(),instanceWF.MVDR(),instanceWF.MUSIC()

# 初始化图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.35)

# 初始角度偏移量
angle_offset = -30
angle_offset2 = 10
angle_offset3 = 60

# 初始信噪比,阵元数目，快拍数
SNR_init = 10
N_array_init = 16
snaps_init = 1024

theta = np.arange(-90, 90, 0.1)
P_CBF,P_MVDR,P_MUSIC = generate_y([angle_offset,angle_offset2,angle_offset3],SNR_init,N_array_init,snaps_init)
line_CBF, = ax.plot((theta), P_CBF, lw=1,label='CBF')
line_MVDR, = ax.plot((theta), P_MVDR, lw=1,label='MVDR')
line_MUSIC, = ax.plot((theta), P_MUSIC, lw=1,label='MUSIC')
ax.set_title('Power-Angle Pattern')
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Power (dB)')
ax.grid()
ax.set_xlim([-90, 90])
ax.set_ylim([-100, 10])
ax.legend()

# 添加滑块
ax_slider = plt.axes([0.1, 0.25, 0.8, 0.03])
slider = Slider(ax_slider, 'Angle1', -90, 90, valinit=angle_offset,valstep=0.1)

ax_slider2 = plt.axes([0.1, 0.20, 0.8, 0.03])
slider2 = Slider(ax_slider2, 'Angle2', -90, 90, valinit=angle_offset2,valstep=0.1)

ax_slider3 = plt.axes([0.1, 0.15, 0.8, 0.03])
slider3 = Slider(ax_slider3, 'Angle3', -90, 90, valinit=angle_offset3,valstep=0.1)

ax_slider4 = plt.axes([0.1, 0.10, 0.8, 0.03])
slider4 = Slider(ax_slider4, 'SNR', 1, 30, valinit=SNR_init,valstep=0.1)

ax_slider5 = plt.axes([0.1, 0.05, 0.8, 0.03])
slider5 = Slider(ax_slider5, 'N_array', 8, 32, valinit=N_array_init,valstep=1)

ax_slider6 = plt.axes([0.1, 0.0, 0.8, 0.03])
slider6 = Slider(ax_slider6, 'snaps', 100, 1000, valinit=snaps_init,valstep=1)

# 更新函数
def update(val):
    angle = (slider.val)
    angle2 = (slider2.val)
    angle3 = (slider3.val)
    SNR = (slider4.val)
    N_array = (slider5.val)
    snaps = (slider6.val)
    P_CBF,P_MVDR,P_MUSIC = generate_y([angle,angle2,angle3],SNR=SNR,N_array=N_array,snaps=snaps)    
    line_CBF.set_ydata(P_CBF)
    line_MVDR.set_ydata(P_MVDR)
    line_MUSIC.set_ydata(P_MUSIC)
    fig.canvas.draw_idle()

# 连接滑块事件
slider.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)
slider4.on_changed(update)
slider5.on_changed(update)
slider6.on_changed(update)
# 显示窗口
plt.show()
