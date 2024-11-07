import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from beamforming import beamform_alg

# 生成功率-角度数据
def generate_y(angle_in,SNR,N_array,snaps,sig_type,space_smooth,eq_array):
    ########## 此处调节信噪比等参数
    instanceWF = beamform_alg(f0=1.5e9,
                              theta_in=angle_in,
                              snr=SNR,
                              signal_type=sig_type,
                              N_array=N_array,
                              L1=snaps,
                              space_smooth=space_smooth,
                              eq_array=eq_array)
    ########## signal_type=1使得前两个信号为相关信号，取0则全是非相干信号
    return (
        instanceWF.CBF(),
        instanceWF.MVDR(),
        instanceWF.MUSIC(),
        instanceWF.Espirit(),
        instanceWF.DML(),
        instanceWF.MSINR(),
    )

# 初始化图形和坐标轴
fig, ax = plt.subplots(figsize=(9, 7))
plt.subplots_adjust(bottom=0.4,right=0.7)

# 初始角度偏移量
angle_offset = -30
angle_offset2 = 10
angle_offset3 = 60

# 初始信噪比,阵元数目，快拍数
SNR_init = 10
N_array_init = 16
snaps_init = 1024
sig_type = 1
space_smooth = False
eq_array = 8

theta = np.arange(-90, 90, 0.1)
P_CBF, P_MVDR, P_MUSIC, P_Espirit, P_DML, P_MSINR = generate_y(
                                                    [angle_offset,angle_offset2,angle_offset3],
                                                    SNR_init,
                                                    N_array_init,
                                                    snaps_init,
                                                    sig_type,
                                                    space_smooth,
                                                    eq_array)
line_CBF, = ax.plot((theta), P_CBF, lw=1,label='CBF')
line_MVDR, = ax.plot((theta), P_MVDR, lw=1,label='MVDR')
line_MUSIC, = ax.plot((theta), P_MUSIC, lw=1,label='MUSIC')
Dot_Espirit = ax.scatter(P_Espirit, np.zeros(len(P_Espirit)), marker='x', label='Espirit')
line_DML, = ax.plot((theta), P_DML, lw=1,label='DML')
line_MSINR = ax.plot((theta), P_MSINR, lw=1,label='MSINR')
ax.set_title('Power-Angle Pattern')
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Power (dB)')
ax.grid()
ax.set_xlim([-90, 90])
ax.set_ylim([-70, 10])
ax.legend(bbox_to_anchor=(1.05, 1))

# 添加滑块
ax_slider = plt.axes([0.1, 0.30, 0.8, 0.03])
slider = Slider(ax_slider, 'Angle1', -90, 90, valinit=angle_offset,valstep=0.1)

ax_slider2 = plt.axes([0.1, 0.25, 0.8, 0.03])
slider2 = Slider(ax_slider2, 'Angle2', -90, 90, valinit=angle_offset2,valstep=0.1)

ax_slider3 = plt.axes([0.1, 0.20, 0.8, 0.03])
slider3 = Slider(ax_slider3, 'Angle3', -90, 90, valinit=angle_offset3,valstep=0.1)

ax_slider4 = plt.axes([0.1, 0.15, 0.8, 0.03])
slider4 = Slider(ax_slider4, 'SNR', 1, 30, valinit=SNR_init,valstep=0.1)

ax_slider5 = plt.axes([0.1, 0.10, 0.8, 0.03])
slider5 = Slider(ax_slider5, 'N_array', 8, 32, valinit=N_array_init,valstep=1)

ax_slider6 = plt.axes([0.1, 0.05, 0.8, 0.03])
slider6 = Slider(ax_slider6, 'snaps', 100, 1000, valinit=snaps_init,valstep=1)

ax_slider7 = plt.axes([0.1, 0.00, 0.8, 0.03])
slider7 = Slider(ax_slider7, 'eq_array', 1, 32, valinit=eq_array,valstep=1)

rax = plt.axes([0.75, 0.4, 0.15, 0.1])  # 开关放在画布右侧
check = CheckButtons(rax, ['Coherent'], [False])

rax2 = plt.axes([0.75, 0.5, 0.15, 0.1])  # 开关放在画布右侧
check2 = CheckButtons(rax2, ['Space Smooth'], [False])

# 更新函数
def update(val):
    angle = (slider.val)
    angle2 = (slider2.val)
    angle3 = (slider3.val)
    SNR = (slider4.val)
    N_array = (slider5.val)
    snaps = (slider6.val)
    eq_array = (slider7.val)
    P_CBF,P_MVDR,P_MUSIC,P_Espirit,P_DML,P_MSINR = generate_y([angle,angle2,angle3],
                                                      SNR=SNR,
                                                      N_array=N_array,
                                                      snaps=snaps,
                                                      sig_type=sig_type,
                                                      space_smooth=space_smooth,
                                                      eq_array=eq_array
                                                      )
    line_CBF.set_ydata(P_CBF)
    line_MVDR.set_ydata(P_MVDR)
    line_MUSIC.set_ydata(P_MUSIC)
    Dot_Espirit.set_offsets(np.c_[P_Espirit,np.zeros(len(P_Espirit))])
    line_DML.set_ydata(P_DML)
    line_MSINR[0].set_ydata(P_MSINR)
    fig.canvas.draw_idle()

def toggle_signal_type(val):
    global sig_type
    if sig_type == 0:
        sig_type = 1
    else:
        sig_type = 0  # 切换信号类型
    update(val)

def toggle_space_smooth(val):
    global space_smooth
    if space_smooth == False:
        ax.set_ylim([-200, 20])
        space_smooth = True
    else:
        ax.set_ylim([-70, 10])
        space_smooth = False  # 切换信号类型
    update(val)

# 连接滑块事件
slider.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)
slider4.on_changed(update)
slider5.on_changed(update)
slider6.on_changed(update)
slider7.on_changed(update)
check.on_clicked(toggle_signal_type)
check2.on_clicked(toggle_space_smooth)

# 显示窗口
plt.show()
