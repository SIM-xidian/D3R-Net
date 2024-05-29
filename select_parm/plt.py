import numpy as np
import matplotlib.pyplot as plt

# 创建随机数据
data = np.zeros((7, 22))
data[1][0]=1
# 绘制网格分布图
fig, ax = plt.subplots(figsize=(16,16))
im = ax.imshow(data, cmap='Blues', interpolation='nearest')
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))
labelx = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
# ax.set_xticklabels(np.arange(1, data.shape[1] + 0.5))
ax.set_xticklabels(labelx)
labely=[7,6,5,4,3,2,1]
ax.set_yticklabels(labely)

# 获取当前图像的尺寸（宽度，高度）
fig_width, fig_height = fig.get_size_inches()
# 获取轴的相对尺寸在图像中的比例
ax_width, ax_height = ax.get_position().width, ax.get_position().height
# 根据图像尺寸和轴尺寸计算y轴的实际高度（以英寸为单位）
y_axis_height_inches = fig_height * ax_height

# 计算颜色条的shrink参数，使其与y轴高度相同
# 注意：这里的计算是示意性的，实际可能需要根据具体情况微调
cbar_shrink = y_axis_height_inches / (fig_width * 0.5)  # 0.5 是一个示例值，代表默认全宽颜色条的一半

# 添加颜色条，使其与y轴同高
cbar = fig.colorbar(im, ax=ax, shrink=cbar_shrink)
ax.tick_params(bottom=False, top=False, left=False, right=False)

plt.show()


