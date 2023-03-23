import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from pathlib import Path

calibration_df = pd.read_csv('')



df = pd.read_csv('data_5hprobe.csv')
column_headers = df.columns
print(column_headers[0])
data_array = df.to_numpy()

ps = data_array[:1075,12]
xs = data_array[:1075,15]
ys = data_array[:1075,16]

y_len = 0
x_len = 0
for i in range(len(xs)):
    if xs[i] != xs[i+1]:
        x_len += 1
        y_len = i+1
        break

for i in range(len(xs)-1):
    if xs[i] != xs[i+1]:
        x_len += 1

print(x_len, y_len)

n=0
data_matrix = np.empty((x_len, y_len, 3))
data_matrix[:,:,:] = np.nan
for i, row in enumerate(data_matrix):
    row_length = 0
    for j, cell in enumerate(row):
        cell[0] = xs[n]
        cell[1] = ys[n]
        cell[2] = ps[n]

        n += 1
        row_length+=1
        if n >= len(xs):
            break
        elif (xs[n] != xs[n-1]):
            print(f'DIFFERENCE: {(xs[n-1], xs[n])}')
            print(f'count: {n}, rowlength = {row_length}, {(i,j)}')
            assert row[0,0] == xs[n-1] and row_length<=35

            break

print(data_matrix[:,:25,0])

class cp_results:
    def __init__(self, cp_theta, cp_phi, cp_center, cp_ave):
        self.cp_theta = cp_theta
        self.cp_phi = cp_phi
        self.cp_center = cp_center
        self.cp_ave = cp_ave

    @staticmethod
    def get_ave(my_pressures: list):
        return sum(my_pressures) / len(my_pressures)

    @staticmethod
    def get_cp_theta(p_upper, p_lower, p_center, p_ave):
        value = (p_upper - p_lower) / (p_center - p_ave)
        return value

    @staticmethod
    def get_cp(pressure, p_stat, p_tot):
        value = (pressure-p_stat)/(p_tot-p_stat)

    @classmethod
    def from_pressures(cls, probe_pressures, p_static, p_tot):
        p_ave = cls.get_ave(probe_pressures)
        cp_theta = cls.get_cp_theta(probe_pressures[0], probe_pressures[2], probe_pressures[4], p_ave)
        cp_phi = cls.get_cp_theta(probe_pressures[1], probe_pressures[3], probe_pressures[4], p_ave)
        cp_center = cls.get_cp(probe_pressures[4], p_static, p_tot)
        cp_ave = cls.get_cp(p_ave, p_static, p_tot)

        my_results = cls(cp_theta, cp_phi, cp_center, cp_ave)
        return my_results

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(data_matrix[:,:,0], data_matrix[:,:,1], data_matrix[:,:,2], alpha = 1) #, lw=0.5, rstride=8, cstride=8, alpha=0.3)

ax.set(xlabel='X position', ylabel='Y position', zlabel='Pressure')
plt.savefig("plot1.png")
# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
#ax.contour(Xs, Ys, Zs, zdir='z', offset=-100, cmap='coolwarm')
#ax.contour(Xs, Ys, Zs, zdir='x', offset=-40, cmap='coolwarm')
#ax.contour(Xs, Ys, Zs, zdir='y', offset=40, cmap='coolwarm')
plt.show()

