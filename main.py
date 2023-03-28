import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from pathlib import Path
import calibration

#calibration_df = pd.read_csv('')

df = pd.read_csv('data_5hprobe.csv')
column_headers = df.columns
#print(column_headers[0])

data_array = df.to_numpy()

# separate data into regen and propulsive stage and into the 3 different conditions
data_prop_stage = np.array_split(data_array[:1050,:],3)
data_regen_stage = np.array_split(data_array[1050:,:],3)

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

p1 = data_prop_stage[0][:,10]
p2 = data_prop_stage[0][:,11]
p3 = data_prop_stage[0][:,12]
p4 = data_prop_stage[0][:,13]
p5 = data_prop_stage[0][:,14]

pavg = np.array([cp_results.get_ave(pressures) for pressures in np.array([p1,p2,p3,p4,p5]).T])

cp_alphas = cp_results.get_cp_theta(p1, p3, p5 , pavg)
cp_betas = cp_results.get_cp_theta(p2, p4, p5 , pavg)




alphas = [calibration.alpha_model([cp_alphas[i], cp_betas[i]])*180/np.pi for i in range(len(cp_alphas))]
betas = [calibration.beta_model([cp_alphas[i], cp_betas[i]])*180/np.pi for i in range(len(cp_alphas))]

print(alphas)

plt.scatter(alphas, betas)
plt.show()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(data_matrix[:,:,0], data_matrix[:,:,1], data_matrix[:,:,2], alpha = 1) #, lw=0.5, rstride=8, cstride=8, alpha=0.3)

#ax.set(xlabel='X position', ylabel='Y position', zlabel='Pressure')
#plt.savefig("plot1.png")
# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph.
#ax.contour(Xs, Ys, Zs, zdir='z', offset=-100, cmap='coolwarm')
#ax.contour(Xs, Ys, Zs, zdir='x', offset=-40, cmap='coolwarm')
#ax.contour(Xs, Ys, Zs, zdir='y', offset=40, cmap='coolwarm')
#plt.show()

