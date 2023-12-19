import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 8*np.pi, 1000)

t_1 = np.ones(1000)
t_2 = 3*np.ones(1000)

line_over = np.cos(x/2 + np.pi/4) + 3
line_under = 2*np.sin(x/3)/3 + 2
line_both = 2*np.cos(x/4) + 2
plt.plot(x, t_1, linestyle='--', color="r", label="$T_1$")
plt.plot(x, t_2, linestyle='--', color="b", label="$T_2$")

plt.plot(x, line_over, color="y", label="Line 1")
plt.plot(x, line_under, color="g", label="Line 2")
plt.plot(x, line_both, color="m", label="Line 3")

plt.title('Lines before hysteresis')
plt.xlabel('x-location')
plt.ylabel('Value')
plt.ylim(0-0.5, np.max(line_over) + 0.5)
plt.legend(loc="lower left")
plt.show()


out_id = np.where(line_both > 1)

split = np.argmax(out_id[0][1:] - out_id[0][:-1])

plt.plot(x, t_1, linestyle='--',  color="r", label="$T_1$")
plt.plot(x, t_2, linestyle='--', color="b", label="$T_2$")

plt.plot(x, line_over, color="y", label="Line 1")

plt.plot(x[out_id[0][:split]], line_both[out_id[0][:split]], color="m", label="Line 3")
plt.plot(x[out_id[0][split + 1:]], line_both[out_id[0][split + 1:]], color="m")#    , label="Line 3")
plt.title('Lines after hysteresis')
plt.xlabel('x-location')
plt.ylabel('Value')
plt.ylim(0-0.5, np.max(line_over) + 0.5)
plt.legend(loc="lower left")

plt.show()



# plt.show()







