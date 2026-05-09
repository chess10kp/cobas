import matplotlib.pyplot as plt
import numpy as np

temps_soc0 = [24.2, 23.9, 24.2, 24.5, 24.1, 24.1, 24.6, 24.6, 24.6, 25.1, 25.1, 25.4, 25.6, 25.9, 25.9, 26.1, 26.5, 26.5, 26.7, 27.0, 27.0, 27.1, 27.6, 27.6, 27.7, 27.9, 28.1, 28.1, 28.4, 28.5, 28.6, 28.7, 28.7, 28.7, 29.1, 29.2, 29.2, 29.2, 29.2, 29.4, 29.6]
temps_soc50 = [24.6, 24.2, 24.6, 24.6, 24.9, 25.1, 25.2, 25.4, 25.6, 25.9, 26.1, 26.1, 26.5, 26.6, 26.7, 27.0, 27.1, 27.4, 27.4, 27.6, 27.6, 27.9, 27.9, 28.1, 28.1, 28.5, 28.2, 28.2, 28.5, 28.6, 28.6, 28.7, 28.6, 28.6, 29.0, 29.0, 29.0, 29.1, 29.1, 29.2, 29.1]
time = np.arange(0, 10.25, 0.25)

print(len(temps_soc50))
print(len(time))

# m, b = np.polyfit(time, temps_soc0, 1)
# trend = m * np.array(time) + b

plt.scatter(time, temps_soc0, label="0% SoC")
# plt.plot(time, trend, color='red', linewidth=2, label="Trend line")
# plt.title("Temperature vs Time (0% - 16%SoC)")
# plt.xlabel("Time (s)")
# plt.ylabel("Temperature (ºC)")
# plt.grid(True)
# plt.legend()
# plt.savefig("0soc.png")
# plt.show()

# m, b = np.polyfit(time, temps_soc50, 1)
# trend = m * np.array(time) + b

plt.scatter(time, temps_soc50, label="50% SoC")
# plt.plot(time, trend, color='red', linewidth=2, label="Trend line")
plt.title("Temperature vs Time (16% Charge Variation)")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (ºC)")
plt.grid(True)
plt.legend()
plt.savefig("0and50soc.png")
plt.show()