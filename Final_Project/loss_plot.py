# import matplotlib.pyplot as plt
#
# # Sample sizes and corresponding dev accuracies
# sample_sizes = [32, 64, 128, 256, 512, 1024]
# dev_acc_self = [0.6680, 0.6247, 0.6811, 0.7297, 0.7362, 0.7664]
# dev_acc_no_self = [0.6273, 0.6549, 0.6877, 0.7152, 0.7336, 0.7493]  # example baseline
#
# plt.figure(figsize=(9, 5))
# # Plot both lines
# plt.plot(sample_sizes, dev_acc_no_self, marker='s', linestyle='--', linewidth=2, label="True Data")
# plt.plot(sample_sizes, dev_acc_self, marker='o', linestyle='-', linewidth=2, label="Synthetic Data")
#
# plt.xlabel("Synthetic Sample Size", fontsize=12)
# plt.ylabel("Dev Accuracy", fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(sample_sizes, fontsize=10)
# plt.yticks(fontsize=10)
# plt.ylim(0.6, 0.8)
# plt.legend()
#
# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'，如果 TkAgg 不适用的话
# Dev accuracy data from each round
rounds = [1, 2, 3, 4, 5, 6, 7, 8]
dev_accuracy = [80.57, 80.82, 81.06, 81.24, 81.28, 81.32, 80.98, 81.20]

plt.figure(figsize=(9, 5))

# Plot the line
plt.plot(rounds, dev_accuracy, marker='o', linestyle='-', linewidth=2, label="Dev Accuracy")

plt.xlabel("Round", fontsize=12)
plt.ylabel("Dev Accuracy", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rounds, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(80.5, 81.5)  # Adjusting the y-axis for better view of data points
plt.legend()

plt.tight_layout()
plt.show()
