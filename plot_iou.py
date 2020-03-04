import matplotlib.pyplot as plt
import numpy as np

line_prop = ['-or','-xb','-vg','-ok','-oc']
labels = ['AlexNet','ResNet18','Random selection']
x=[7,10,15]
y=[[0.6983,0.7376,0.6976],[0.7111,0.7271,0.7639],[0.6373,0.6403,0.6606]]
max_x = -1
for i in range(0,len(labels)):
    plt.plot(x, y[i], line_prop[i],label=labels[i])

plt.grid()
plt.legend()
plt.xlabel("k (#frames used for training)")
plt.ylabel("IOU on test dataset")
# plt.title("k vs IOU for different frame selection strategy")
# plt.xticks(np.arange(0,12,step=1))
# plt.xticks(rotation='vertical')
# plt.subplots_adjust(bottom=0.25)
pdfname = "k_vs_iou.pdf"
plt.savefig(pdfname, format = "pdf")
# plt.show()