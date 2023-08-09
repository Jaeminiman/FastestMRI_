import h5py
import matplotlib.pyplot as plt
f = h5py.File('/data/wpals113/workspace/brain_acc4_179.h5')
# f = h5py.File('/data/wpals113/workspace/Fastest/home/result/test_varnet/reconstructions_val/brain_acc4_179.h5', 'r')
# f = h5py.File('YOUR FILE PATH', 'r')
print(list(f.keys()))


recon = f['reconstruction']
target = f['target']

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(recon[15, :, :])
plt.title('reconstruction')
plt.subplot(1, 2, 2)
plt.imshow(target[15, :, :])
plt.title('target')
plt.savefig('/data/wpals113/workspace/result.png', dpi=300)