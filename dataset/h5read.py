import h5py

f = h5py.File('./wv3/train.h5', 'r')
print("---keys---")
print(f.keys())
print("---key size---")
print(len(f['pan'][:]))
for i in range(len(f['pan'][:])):
    print(f['pan'][i].shape)
    print(f['ms'][i].shape)
    print(f['lms'][i].shape)

f.close()

