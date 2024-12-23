import h5py

db = h5py.File("crosswalk_features.hdf5")
print(list(db.keys()))
print(db['features'].shape)
row = db['features'][0]
(label,features) = (row[0], row[1:])
print(label)
print(features.shape)