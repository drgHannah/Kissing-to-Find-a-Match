"""***************************************************************************************
*    This code is taken and modified 
*    from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
***************************************************************************************/"""

import numpy as np
import os
from torch.utils.data import Dataset
import glob
# warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch
import random
from torch.utils.data import DataLoader, Sampler, RandomSampler, Dataset
import trimesh
from scipy.spatial.transform import Rotation as R


def savefig(v1):
    ax = plt.axes(projection='3d')
    ax.scatter(v1[:,0], v1[:,1], v1[:,2], c=np.arange(0,v1.shape[0]), cmap='viridis', linewidth=0.5)
    plt.savefig("tmp.png")

def data_augmentation(point_set):
        point_set=point_set.dot(R.from_euler('zyx', [np.random.uniform(0, 360), np.random.uniform(0, 360), np.random.uniform(0, 360)], degrees=True).as_matrix()).astype(point_set.dtype)
        return point_set

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m 
    assert m != 0
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class Surr12kModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000, augm=False):
        self.uniform = uniform
        self.augm = augm
        np.random.seed(0)
        if split =="train":
            self.data = np.load(os.path.join(root, "12k_shapes_train.npy")).astype(dtype=np.float32)
            #self.data = self.data[0:1000,:,:]
        if split =="test":
            self.data = np.load(os.path.join(root, "12k_shapes_test.npy")).astype(dtype=np.float32)
            #self.data = self.data[0:100,:,:]
        if split =="FAUST":
            self.data = np.load(os.path.join(root, "FAUST_noise.npy")).astype(dtype=np.float32)
        EDGES_PATH = os.path.join(root, "12ktemplate.ply")


    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        point_set = self.data[index]
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        if self.augm:
            point_set = data_augmentation(point_set)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return point_set, []

    def __getitem__(self, index):
        return self._get_item(index)




class Tosca_DataLoader(Dataset):
    def __init__(self, root, npoint=1024, uniform=False, augm=False, split = "train"):
        self.uniform = uniform
        self.augm = augm
        self.no_samples = npoint
        root = "/bigdata/Datasets/Tosca/"
        root = os.path.join(root, "tosca_dataset")
        self.root = root
        objects = [f.split("/")[-1].split("0.off")[0] for f in glob.glob(root + "/*0.off") if not "10" in f]

        # get indices human shapes
        id_v, id_m, id_d = objects.index("victoria"), objects.index("michael"), objects.index("david")
        idx_collection = [id_v, id_m, id_d]

        # split train (victoria and michael) and testset (david)
        if split == "test":
            idx_collection = [id_d]
            self.batchsize = 4
            self.class_names = []
            for idx_i in idx_collection:
                with open(os.path.join(root, "tosca_mesh_train_" + objects[idx_i] + ".txt")) as f:
                    list_files = f.read().splitlines()
                self.class_names.append([lists.split("/")[-1] for lists in list_files])
            self.shifts = [0]

        elif split == "train":
            idx_collection = [id_v, id_m]
            self.batchsize = 4
            self.class_names = []
            for idx_i in idx_collection:
                with open(os.path.join(root, "tosca_mesh_train_" + objects[idx_i] + ".txt")) as f:
                    list_files = f.read().splitlines()
                self.class_names.append([lists.split("/")[-1] for lists in list_files])
            self.shifts = np.arange(0,1000,1)
        
        self.data_size = len([item for sublist in self.class_names for item in sublist])

    def load_element(self, list_files, element):
        try:
            mesh = trimesh.load_mesh(os.path.join(self.root, list_files[element]))
        except:
            mesh = trimesh.load_mesh('../'+list_files[element])
        # vert1, tria1 = igl.read_triangle_mesh(list_files[element])#, dtypef='float'
        vert = np.array(mesh.vertices)
        tria = np.array(mesh.faces)
        return vert, tria

    def load_all_elements(self, list_files):
        all_elements = []
        for it in range(len(list_files)):
            vert, _ = self.load_element(list_files, it)
            all_elements.append(vert)
        all_elements = np.stack(all_elements)
        return all_elements

    def sample_object(self, object, no_samples, shift):
        sample_points = self.get_sample_points(object, no_samples, shift)
        return object[sample_points]

    def get_sample_points(self, object, no_samples, shift):
        max_no_vert = object.shape[0]
        sample_points = np.round(np.linspace(0, max_no_vert- 1, no_samples)).astype(int)
        # sample_points = np.arange(0,max_no_vert,max_no_vert//no_samples)[:no_samples]
        return (sample_points+shift) % max_no_vert
 

    def update_base_class(self):
        self.base_class += 1
        self.base_class = self.base_class % 100000

    def __len__(self):
        return self.data_size

    def __getitem__(self,index):
        idx, batchid = index[0],index[1]
        list_files = self.class_names[batchid % len(self.class_names)]
        vert, tria = self.load_element(list_files, idx%len(list_files))
        if self.no_samples != None:
            vert = self.sample_object(vert, self.no_samples, self.shifts[batchid%len(self.shifts)])
            tria = self.sample_object(tria, self.no_samples, self.shifts[batchid%len(self.shifts)])
        if self.uniform:
            vert = farthest_point_sample(vert, self.npoints)
        if self.augm:
            vert = data_augmentation(vert)
        vert[:, 0:3] = pc_normalize(vert[:, 0:3])
        return torch.tensor(vert).float(), torch.tensor(tria)



class MyBatchSampler(Sampler):
    def __init__(self, my_dataset, batch_size=8, drop_last=True):
        self.sampler = RandomSampler(my_dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        batch_id = 0
        for idx in self.sampler:
            batch.append((idx, batch_id))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                batch_id+=1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]



