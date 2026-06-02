import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import nmslib
import torch
import torch.nn.functional as F


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}
        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)
        self.index_ = index
        return self

    def query(self, vector, topn):
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


def consensus_aware_aggregate(x, edge_index, num_nodes, scale=0.5):
    src, dst = edge_index[0], edge_index[1]

    x_src = x[src]
    x_dst = x[dst]

    raw_scores = (x_src * x_dst).sum(dim=-1) / np.sqrt(x.size(1))

    scores = torch.exp(raw_scores * scale)

    sum_scores = torch.zeros(num_nodes, device=x.device)
    sum_scores.index_add_(0, dst, scores)
    sum_scores = sum_scores.clamp(min=1e-6)

    att_weights = scores / sum_scores[dst]

    weighted_messages = x_src * att_weights.unsqueeze(1)
    x_out = torch.zeros_like(x)
    x_out.index_add_(0, dst, weighted_messages)

    consensus = torch.log(sum_scores)
    consensus = torch.sigmoid(consensus)  # [N]

    return x_out, consensus.unsqueeze(1)  # [N, 1]


def dual_graph_enhancement(coords, features, radius=9):

    if torch.is_tensor(features):
        features_np = features.cpu().numpy()
        features_tensor = features
    else:
        features_np = features
        features_tensor = torch.from_numpy(features).float()

    if torch.is_tensor(coords):
        coords = coords.cpu().numpy()

    num_patches = coords.shape[0]
    device = features_tensor.device

    model_s = Hnsw(space='l2', print_progress=False)
    model_s.fit(coords)
    idx_list_s = [model_s.query(coords[i], topn=radius) for i in range(num_patches)]
    src_s = np.concatenate([row for row in idx_list_s])
    dst_s = np.repeat(range(num_patches), radius)
    edge_index_s = torch.LongTensor(np.stack([src_s, dst_s])).to(device)

    feat_s, conf_s = consensus_aware_aggregate(features_tensor, edge_index_s, num_patches, scale=2.0)

    model_l = Hnsw(space='l2', print_progress=False)
    model_l.fit(features_np)
    idx_list_l = [model_l.query(features_np[i], topn=radius) for i in range(num_patches)]
    src_l = np.concatenate([row for row in idx_list_l])
    dst_l = np.repeat(range(num_patches), radius)
    edge_index_l = torch.LongTensor(np.stack([src_l, dst_l])).to(device)

    feat_l, conf_l = consensus_aware_aggregate(features_tensor, edge_index_l, num_patches, scale=1.0)

    diff_s = feat_s - features_tensor
    diff_l = feat_l - features_tensor
    features_enhanced = features_tensor + (conf_s * diff_s) + (conf_l * diff_l)

    return features_enhanced


def CAGE(h5_dir, pt_dir, save_dir):
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    pbar = tqdm(h5_files)
    success, fail = 0, 0

    for h5_fname in pbar:
        pbar.set_description(f"Processing {h5_fname[:15]}")
        pt_fname = h5_fname.replace('.h5', '.pt')
        h5_full_path = os.path.join(h5_dir, h5_fname)
        pt_full_path = os.path.join(pt_dir, pt_fname)
        save_full_path = os.path.join(save_dir, pt_fname)

        if not os.path.exists(pt_full_path):
            fail += 1
            continue

        try:
            with h5py.File(h5_full_path, "r") as f:
                key = 'coords' if 'coords' in f else 'coordinates'
                coords = np.array(f[key])

            features = torch.load(pt_full_path, map_location='cpu', weights_only=False)
            if isinstance(features, dict) and 'features' in features:
                features = features['features']

            if torch.cuda.is_available():
                features = features.cuda()

            enhanced_feat = dual_graph_enhancement(coords, features, radius=12)

            torch.save(enhanced_feat.cpu(), save_full_path)
            success += 1

        except Exception as e:
            tqdm.write(f"\nError {h5_fname}: {e}")
            fail += 1

    print(f"\nDone. Success: {success}, Fail: {fail}")


def main(args):
    os.makedirs(args.graph_save_path, exist_ok=True)
    CAGE(args.h5_path, args.pt_path, args.graph_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, required=True)
    parser.add_argument('--pt_path', type=str, required=True)
    parser.add_argument('--graph_save_path', type=str, required=True)
    args = parser.parse_args()
    main(args)