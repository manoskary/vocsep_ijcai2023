from torch_geometric.loader import DataLoader as PygDataLoader
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset
from struttura.data.datasets import (
    MCMAGraphPGVoiceSeparationDataset,
    Bach370ChoralesPGVoiceSeparationDataset,
    HaydnStringQuartetPGVoiceSeparationDataset,
    MCMAGraphVoiceSeparationDataset,
    Bach370ChoralesGraphVoiceSeparationDataset,
    HaydnStringQuartetGraphVoiceSeparationDataset,
    MozartStringQuartetGraphVoiceSeparationDataset,
    MozartStringQuartetPGGraphVoiceSeparationDataset,
)
from torch.nn import functional as F
from collections import defaultdict
from sklearn.model_selection import train_test_split
from struttura.utils import add_reverse_edges_from_edge_index
from struttura.data.samplers import BySequenceLengthSampler
import numpy as np


class GraphPGMixVSDataModule(LightningDataModule):
    def __init__(
        self, batch_size=1, num_workers=4, force_reload=False, test_collections=None, pot_edges_dist = 2
    ):
        super(GraphPGMixVSDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.datasets = [
            # CrimGraphPGVoiceSeparationDataset(
            #     force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist = pot_edges_dist
            # ),
            Bach370ChoralesPGVoiceSeparationDataset(
                force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist = pot_edges_dist
            ),
            MCMAGraphPGVoiceSeparationDataset(
                force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_dist
            ),
            # HaydnStringQuartetPGVoiceSeparationDataset(
            #     force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_dist
            # ),
            # MozartStringQuartetPGGraphVoiceSeparationDataset(
            #     force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_dist
            # )

        ]
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features")
        self.features = self.datasets[0].features
        self.test_collections = test_collections

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i,piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in range(len(dataset))]
        if self.test_collections is None:
            idxs = range(len(self.datasets_map))
            collections = [self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i in idxs]
            trainval_idx, test_idx = train_test_split(idxs, test_size=0.3, stratify=collections, random_state=0)
            trainval_collections = [collections[i] for i in trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections, random_state=0)

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print("Running on all collections")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )
        else:
            # idxs = torch.randperm(len(self.datasets_map)).long()
            idxs = range(len(self.datasets_map))
            test_idx = [
                i
                for i in idxs
                if self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection in self.test_collections
            ]
            trainval_idx = [i for i in idxs if i not in test_idx]
            trainval_collections = [self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i in trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections, random_state=0)
            # nidx = int(len(trainval_idx) * 0.9)
            # train_idx = trainval_idx[:nidx]
            # val_idx = trainval_idx[nidx:]

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print(f"Running evaluation on collections {self.test_collections}")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )
        # compute the ratio between real edges and potential edges
        # real_pot_ratios = list()
        # self.real_pot_ratio = sum([graph["truth_edges_mask"].shape[0]/torch.sum(graph["truth_edges_mask"]) for dataset in self.datasets for graph in dataset.graphs])/len(self.datasets_map)
        self.pot_real_ratio = sum([d.get_positive_weight() for d in self.datasets])/len(self.datasets)

    def train_dataloader(self):
        return PygDataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return PygDataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return PygDataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def num_dropped_truth_edges(self):
        return sum([d.num_dropped_truth_edges() for d in self.datasets])




def idx_tuple_to_dict(idx_tuple, datasets_map):
    """Transforms indices of a list of tuples of indices (dataset, piece_in_dataset) 
    into a dict {dataset: [piece_in_dataset,...,piece_in_dataset]}"""
    result_dict = defaultdict(list)
    for x in idx_tuple:
        result_dict[datasets_map[x][0]].append(datasets_map[x][1])
    return result_dict


class GraphMixVSDataModule(LightningDataModule):
    def __init__(
            self, batch_size=50, num_workers=4, force_reload=False, test_collections=None, pot_edges_max_dist=2, include_measures=False
    ):
        super(GraphMixVSDataModule, self).__init__()
        self.batch_size = batch_size
        self.bucket_boundaries = [200, 300, 400, 500, 700, 1000]
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.include_measures = include_measures
        self.normalize_features = True
        self.datasets = [
            Bach370ChoralesGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist, include_measures=self.include_measures),
            MCMAGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist, include_measures=self.include_measures),
            HaydnStringQuartetGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist, include_measures=self.include_measures),
            MozartStringQuartetGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist, include_measures=self.include_measures),
        ]
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        self.features = self.datasets[0].features
        self.test_collections = test_collections

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i, piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in
                             range(len(dataset))]
        if self.test_collections is None:
            idxs = range(len(self.datasets_map))
            collections = [self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i
                           in idxs]
            trainval_idx, test_idx = train_test_split(idxs, test_size=0.3, stratify=collections, random_state=0)
            trainval_collections = [collections[i] for i in trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections,
                                                  random_state=0)

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print("Running on all collections")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )
        else:
            # idxs = torch.randperm(len(self.datasets_map)).long()
            idxs = range(len(self.datasets_map))
            test_idx = [
                i
                for i in idxs
                if self.datasets[self.datasets_map[i][0]].graphs[
                       self.datasets_map[i][1]].collection in self.test_collections
            ]
            trainval_idx = [i for i in idxs if i not in test_idx]
            trainval_collections = [
                self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i in
                trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections,
                                                  random_state=0)
            # nidx = int(len(trainval_idx) * 0.9)
            # train_idx = trainval_idx[:nidx]
            # val_idx = trainval_idx[nidx:]

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print(f"Running evaluation on collections {self.test_collections}")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )

    def collate_fn(self, batch):
        if self.include_measures:
            batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name, beat_nodes, beat_index, measure_nodes, measure_index = batch[0]
            beat_nodes = torch.tensor(beat_nodes).long().squeeze()
            beat_index = torch.tensor(beat_index).long().squeeze()
            measure_nodes = torch.tensor(measure_nodes).long().squeeze()
            measure_index = torch.tensor(measure_index).long().squeeze()
        else:
            batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name = batch[0]
        batch_inputs = F.normalize(batch_inputs.squeeze(0).float()) if self.normalize_features else batch_inputs.squeeze(0).float()
        batch_label = batch_label.squeeze(0)
        edges = edges.squeeze(0)
        edge_type = edge_type.squeeze(0)
        pot_edges = pot_edges.squeeze(0)
        truth_edges = torch.tensor(truth_edges.squeeze()).to(pot_edges.device)
        na = torch.tensor(na).float()
        if self.include_measures:
            return batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name, beat_nodes, beat_index, measure_nodes, measure_index
        else:
            return batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name

    def collate_train_fn(self, examples):
        lengths = list()
        x = list()
        edge_index = list()
        edge_types = list()
        y = list()
        note_array = list()
        potential_edges = list()
        true_edges = list()
        max_idx = []
        beats = []
        beat_eindex = []
        measures = []
        measure_eindex = []
        for e in examples:
            if self.include_measures:
                batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name, beat_nodes, beat_index, measure_nodes, measure_index = e
                beats.append(torch.tensor(beat_nodes).long())
                beat_eindex.append(torch.tensor(beat_index).long())
                measures.append(torch.tensor(measure_nodes).long())
                measure_eindex.append(torch.tensor(measure_index).long())
            else:
                batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name = e
            x.append(batch_inputs)
            lengths.append(batch_inputs.shape[0])
            edge_index.append(edges)
            edge_types.append(edge_type)
            y.append(batch_label)
            note_array.append(torch.tensor(na))
            max_idx.append(batch_inputs.shape[0])
            potential_edges.append(pot_edges)
            true_edges.append(torch.tensor(truth_edges).long())
        lengths = torch.tensor(lengths).long()
        lengths, perm_idx = lengths.sort(descending=True)
        perm_idx = perm_idx.tolist()
        max_idx = np.cumsum(np.array([0] + [max_idx[i] for i in perm_idx]))
        x = torch.cat([x[i] for i in perm_idx], dim=0).float()
        edge_index = torch.cat([edge_index[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        potential_edges = torch.cat([potential_edges[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        true_edges = torch.cat([true_edges[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        edge_types = torch.cat([edge_types[i] for i in perm_idx], dim=0).long()
        y = torch.cat([y[i] for i in perm_idx], dim=0).long()
        note_array = torch.cat([note_array[i] for i in perm_idx], dim=0).float()
        if self.include_measures:
            max_beat_idx = np.cumsum(np.array([0] + [beats[i].shape[0] for i in perm_idx]))
            beats = torch.cat([beats[i] + max_beat_idx[i] for i in perm_idx], dim=0).long()
            beat_eindex = torch.cat([torch.vstack((beat_eindex[pi][0] + max_idx[i], beat_eindex[pi][1] + max_beat_idx[i])) for i, pi in enumerate(perm_idx)], dim=1).long()
            max_measure_idx = np.cumsum(np.array([0] + [measures[i].shape[0] for i in perm_idx]))
            measures = torch.cat([measures[i] + max_measure_idx[i] for i in perm_idx], dim=0).long()
            measure_eindex = torch.cat([torch.vstack((measure_eindex[pi][0] + max_idx[i], measure_eindex[pi][1] + max_measure_idx[i])) for i, pi in enumerate(perm_idx)], dim=1).long()
            data = x, edge_index, y, edge_types, potential_edges, true_edges, note_array, "batch", beats, beat_eindex, measures, measure_eindex
            return data
        else:
            return x, edge_index, y, edge_types, potential_edges, true_edges, note_array, "batch"

    def train_dataloader(self):
        sampler = BySequenceLengthSampler(self.dataset_train, self.bucket_boundaries, self.batch_size)
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_sampler=sampler,
            batch_size=1,
            num_workers=0,
            collate_fn=self.collate_train_fn,
            drop_last=False,
            pin_memory=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

