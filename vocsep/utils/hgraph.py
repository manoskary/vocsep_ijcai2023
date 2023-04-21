import os
import random, string
import pickle
import numpy as np
from vocsep.utils.general import exit_after

from vocsep.descriptors.general import *
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class HeteroScoreGraph(object):
    def __init__(self, note_features, edges, etypes=["onset", "consecutive", "during", "rest"], name=None, note_array=None, edge_weights=None, labels=None):
        self.node_features = note_features.dtype.names if note_features.dtype.names else []
        self.features = note_features
        # Filter out string fields of structured array.
        if self.node_features:
            self.node_features = [feat for feat in self.node_features if note_features.dtype.fields[feat][0] != np.dtype('U256')]
            self.features = self.features[self.node_features]
        self.x = torch.from_numpy(np.asarray(rfn.structured_to_unstructured(self.features) if self.node_features else self.features, dtype=np.float32))
        assert etypes is not None
        self.etypes = {t: i for i, t in enumerate(etypes)}
        self.note_array = note_array
        self.edge_type = torch.from_numpy(edges[-1]).long()
        self.edge_index = torch.from_numpy(edges[:2]).long()
        self.edge_weights = torch.ones(len(self.edge_index[0])) if edge_weights is None else torch.from_numpy(edge_weights)
        self.name = name
        self.y = labels if labels is None else torch.from_numpy(labels)

    def adj(self, weighted=False):
        if weighted:
            return torch.sparse_coo_tensor(self.edge_index, self.edge_weights, (len(self.x), len(self.x)))
        ones = torch.ones(len(self.edge_index[0]))
        matrix = torch.sparse_coo_tensor(self.edge_index, ones, (len(self.x), len(self.x)))
        return matrix

    def _add_measure_nodes(self, measures):
        """Add virtual nodes for every measure"""
        assert "onset_div" in self.note_array.dtype.names, "Note array must have 'onset_div' field to add measure nodes."
        # if not hasattr(self, "beat_nodes"):
        #     self._add_beat_nodes()
        nodes = np.arange(len(measures))
        # Add new attribute to hg
        edges = []
        for i, m in enumerate(measures):
            idx = np.where((self.note_array["onset_div"] >= m.start.t) & (self.note_array["onset_div"] < m.end.t))[0]
            if idx.size:
                edges.append(np.vstack((idx, np.full(idx.size, i))))
        self.measure_nodes = nodes
        self.measure_edges = np.hstack(edges)

    def _add_beat_nodes(self):
        """Add virtual nodes for every beat"""
        assert "onset_beat" in self.note_array.dtype.names, "Note array must have 'onset_beat' field to add measure nodes."
        nodes = np.arange(int(self.note_array["onset_beat"].max()))
        # Add new attribute to hg

        edges = []
        for b in nodes:
            idx = np.where((self.note_array["onset_beat"] >= b) & (self.note_array["onset_beat"] < b + 1))[0]
            if idx.size:
                edges.append(np.vstack((idx, np.full(idx.size, b))))
        self.beat_nodes = nodes
        self.beat_edges = np.hstack(edges)

    def assign_typed_weight(self, weight_dict:dict):
        assert weight_dict.keys() == self.etypes.keys()
        for k, v in weight_dict.items():
            etype = self.etypes[k]
            self.edge_weights[self.edge_type == etype] = v

    def get_edges_of_type(self, etype):
        assert etype in self.etypes.keys()
        etype = self.etypes[etype]
        return self.edge_index[:, self.edge_type == etype]

    def save(self, save_dir):
        save_name = self.name if self.name else ''.join(random.choice(string.ascii_letters) for i in range(10))
        (os.makedirs(os.path.join(save_dir, save_name)) if not os.path.exists(os.path.join(save_dir, save_name)) else None)
        with open(os.path.join(save_dir, save_name, "x.npy"), "wb") as f:
            np.save(f, self.x.numpy())
        with open(os.path.join(save_dir, save_name, "edge_index.npy"), "wb") as f:
            np.save(f, torch.cat((self.edge_index, self.edge_type.unsqueeze(0))).numpy())
        if isinstance(self.y, torch.Tensor):
            with open(os.path.join(save_dir, save_name, "y.npy"), "wb") as f:
                np.save(f, self.y.numpy())
        if isinstance(self.edge_weights, torch.Tensor):
            np.save(open(os.path.join(save_dir, save_name, "edge_weights.npy"), "wb"), self.edge_weights.numpy())
        if isinstance(self.note_array, np.ndarray):
            np.save(open(os.path.join(save_dir, save_name, "note_array.npy"), "wb"), self.note_array)
        with open(os.path.join(save_dir, save_name, 'graph_info.pkl'), 'wb') as handle:
            object_properties = vars(self)
            del object_properties['x']
            del object_properties['edge_index']
            del object_properties['edge_type']
            del object_properties['edge_weights']
            del object_properties['y']
            del object_properties['note_array']
            pickle.dump(object_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)


@exit_after(10)
def load_score_hgraph(load_dir, name=None):
    path = os.path.join(load_dir, name) if os.path.basename(load_dir) != name else load_dir
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError("The directory is not recognized.")
    x = np.load(open(os.path.join(path, "x.npy"), "rb"))
    edge_index = np.load(open(os.path.join(path, "edge_index.npy"), "rb"))
    graph_info = pickle.load(open(os.path.join(path, "graph_info.pkl"), "rb"))
    y = np.load(open(os.path.join(path, "y.npy"), "rb")) if os.path.exists(os.path.join(path, "y.npy")) else None
    edge_weights = np.load(open(os.path.join(path, "edge_weights.npy"), "rb")) if os.path.exists(os.path.join(path, "edge_weights.npy")) else None
    note_array = np.load(open(os.path.join(path, "note_array.npy"), "rb")) if os.path.exists(
        os.path.join(path, "note_array.npy")) else None
    name = name if name else os.path.basename(path)
    hg = HeteroScoreGraph(note_features=x, edges=edge_index, name=name, labels=y, edge_weights=edge_weights, note_array=note_array)
    for k, v in graph_info.items():
        setattr(hg, k, v)
    return hg


def check_note_array(na):
    dtypes = na.dtype.names
    if not all([x in dtypes for x in ["onset_beat", "duration_beat", "ts_beats", "ts_beat_type"]]):
        raise(TypeError("The given Note array is missing necessary fields."))


# def hetero_graph_from_note_array(note_array, rest_array=None, norm2bar=True):
#     '''Turn note_array to homogeneous graph dictionary.

#     Parameters
#     ----------
#     note_array : structured array
#         The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
#     rest_array : structured array
#         A structured rest array similar to the note array but for rests.
#     t_sig : list
#         A list of time signature in the piece.
#     '''

#     edg_src = list()
#     edg_dst = list()
#     etype = list()
#     start_rest_index = len(note_array)
#     for i, x in enumerate(note_array):
#         for j in np.where((np.isclose(note_array["onset_beat"], x["onset_beat"], rtol=1e-04, atol=1e-04) == True) & (note_array["pitch"] != x["pitch"]))[0]:
#             edg_src.append(i)
#             edg_dst.append(j)
#             etype.append(0)

#         for j in np.where(np.isclose(note_array["onset_beat"], x["onset_beat"] + x["duration_beat"], rtol=1e-04, atol=1e-04) == True)[0]:
#             edg_src.append(i)
#             edg_dst.append(j)
#             etype.append(1)

#         if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
#             for j in np.where(np.isclose(rest_array["onset_beat"], x["onset_beat"] + x["duration_beat"], rtol=1e-04, atol=1e-04) == True)[0]:
#                 edg_src.append(i)
#                 edg_dst.append(j + start_rest_index)
#                 etype.append(1)

#         for j in np.where(
#                 (x["onset_beat"] < note_array["onset_beat"]) & (x["onset_beat"] + x["duration_beat"] > note_array["onset_beat"]))[0]:
#             edg_src.append(i)
#             edg_dst.append(j)
#             etype.append(2)

#     if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
#         for i, r in enumerate(rest_array):
#             for j in np.where(np.isclose(note_array["onset_beat"], r["onset_beat"] + r["duration_beat"], rtol=1e-04, atol=1e-04) == True)[0]:
#                 edg_src.append(start_rest_index + i)
#                 edg_dst.append(j)
#                 etype.append(1)

#         feature_fn = [dname for dname in note_array.dtype.names if dname not in rest_array.dtype.names]
#         if feature_fn:
#             rest_feature_zeros = np.zeros((len(rest_array), len(feature_fn)))
#             rest_feature_zeros = rfn.unstructured_to_structured(rest_feature_zeros, dtype=list(map(lambda x: (x, '<4f'), feature_fn)))
#             rest_array = rfn.merge_arrays((rest_array, rest_feature_zeros))
#     else:
#         end_times = note_array["onset_beat"] + note_array["duration_beat"]
#         for et in np.sort(np.unique(end_times))[:-1]:
#             if et not in note_array["onset_beat"]:
#                 scr = np.where(end_times == et)[0]
#                 diffs = note_array["onset_beat"] - et
#                 tmp = np.where(diffs > 0, diffs, np.inf)
#                 dst = np.where(tmp == tmp.min())[0]
#                 for i in scr:
#                     for j in dst:
#                         edg_src.append(i)
#                         edg_dst.append(j)
#                         etype.append(1)


#     edges = np.array([edg_src, edg_dst, etype])
#     # Resize Onset Beat to bar
#     if norm2bar:
#         note_array["onset_beat"] = np.mod(note_array["onset_beat"], note_array["ts_beats"])
#         if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
#             rest_array["onset_beat"] = np.mod(rest_array["onset_beat"], rest_array["ts_beats"])

#     nodes = np.hstack((note_array, rest_array))
#     return nodes, edges

def hetero_graph_from_note_array(note_array, rest_array=None, norm2bar=False, pot_edge_dist=0):
    '''Turn note_array to homogeneous graph dictionary.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    rest_array : structured array
        A structured rest array similar to the note array but for rests.
    t_sig : list
        A list of time signature in the piece.
    '''

    edg_src = list()
    edg_dst = list()
    etype = list()
    pot_edges = list()
    start_rest_index = len(note_array)
    for i, x in enumerate(note_array):
        for j in np.where(np.isclose(note_array["onset_div"], x["onset_div"], rtol=1e-04, atol=1e-04) == True)[0]:
            if i != j:
                edg_src.append(i)
                edg_dst.append(j)
                etype.append(0)
        if pot_edge_dist:
            for j in np.where(
                    (note_array["onset_div"] > x["onset_div"]+x["duration_div"]) &
                    (note_array["onset_beat"] <= x["onset_beat"] + x["duration_beat"] + pot_edge_dist*x["ts_beats"])
            )[0]:
                pot_edges.append([i, j])
        for j in np.where(np.isclose(note_array["onset_div"], x["onset_div"] + x["duration_div"], rtol=1e-04, atol=1e-04) == True)[0]:
            edg_src.append(i)
            edg_dst.append(j)
            etype.append(1)

        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            for j in np.where(np.isclose(rest_array["onset_div"], x["onset_div"] + x["duration_div"], rtol=1e-04, atol=1e-04) == True)[0]:
                edg_src.append(i)
                edg_dst.append(j + start_rest_index)
                etype.append(1)

        for j in np.where(
                (x["onset_div"] < note_array["onset_div"]) & (x["onset_div"] + x["duration_div"] > note_array["onset_div"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)
            etype.append(2)

    if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
        for i, r in enumerate(rest_array):
            for j in np.where(np.isclose(note_array["onset_div"], r["onset_div"] + r["duration_div"], rtol=1e-04, atol=1e-04) == True)[0]:
                edg_src.append(start_rest_index + i)
                edg_dst.append(j)
                etype.append(1)

        feature_fn = [dname for dname in note_array.dtype.names if dname not in rest_array.dtype.names]
        if feature_fn:
            rest_feature_zeros = np.zeros((len(rest_array), len(feature_fn)))
            rest_feature_zeros = rfn.unstructured_to_structured(rest_feature_zeros, dtype=list(map(lambda x: (x, '<4f'), feature_fn)))
            rest_array = rfn.merge_arrays((rest_array, rest_feature_zeros))
    else:
        end_times = note_array["onset_div"] + note_array["duration_div"]
        for et in np.sort(np.unique(end_times))[:-1]:
            if et not in note_array["onset_div"]:
                scr = np.where(end_times == et)[0]
                diffs = note_array["onset_div"] - et
                tmp = np.where(diffs > 0, diffs, np.inf)
                dst = np.where(tmp == tmp.min())[0]
                for i in scr:
                    for j in dst:
                        edg_src.append(i)
                        edg_dst.append(j)
                        etype.append(3)


    edges = np.array([edg_src, edg_dst, etype])

    # Resize Onset Beat to bar
    if norm2bar:
        note_array["onset_beat"] = np.mod(note_array["onset_beat"], note_array["ts_beats"])
        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            rest_array["onset_beat"] = np.mod(rest_array["onset_beat"], rest_array["ts_beats"])

    nodes = np.hstack((note_array, rest_array))
    if pot_edge_dist:
        pot_edges = np.hstack((np.array(pot_edges).T, edges[:, edges[2] == 1][:2]))
        return nodes, edges, pot_edges
    return nodes, edges


@exit_after(120)
def hetero_graph_from_part(x : Union[Union[partitura.score.Part, partitura.score.PartGroup], np.ndarray], features=None, name=None, norm2bar=True, include_rests=False, labels=None) -> HeteroScoreGraph:
    if isinstance(x, partitura.score.Score) or isinstance(x, partitura.score.Part) or isinstance(x, partitura.score.PartGroup) or isinstance(x, list):
        part = x
        part = partitura.score.merge_parts(part)
        part = partitura.score.unfold_part_maximal(part)
        note_array = part.note_array(include_time_signature=True, include_grace_notes=True, include_staff=True)
        note_features = select_features(part, features)

        if include_rests:
            rest_array = part.rest_array(include_time_signature=True, include_grace_notes=True, collapse=True, include_staff=True)
            if labels is None:
                labels = np.vstack((rfn.structured_to_unstructured(note_array[["voice", "staff"]]),
                                    rfn.structured_to_unstructured(rest_array[["voice", "staff"]])))
        else:
            rest_array = None
            if labels is None:
                labels = rfn.structured_to_unstructured(note_array[["voice", "staff"]])
    elif isinstance(x, str):
        return hetero_graph_from_part(partitura.load_score(x), features=features, name=name, norm2bar=norm2bar, include_rests=include_rests)
    else:
        check_note_array(x)
        note_array = note_features = x
        rest_array = None
        if labels is None:
            labels = rfn.structured_to_unstructured(note_array[["voice", "staff"]])
    nodes, edges = hetero_graph_from_note_array(note_array, rest_array, norm2bar=norm2bar)
    return HeteroScoreGraph(note_features, edges, name=name, labels=labels, note_array=note_array)


def voice_from_edges(edges, number_of_nodes):
    """
        Assign to each disconnected node cluster an unique voice number.
    """
    data = np.ones(edges.shape[1])
    row, col = edges
    scipy_graph = csr_matrix((data, (row.cpu(), col.cpu())), shape=(number_of_nodes, number_of_nodes))       
    number_of_voices, voices = connected_components(csgraph=scipy_graph, directed=False, return_labels=True)
    return voices+1, number_of_voices


def adj_matrix_from_edges(edges, number_of_nodes):
    """
        Create adjacency matrix from edges.
    """
    data = np.ones(edges.shape[1])
    row, col = edges
    scipy_graph = csr_matrix((data, (row.cpu(), col.cpu())), shape=(number_of_nodes, number_of_nodes))       
    return scipy_graph
        

def add_reverse_edges(graph, mode):
    if mode == "new_type":
        # Add reverse During Edges
        graph.edge_index = torch.cat((graph.edge_index, graph.get_edges_of_type("during").flip(0)), dim=1)
        graph.edge_type = torch.cat((graph.edge_type, 2 + torch.zeros(graph.edge_index.shape[1] - graph.edge_type.shape[0],dtype=torch.long)), dim=0)
        # Add reverse Consecutive Edges
        graph.edge_index = torch.cat((graph.edge_index, graph.get_edges_of_type("consecutive").flip(0)), dim=1)
        graph.edge_type = torch.cat((graph.edge_type, 4+torch.zeros(graph.edge_index.shape[1] - graph.edge_type.shape[0], dtype=torch.long)), dim=0)
        graph.etypes["consecutive_rev"] = 4
    else:
        graph.edge_index = torch.cat((graph.edge_index, graph.edge_index.flip(0)), dim=1)
        raise NotImplementedError("To undirected is not Implemented for HeteroScoreGraph.")
    return graph


def add_reverse_edges_from_edge_index(edge_index, edge_type, mode="new_type"):
    if mode == "new_type":
        unique_edge_types = torch.unique(edge_type)
        for type in unique_edge_types:
            if type == 0:
                continue
            edge_index = torch.cat((edge_index, edge_index[:, edge_type == type].flip(0)), dim=1)
            edge_type = torch.cat((edge_type, torch.max(edge_type) + torch.zeros(edge_index.shape[1] - edge_type.shape[0], dtype=torch.long).to(edge_type.device)), dim=0)
    else:
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        edge_type = torch.cat((edge_type, edge_type), dim=0)
    return edge_index, edge_type