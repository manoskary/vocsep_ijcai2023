from struttura.data import StrutturaDataset
from joblib import Parallel, delayed
from tqdm import tqdm
import partitura
import os
from struttura.utils import hetero_graph_from_note_array, select_features, HeteroScoreGraph, score_graph_to_pyg, load_score_hgraph
from struttura.models.core import positional_encoding
import torch
import partitura as pt
import gc
import numpy as np
import torch_geometric as pyg
from numpy.lib.recfunctions import structured_to_unstructured



class GraphVoiceSeparationDataset(StrutturaDataset):
    r"""Parent class for Graph Voice Sepration Datasets.

    Parameters
    -----------
    dataset_base : StrutturaDataset
        The Base Dataset.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """

    def __init__(
            self, dataset_base, is_pyg=False, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2, include_measures=False
    ):
        self.dataset_base = dataset_base
        self.dataset_base.process()
        if verbose:
            print("Loaded {} Successfully, now processing...".format(dataset_base.name))
        self.graphs = list()
        self.n_jobs = nprocs
        self.dropped_notes = 0
        self.pot_edges_max_dist = pot_edges_dist
        self.is_pyg = is_pyg
        self._force_reload = force_reload
        self.include_measures = include_measures
        if self.is_pyg:
            name = self.dataset_base.name.split("Dataset")[0] + "PGGraphVoiceSeparationDataset"
        else:
            name = self.dataset_base.name.split("Dataset")[0] + "GraphVoiceSeparationDataset"
        print("pot_edges_max_dist", self.pot_edges_max_dist)
        super(GraphVoiceSeparationDataset, self).__init__(
            name=name,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_score)(score, collection)
            for score, collection in tqdm(
                zip(self.dataset_base.scores, self.dataset_base.collections)
            )
        )
        self.load()

    def _process_score(self, score_fn, collection):
        if self._force_reload or \
                not (os.path.exists(
                os.path.join(
                    self.save_path, os.path.splitext(os.path.basename(score_fn))[0]
                )) or \
                os.path.exists(
                os.path.join(
                    self.save_path, os.path.splitext(os.path.basename(score_fn))[0] + ".pt"
                ))):
            try:
                score = partitura.load_score(score_fn)
                if len(score.parts) ==0:
                    print("Something is wrong with the score", score_fn)
                    return
            except:
                print("Something is wrong with the score", score_fn)
                return
            note_array = score.note_array(
                include_time_signature=True,
                include_grace_notes=True,
                include_staff=True,
            )
            note_array, num_dropped = preprocess_na_to_monophonic(note_array, score_fn)
            self.dropped_notes += num_dropped
            # Compute the truth edges
            truth_edges = get_mcma_truth_edges(note_array).numpy()
            note_features = select_features(note_array, "voice")
            nodes, edges, pot_edges = hetero_graph_from_note_array(note_array, pot_edge_dist=self.pot_edges_max_dist)
            hg = HeteroScoreGraph(
                note_features,
                edges,
                name=os.path.splitext(os.path.basename(score_fn))[0],
                labels=truth_edges,
                note_array=note_array,
            )
            if self.include_measures:
                measures = score[0].measures
                hg._add_beat_nodes()
                hg._add_measure_nodes(measures)
            # Adding positional encoding to the graph features.
            pos_enc = positional_encoding(hg.edge_index, len(hg.x), 20)
            hg.x = torch.cat((hg.x, pos_enc), dim=1)
            pot_edges = get_mcma_potential_edges(hg, max_dist=self.pot_edges_max_dist)
            # compute the truth edges mask over potential edges
            truth_edges_mask, dropped_truth_edges = get_edges_mask(truth_edges, pot_edges, check_strict_subset=True)
            hg.y = truth_edges_mask
            setattr(hg, "truth_edges", truth_edges)
            # Save edges to use for prediction as a new attribute of the graph.
            setattr(hg, "pot_edges", torch.tensor(pot_edges))
            # Save collection as an attribute of the graph.
            setattr(hg, "collection", collection)
            setattr(hg, "truth_edges_mask", truth_edges_mask)
            setattr(hg, "dropped_truth_edges", dropped_truth_edges)
            if self.is_pyg:
                pg_graph = score_graph_to_pyg(hg)
                file_path = os.path.join(self.save_path, pg_graph["name"] + ".pt")
                torch.save(pg_graph, file_path)
                del pg_graph
            else:
                hg.save(self.save_path)
            del hg, note_array, truth_edges, nodes, edges, note_features, score
            gc.collect()
        return

    def save(self):
        """save the graph list and the labels"""
        pass

    def has_cache(self):
        if self.is_pyg:
            if all(
                    [os.path.exists(os.path.join(self.save_path, os.path.splitext(os.path.basename(path))[0] + ".pt",))
                     for path in self.dataset_base.scores]
            ):
                return True
        else:
            if all(
                    [os.path.exists(os.path.join(self.save_path, os.path.splitext(os.path.basename(path))[0]))
                     for path in self.dataset_base.scores]
            ):
                return True
        return False

    def load(self):
        for fn in os.listdir(self.save_path):
            path_graph = os.path.join(self.save_path, fn)
            graph = torch.load(path_graph) if self.is_pyg else load_score_hgraph(path_graph, fn)
            self.graphs.append(graph)

    def __getitem__(self, idx):
        if self.is_pyg:
            return [[self.graphs[i]] for i in idx]
        else:
            if self.include_measures:
                return [
                [
                    self.graphs[i].x,
                    self.graphs[i].edge_index,
                    self.graphs[i].y,
                    self.graphs[i].edge_type,
                    self.graphs[i].pot_edges,
                    self.graphs[i].truth_edges,
                    structured_to_unstructured(
                        self.graphs[i].note_array[["pitch", "onset_div", "duration_div", "onset_beat", "duration_beat", "ts_beats"]]
                    ),
                    self.graphs[i].name,
                    self.graphs[i].beat_nodes,
                    self.graphs[i].beat_edges,
                    self.graphs[i].measure_nodes,
                    self.graphs[i].measure_edges,
                ]
                for i in idx
            ]
            return [
                [
                    self.graphs[i].x,
                    self.graphs[i].edge_index,
                    self.graphs[i].y,
                    self.graphs[i].edge_type,
                    self.graphs[i].pot_edges,
                    self.graphs[i].truth_edges,
                    structured_to_unstructured(
                        self.graphs[i].note_array[["pitch", "onset_div", "duration_div", "onset_beat", "duration_beat", "ts_beats"]]
                    ),
                    self.graphs[i].name,
                ]
                for i in idx
            ]

    def __len__(self):
        return len(self.graphs)

    @property
    def features(self):
        if self.is_pyg:
            return self.graphs[0]["note"].x.shape[-1]
        else:
            if self.graphs[0].node_features:
                return self.graphs[0].node_features
            else:
                return self.graphs[0].x.shape[-1]

    @property
    def metadata(self):
        if self.is_pyg:
            return self.graphs[0].metadata()
        else:
            return None

    def num_dropped_truth_edges(self):
        if self.is_pyg:
            return sum([len(graph["dropped_truth_edges"]) for graph in self.graphs])
        else:
            return None

    def get_positive_weight(self):
        if self.is_pyg:
            return sum([len(g.truth_edges_mask)/torch.sum(g.truth_edges_mask) for g in self.graphs])/len(self.graphs)
        else:
            raise Exception("Get positive weight not supported for non pyg graphs")


def get_mcma_potential_edges(hg, max_dist=16):
    """Get potential edges for the MCMADataset."""
    # Compute which edge to use for prediction.
    onset_edges = hg.get_edges_of_type("onset")
    during_edges = hg.get_edges_of_type("during")
    consecutive_edges = hg.get_edges_of_type("consecutive")
    consecutive_dense = torch.sparse_coo_tensor(consecutive_edges, torch.ones(consecutive_edges.shape[1]),
                                                size=(len(hg.x), len(hg.x))).to_dense()
    predsub_edges = torch.cat((onset_edges, during_edges), dim=1)
    trim_adj = torch.sparse_coo_tensor(predsub_edges, torch.ones(predsub_edges.shape[1]),
                                       size=(len(hg.x), len(hg.x)))
    trim_adj = trim_adj.to_dense()
    # Remove onset and during edges from a full adjacency matrix.
    trim_adj = torch.ones((len(hg.x), len(hg.x))) - trim_adj
    # Take only the upper triangular part of the adjacency matrix.
    # without the self loops (diagonal=1)
    trim_adj = torch.triu(trim_adj, diagonal=1)
    # remove indices that are further than x units apart.
    trim_adj = trim_adj - torch.triu(trim_adj, diagonal=max_dist)
    # readd consecutive edges if they were deleted
    trim_adj[consecutive_dense == 1] = 1
    # transform to edge index
    pot_edges = pyg.utils.sparse.dense_to_sparse(trim_adj)[0]
    return pot_edges


def get_mcma_truth_edges(note_array):
    """Get the ground truth edges for the MCMA dataset.
    Parameters
    ----------
    note_array : np.array
        The note array of the score.
    Returns
    -------
    truth_edges : np.array
        Ground truth edges.
    """
    part_ids = np.char.partition(note_array["id"], sep="_")[:, 0]
    truth_edges = list()
    # Append edges for consecutive notes in the same voice.
    for un in np.unique(part_ids):
        # Sort indices which are in the same voice.
        voc_inds = np.sort(np.where(part_ids == un)[0])
        # edge indices between consecutive notes in the same voice.
        truth_edges.append(np.vstack((voc_inds[:-1], voc_inds[1:])))
    truth_edges = np.hstack(truth_edges)
    return torch.from_numpy(truth_edges)


def get_edges_mask(subset_edges, total_edges, transpose=True, check_strict_subset=False):
    """Get a mask of edges to use for training.
    Parameters
    ----------
    subset_edges : np.array
        A subset of total_edges.
    total_edges : np.array
        Total edges.
    transpose : bool, optional.
        Whether to transpose the subset_edges, by default True.
        This is necessary if the input arrays are (2, n) instead of (n, 2)
    check_strict_subset : bool, optional
        Whether to check that the subset_edges are a strict subset of total_edges.
    Returns
    -------
    edges_mask : np.array
        Mask that identifies subset edges from total_edges.
    dropped_edges : np.array
        Truth edges that are not in potential edges.
        This is only returned if check_strict_subset is True.
    """
    # convert to numpy, custom types are not supported by torch
    total_edges = total_edges.numpy() if not isinstance(total_edges, np.ndarray) else total_edges
    subset_edges = subset_edges.numpy() if not isinstance(subset_edges, np.ndarray) else subset_edges
    # transpose if r; contiguous is required for the type conversion step later
    if transpose:
        total_edges = np.ascontiguousarray(total_edges.T)
        subset_edges = np.ascontiguousarray(subset_edges.T)
    # convert (n, 2) array to an n array of bytes, in order to use isin, that only works with 1d arrays
    # view_total = total_edges.view(np.dtype((np.void, total_edges.dtype.itemsize * total_edges.shape[-1])))
    # view_subset = subset_edges.view(np.dtype((np.void, subset_edges.dtype.itemsize * subset_edges.shape[-1])))
    view_total = np.char.array(total_edges.astype(str))
    view_subset = np.char.array(subset_edges.astype(str))
    view_total = view_total[:, 0] + "-" + view_total[:, 1]
    view_subset = view_subset[:, 0] + "-" + view_subset[:, 1]
    if check_strict_subset:
        dropped_edges = subset_edges[(~np.isin(view_subset, view_total))]
        if dropped_edges.shape[0] > 0:
            print(f"{dropped_edges.shape[0]} truth edges are not part of potential edges")
        return torch.from_numpy(np.isin(view_total, view_subset)).squeeze(), dropped_edges
    else:
        return torch.from_numpy(np.isin(view_total, view_subset)).squeeze()


def preprocess_na_to_monophonic(note_array, score_fn, drop_extra_voices=True, drop_chords=True):
    """Preprocess the note array to remove polyphonic artifacts.
    Parameters
    ----------
    note_array : np.array
        The note array of the score.
        score_fn : str
        The score filename.
    drop_extra_voices : bool, optional
        Whether to drop extra voices in parts, by default True.
    drop_chords : bool, optional
        Whether to drop chords all notes in chords except the highest, by default True.
        Returns
        -------
    note_array : np.array
        The preprocessed note array.
    """
    num_dropped = 0
    if drop_chords and not drop_extra_voices:
        raise ValueError("Drop chords work correctly only if drop_extra_voices is True.")
    if drop_extra_voices:
        # Check how many voices per part:
        num_voices_per_part = np.count_nonzero(note_array["voice"] > 1)
        if num_voices_per_part > 0:
            print("More than one voice on part of score: {}".format(score_fn))
            print("Dropping {} notes".format(num_voices_per_part))
            num_dropped += num_voices_per_part
            note_array = note_array[note_array["voice"] == 1]
    if drop_chords:
        ids_to_drop = []
        part_ids = np.char.partition(note_array["id"], sep="_")[:, 0]
        for id in np.unique(part_ids):
            part_na = note_array[part_ids == id]
            for onset in np.unique(part_na["onset_div"]):
                if len(part_na[part_na["onset_div"] == onset]) > 1:
                    to_drop = list(part_na[part_na["onset_div"] == onset]["id"][:-1])
                    num_dropped += len(to_drop)
                    ids_to_drop.extend(to_drop)
                    print("Dropping {} notes from chord in score: {}".format(len(to_drop), score_fn))
        return note_array[~np.isin(note_array["id"], ids_to_drop)], num_dropped


def create_polyphonic_groundtruth(part, note_array):
    """Create a groundtruth for polyphonic music.

    The function creates edges between (consecutive) notes in the same voice.
    The current version only works for monophonic parts.
    """

    edges = []
    if isinstance(part, pt.score.Score):
        part = part.parts[0]
    measures = part.measures
    # split note_array to measures where every onset_div is within range of measure start and end
    for m in measures:
        indices = np.nonzero((note_array["onset_div"] >= m.start.t) & (note_array["onset_div"] < m.end.t))[0]
        start_counting = indices.min()
        na = note_array[indices]
        for voice in np.unique(na["voice"]):
            voc_indices = np.nonzero(na["voice"] == voice)[0] + start_counting
            voc_edges = np.vstack((voc_indices[:-1], voc_indices[1:]))
            edges.append(voc_edges)
    return np.hstack(edges).T
