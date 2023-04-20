from struttura.utils.graph import *
from struttura.utils.hgraph import *
import gc
import os
from struttura.data.dataset import BuiltinDataset, StrutturaDataset
from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.lib.recfunctions import structured_to_unstructured
from struttura.models.core import positional_encoding
from struttura.data.datasets.mcma import preprocess_na_to_monophonic, get_mcma_truth_edges, get_mcma_potential_edges, get_edges_mask
from struttura.data.vocsep import GraphVoiceSeparationDataset


class HaydnStringQuartetDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        url = "https://github.com/musedata/humdrum-haydn-quartets"
        super(HaydnStringQuartetDataset, self).__init__(
            name="HaydnStringQuartets",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        root = os.path.join(self.raw_path, "kern")
        self.scores = [os.path.join(root, file) for file in os.listdir(root) if file.endswith(".krn")]
        self.collections = ["haydn"]*len(self.scores)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


class HaydnStringQuartetGraphVoiceSeparationDataset(GraphVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2, include_measures=False):
        r"""The Haydn String Quartet Graph Voice Separation Dataset.

    Four-part Haydn string quartets digital edition of the quartets composed by Joseph Haydn,
    encoded in the Humdrum file format.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Haydn String Quartet Dataset scores are already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
        dataset_base = HaydnStringQuartetDataset(raw_dir=raw_dir)
        super(HaydnStringQuartetGraphVoiceSeparationDataset, self).__init__(
            dataset_base=dataset_base,
            raw_dir=raw_dir,
            is_pyg=False,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
            include_measures=include_measures,
        )

class HaydnStringQuartetPGVoiceSeparationDataset(GraphVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2):
        r"""The Haydn String Quartet Graph Voice Separation Dataset.

    Four-part Haydn string quartets digital edition of the quartets composed by Joseph Haydn,
    encoded in the Humdrum file format.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Haydn String Quartet Dataset scores are already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
        dataset_base = HaydnStringQuartetDataset(raw_dir=raw_dir)
        super().__init__(
            dataset_base=dataset_base,
            raw_dir=raw_dir,
            is_pyg=True,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
        )


# class HaydnStringQuartetPGVoiceSeparationDataset(StrutturaDataset):
#     r"""The Haydn String Quartet Graph Voice Separation Dataset for pytorch geometric.

#     Four-part Haydn string quartets digital edition of the quartets composed by Joseph Haydn,
#     encoded in the Humdrum file format.

#     Parameters
#     -----------
#     raw_dir : str
#         Raw file directory to download/contains the input data directory.
#         Dataset will search if  Haydn String Quartet Dataset scores are already available otherwise it will download it.
#         Default: ~/.struttura/
#     force_reload : bool
#         Whether to reload the dataset. Default: False
#     verbose : bool
#         Whether to print out progress information. Default: True.
#     """

#     def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_max_dist=8):
#         self.dataset_base = HaydnStringQuartetDataset(raw_dir=raw_dir)
#         self.dataset_base.process()
#         if verbose:
#             print("Loaded The Haydn String Quartet Dataset Successfully, now processing...")
#         self.graphs = list()
#         self.n_jobs = nprocs
#         self.pot_edges_max_dist = pot_edges_max_dist
#         print("pot_edges_max_dist", self.pot_edges_max_dist)
#         super(HaydnStringQuartetPGVoiceSeparationDataset, self).__init__(
#             name="HaydnStringQuartetPGVoiceSeparationDataset",
#             raw_dir=raw_dir,
#             force_reload=force_reload,
#             verbose=verbose,
#         )

#     def process(self):
#         if not os.path.exists(self.save_path):
#             os.makedirs(self.save_path)
#         Parallel(self.n_jobs)(
#             delayed(self._process_score)(fn) for fn in tqdm(self.dataset_base.scores)
#         )
#         self.load()

#     def _process_score(self, score_fn, collection="haydn"):
#         if not os.path.exists(
#                 os.path.join(
#                     self.save_path, os.path.splitext(os.path.basename(score_fn))[0]
#                 )
#         ):
#             try:
#                 score = partitura.load_kern(score_fn)
#             except:
#                 return
#             note_array = score.note_array(
#                 include_time_signature=True,
#                 include_grace_notes=True,
#                 include_staff=True,
#             )
#             # preprocess to remove extra voices and chords
#             note_array = preprocess_na_to_monophonic(note_array, score_fn)
#             # build the HeteroGraph
#             nodes, edges, pot_edges = hetero_graph_from_note_array(note_array, pot_edge_dist=2)
#             note_features = select_features(note_array, "voice")
#             hg = HeteroScoreGraph(
#                 note_features,
#                 edges,
#                 name=os.path.splitext(os.path.basename(score_fn))[0],
#                 labels=None,
#                 note_array=note_array,
#             )
#             # Adding positional encoding to the graph features.
#             pos_enc = positional_encoding(hg.edge_index, len(hg.x), 20)
#             hg.x = torch.cat((hg.x, pos_enc), dim=1)
#             # Compute the truth edges
#             truth_edges = get_mcma_truth_edges(note_array)
#             setattr(hg, "truth_edges", truth_edges)
#             # Compute the potential edges to use for prediction.
#             # pot_edges = get_mcma_potential_edges(hg, max_dist=self.pot_edges_max_dist)
#             setattr(hg, "pot_edges", torch.tensor(pot_edges))
#             # compute the truth edges mask over potential edges
#             truth_edges_mask, dropped_truth_edges = get_edges_mask(truth_edges, pot_edges, check_strict_subset=True)
#             setattr(hg, "truth_edges_mask", truth_edges_mask)
#             setattr(hg, "dropped_truth_edges", dropped_truth_edges)
#             setattr(hg, "collection", collection)
#             pg_graph = score_graph_to_pyg(hg)
#             file_path = os.path.join(self.save_path, pg_graph["name"] + ".pt")
#             torch.save(pg_graph, file_path)
#             del hg, note_array, truth_edges, nodes, edges, note_features, score
#             del pg_graph
#         return

#     def has_cache(self):
#         if all(
#             [
#                 os.path.exists(
#                     os.path.join(
#                         self.save_path,
#                         os.path.splitext(os.path.basename(path))[0] + ".pt",
#                     )
#                 )
#                 for path in self.dataset_base.scores
#             ]
#         ):
#             return True
#         ### WARNING:::: THIS IS  A TEMPORAL FIX, UNTIL MANOS CORRECT THE INPUT
#         return True

#     def load(self):
#         for fn in os.listdir(self.save_path):
#             path_graph = os.path.join(self.save_path, fn)
#             graph = torch.load(path_graph)
#             self.graphs.append(graph)

#     def __getitem__(self, idx):
#         return [[self.graphs[i]] for i in idx]

#     def __len__(self):
#         return len(self.graphs)

#     def save(self):
#         pass

#     @property
#     def metadata(self):
#         return self.graphs[0].metadata()

#     @property
#     def features(self):
#         return self.graphs[0]["note"].x.shape[-1]

#     def num_dropped_truth_edges(self):
#         return sum([len(graph["dropped_truth_edges"]) for graph in self.graphs])