from vocsep.utils.hgraph import *
from vocsep.utils.pianoroll import pianorolls_from_part
import os
from ..dataset import BuiltinDataset, StrutturaDataset
from joblib import Parallel, delayed
from tqdm import tqdm
from ..vocsep import GraphVoiceSeparationDataset

class Bach370ChoralesDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        url = "https://github.com/craigsapp/bach-370-chorales"
        super(Bach370ChoralesDataset, self).__init__(
            name="Bach370ChoralesDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        root = os.path.join(self.raw_path, "kern")
        self.scores = [os.path.join(root, file) for file in os.listdir(root) if file.endswith(".krn")]
        self.collections = ["chorales"] * len(self.scores)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False


# TODO : to complete
class Bach370ChoralesPianorollVoiceSeparationDataset(StrutturaDataset):
    r"""The Bach 370 Chorales Graph Voice Separation Dataset.

    Four-part chorales collected after J.S. Bach's death by his son C.P.E. Bach
    (and finished by Kirnberger, J.S. Bach's student, after C.P.E. Bach's death).
    Ordered by Breitkopf & Härtel numbers.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Bach 370 Chorales Dataset scores are already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """

    def __init__(
        self,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        time_unit="beat",
        time_div=12,
        musical_beat=True,
        nprocs=4,
    ):
        self.dataset_base = Bach370ChoralesDataset(raw_dir=raw_dir)
        self.dataset_base.process()
        if verbose:
            print("Loaded 370 Bach Chorales Dataset Successfully, now processing...")
        self.pianorolls_dicts = list()
        self.n_jobs = nprocs
        self.time_unit = time_unit
        self.time_div = time_div
        self.musical_beat = musical_beat
        super(Bach370ChoralesPianorollVoiceSeparationDataset, self).__init__(
            name="Bach370ChoralesPianorollVoiceSeparationDataset",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        # sys.setrecursionlimit(5000)
        def gfunc(fn, path):
            if fn.endswith(".krn"):
                return pianorolls_from_part(
                    os.path.join(path, fn),
                    self.time_unit,
                    self.time_div,
                    self.musical_beat,
                )

        path = os.path.join(self.raw_dir, "Bach370ChoralesDataset", "kern")
        self.pianorolls_dicts = Parallel(self.n_jobs)(
            delayed(gfunc)(fn, path) for fn in tqdm(os.listdir(path))
        )

        # for fn in tqdm(os.listdir(path)):
        #     out = pianorolls_from_part(os.path.join(path, fn), self.time_unit, self.time_div, self.musical_beat)
        #     self.pianorolls_dicts.append(out)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True

        return False

    # TODO save as numpy arrays for faster loading.
    def save(self):
        """save the pianorolls dicts"""
        if not os.path.exists(os.path.join(self.save_path)):
            os.mkdir(os.path.join(self.save_path))
        for i, prdicts in tqdm(enumerate(self.pianorolls_dicts)):
            file_path = os.path.join(
                self.save_path, os.path.splitext(prdicts["path"])[0] + ".pkl"
            )
            pickle.dump(prdicts, open(file_path, "wb"))

    def load(self):
        for fn in os.listdir(self.save_path):
            path_prdict = os.path.join(self.save_path, fn)
            prdict = pickle.load(open(path_prdict, "rb"))
            self.pianorolls_dicts.append(prdict)

    # Return opnly dense matrices to avoid conflicts with torch dataloader.
    def __getitem__(self, idx):
        return [[self.pianorolls_dicts[i]] for i in idx]

    def __len__(self):
        return len(self.pianorolls_dicts)

    @property
    def save_name(self):
        return self.name

    # @property
    # def features(self):
    #     if self.graphs[0].node_features:
    #         return self.graphs[0].node_features
    #     else:
    #         return list(range(self.graphs[0].x.shape[-1]))


class Bach370ChoralesGraphVoiceSeparationDataset(GraphVoiceSeparationDataset):
    r"""The Bach 370 Chorales Graph Voice Separation Dataset.

    Four-part chorales collected after J.S. Bach's death by his son C.P.E. Bach
    (and finished by Kirnberger, J.S. Bach's student, after C.P.E. Bach's death).
    Ordered by Breitkopf & Härtel numbers.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Bach 370 Chorales Dataset scores are already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=1, include_measures=False):
        dataset_base = Bach370ChoralesDataset(raw_dir=raw_dir)
        super(Bach370ChoralesGraphVoiceSeparationDataset, self).__init__(
            dataset_base=dataset_base,
            is_pyg=False,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
            include_measures=include_measures
        )
