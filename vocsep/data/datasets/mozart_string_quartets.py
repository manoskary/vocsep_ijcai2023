import os
from struttura.data.dataset import BuiltinDataset
from struttura.data.vocsep import GraphVoiceSeparationDataset


class MozartStringQuartetDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        url = "https://github.com/musedata/humdrum-mozart-quartets"
        super(MozartStringQuartetDataset, self).__init__(
            name="MozartStringQuartets",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        root = os.path.join(self.raw_path, "kern")
        self.scores = [os.path.join(root, file) for file in os.listdir(root) if file.endswith(".krn")]
        self.collections = ["mozart"]*len(self.scores)

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


class MozartStringQuartetGraphVoiceSeparationDataset(GraphVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2, include_measures=False):
        r"""The Mozart String Quartet Graph Voice Separation Dataset.

    Four-part Mozart string quartets digital edition of the quartets composed by Wolfgang Amadeus Mozart,
    encoded in the Humdrum file format.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Mozart String Quartet Dataset scores are already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
        dataset_base = MozartStringQuartetDataset(raw_dir=raw_dir)
        super(MozartStringQuartetGraphVoiceSeparationDataset, self).__init__(
            dataset_base=dataset_base,
            raw_dir=raw_dir,
            is_pyg=False,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
            include_measures=include_measures,
        )


class MozartStringQuartetPGGraphVoiceSeparationDataset(GraphVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, pot_edges_dist=2):
        r"""The Mozart String Quartet Graph Voice Separation Dataset.

    Four-part Mozart string quartets digital edition of the quartets composed by Wolfgang Amadeus Mozart,
    encoded in the Humdrum file format.

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Dataset will search if Mozart String Quartet Dataset scores are already available otherwise it will download it.
        Default: ~/.struttura/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    """
        dataset_base = MozartStringQuartetDataset(raw_dir=raw_dir)
        super().__init__(
            dataset_base=dataset_base,
            raw_dir=raw_dir,
            is_pyg=True,
            force_reload=force_reload,
            verbose=verbose,
            nprocs=nprocs,
            pot_edges_dist=pot_edges_dist,
        )