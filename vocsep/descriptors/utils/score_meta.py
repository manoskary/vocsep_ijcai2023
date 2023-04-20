from partitura import score
import itertools

SCORE_META = {
	# "Systems" : score.System(number=1),
	"DaCapo" : score.DaCapo,
	"Fine" : score.Fine,
	"Repeat" : score.Repeat,
	"Barline" : score.Barline,
	"Fermata" : score.Fermata,
	"TimeSignature" : score.TimeSignature,
	# "Tempo" : score.Tempo,
	"KeySignature" : score.KeySignature,
	# "Words" : score.Words
}


def iter_all_meta(part, meta):
	return [part.beat_map(at.start.t).tolist() for at in part.iter_all(meta) if at.start.t !=0]

def get_score_meta(part):
	"""
	Export positions where score events (double lines, time signature changes, etc) occur.

	Parameters
	----------
	part : obj
		A partitura part object
	Returns
	-------
	pivatal_points : list()
		Bars or beats in the score where changes in score meta data happen.
	"""
	# for key, meta in SCORE_META.items():
	# 	print("{} is {} : ".format(key, iter_all_meta(part, meta)))
	if isinstance(part, list):
		part = part[0]
	pivotal_points = sorted(list(set(
		list(itertools.chain.from_iterable(
			[iter_all_meta(part, meta) for meta in SCORE_META.values()]
			)
		)))
	)

	pivotal_points = sorted(list(set(pivotal_points)))
	return pivotal_points



if __name__ == '__main__':
    import partitura
    import os

    dirname = os.path.dirname(__file__)
    par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
    my_kern_score = os.path.join(par(par(dirname)), "samples", "xml", "k080-04.musicxml")
    part = partitura.load_musicxml(my_kern_score)[0]

    print(get_score_meta(part, 4))