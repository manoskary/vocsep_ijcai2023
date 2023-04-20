import numpy as np
from scipy.signal import argrelextrema
import partitura as pt
from scipy import linalg as la
from struttura.descriptors.utils.harmonic_ssm import apply_ssm_analysis


def part_to_pianoroll_window_size(part):
	if isinstance(part, list):
		part = part[0]
		ts = [(ts.start.t, ts.end, (ts.beats, ts.beat_type)) for ts in part.iter_all(pt.score.TimeSignature)]
	else:
		ts = [(ts.start.t, ts.end, (ts.beats, ts.beat_type)) for ts in part.iter_all(pt.score.TimeSignature)]
	bar_size = list()
	for time_sig in ts:
		x = time_sig[2][0]
		beat = part.inv_beat_map(1.)
		bar_size.append(beat*x )
	if len(bar_size) == 0:
		raise ValueError("Part doesn't have a time signature")
	elif len(bar_size) == 1:
		return bar_size[0]
	else:
		return bar_size


def pianoroll_window_analysis(pianoroll, note_array, window_size, reg_jump=6):
	X = np.zeros((len(note_array), int(pianoroll.shape[1]/reg_jump)))
	for i, note in enumerate(note_array):
		X[i] = np.count_nonzero(pianoroll[note["onset_div"]:note["onset_div"]+window_size, i*reg_jump:(i+2)*reg_jump])
	return X


def pca(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T


def get_pssm(part, note_array, window_size=1):
	"""Perform the pianoroll analysis and output register SSM.

	Parameters
	----------
	note_array : structured array
		The note array ordered on onsets


	Returns:
	--------
	peaks : numpy array
		A list of the beats where SSM segments are detected centered.
	"""
	features = ["pssm_novelty_curve", "dssm_pca1", "dssm_pca2"]
	pianoroll = pt.utils.compute_pianoroll(part).todense().T
	window = int(window_size * part.inv_beat_map(max(np.unique(note_array["ts_beats"]))))
	X = pianoroll_window_analysis(pianoroll, note_array, window)
	X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-6)
	nov = apply_ssm_analysis(X_scaled, gaussian_filter_size=2*window+1)
	X_pca = pca(X_scaled)
	d_array = np.hstack((np.expand_dims(nov, axis=1), X_pca))
	return d_array, features



def vpeak(X, hpeaks_dict):
	"""Builds a dictionary where keys are the window index and values are octave index of peaks in note density.
	
	Parameters
	----------
	X : numpy array
		A sparse array where entrys are the number of black pixels for the pianoroll
	hpeaks_dict : dictionary
		a dictionary of horrizontal peaks, we use this to filter X and reduce computation.
	Returns
	-------
	vpeaks : dictionary
		A dictionary where keys are the number of the octave and values are lists of peaks in note density.
		v stands for vertically, we can think this as a vertically sliding window on the pianoroll representation 
		which measures for different time positions on which octave black pixels peaks.
	"""
	vpeaks = dict()
	# Now find the critical bar where peaks are located.
	potential_peaks = list(set([el for v in hpeaks_dict.values() for el in v]))
	for i in potential_peaks:
		x = argrelextrema(X[:, i], np.greater)[0]
		if any(x):
			vpeaks[i] = x.tolist()
	return vpeaks

def hpeak(X):
	"""Builds a dictionary where keys are the number of the octave and values are lists of peaks in note density.
	
	Parameters
	----------
	X : numpy array
		A sparse array where entrys are the number of black pixels for the pianoroll
	Returns
	-------
	hpeaks : dictionary
		A dictionary where keys are the number of the octave and values are lists of peaks in note density.
		H stands for horizontally, we can think this as a horrizontally sliding window on the pianoroll representation 
		which measures for different height positions where the number of black pixels peaks.
	"""
	hpeaks = dict()
	for i in range(X.shape[0]):
		x = argrelextrema(X[i], np.greater)[0]
		if any(x):
			hpeaks[i] = x.tolist()
	return hpeaks

def find_init_peak(X):
	"""Builds a list of octave index where peaks occur in note density for the first appearance of notes.
	
	Parameters
	----------
	X : numpy array
		A sparse array where entrys are the number of black pixels for the pianoroll
	Returns
	-------
	vpeaks : list
		A list with octave indexes where peaks in note density occur, for the beginning of the score.
	i-1 : int
		The index of the window where the score actually begins (typically 0 but it might have some frames of silence)
	"""
	i=0
	vpeaks = list()
	while not any(vpeaks):
		vpeaks = argrelextrema(X[:, i], np.greater)[0]
		i+=1
	return vpeaks, i-1

def get_peak_finding(part):
	"""Finds the number of the bar where there are potential Register changes.
	
	Parameters
	----------
	pianoroll : numpy_array
		Each pianoroll entry contains 3 values an note start a note end and a midi pitch
	bar_in_pianoroll : int
		This value represents the how many of the smallest pianoroll values are contained in a bar.
	Returns
	-------
	peaks : list
		The bar number where potential peaks occur.
	"""
	if isinstance(part, list):
		pianoroll = np.array([(n.start.t, n.end.t, n.midi_pitch) for p in part for n in p.notes])		
		pianoroll = pianoroll[pianoroll[:, 0].argsort()]
		part = part[0]
	else :
		pianoroll = np.array([(n.start.t, n.end.t, n.midi_pitch) for n in part.notes])
	bar_size = part_to_pianoroll_window_size(part)
	bar_in_beats = part.beat_map(bar_size)
	window_size = int(bar_size/2)
	X = pianoroll_window_analysis(pianoroll, window_size)
	peaks = list()
	hpeaks_dict = hpeak(X) 
	# first filter the range of the octaves which contain crital peaks that is the keys of hpeaks.
	vpeaks_dict = vpeak(X, hpeaks_dict) 
	init = find_init_peak(X)[0]
	for key in vpeaks_dict.keys():
		for octave in vpeaks_dict[key]:
			if octave not in init and octave-1 not in init and octave+1 not in init:
				init = vpeaks_dict[key]
				peaks.append(int(key/2)+1)
		if len(vpeaks_dict[key]) != len(init):
			init = vpeaks_dict[key]
			peaks.append(int(key/2)+1)
	return sorted(list(set(map(lambda x: x*bar_in_beats, peaks))))



	