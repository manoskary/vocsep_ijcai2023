from itertools import groupby
import numpy as np
from scipy.signal import argrelextrema
from vocsep.descriptors.utils.harmonic_ssm import apply_ssm_analysis


def note_array2chords(note_array):
    '''Group note_array list by time, effectively to chords.
    
    Parameters
    ----------
    note_array : array(N, 5)
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
        
    Returns
    -------
    chords : list(list(tuples))
    '''
    note_array = [(n["onset_beat"], n["duration_beat"], n["pitch"]) for n in note_array]
    chords = [list(item[1]) for item in groupby(sorted(note_array), key=lambda x: x[0])]
    for i in range(1, len(chords)):
        temp = []
        # 
        for index, chordTriplePrev in enumerate(chords[i-1]):
            for chordTripleNext in chords[i]:
                if chordTripleNext[0] < chordTriplePrev[1]+chordTriplePrev[0] and (chordTripleNext[0], chordTriplePrev[1], chordTriplePrev[2]) not in temp:            
                    temp.append((chordTripleNext[0], chordTriplePrev[1], chordTriplePrev[2]))                 
                    chords[i-1][index] = (chordTriplePrev[0], chordTripleNext[0] - chordTriplePrev[0], chordTriplePrev[2])
        chords[i] = chords[i] + temp
    return chords


def chord_cardinalites(chords):
	chords_card = np.zeros(len(chords), dtype=[('onset', 'f4'), ('note_end', 'f4'), ('cardinality', 'i4')])
	for index, clist in enumerate(chords):
		on, dur, pitch = zip(*clist)
		chords_card[index] = (min(on), min(dur)+min(on), len(pitch))
	return chords_card


def get_note_density(note_array, window_size):
	"""Vertical and Horrizontal Normalized Note Density.
	
	Parameters
	----------
	note_array : structured array
		A structured array containing onsets, durations and pitch of notes extracted with partitura from a xml score.
	window_size : int
		Usually the bar size relative to beats

	Returns
	-------
	horrizontal : numpy array
		An array containing the normalized note density of different note onsets
	vertical : numpy array

	Examples
	--------
	import partitura
	my_musicxml_file = partitura.EXAMPLE_MUSICXML

	note_array = partitura.musicxml_to_notearray(my_musicxml_file)
	print(note_array)
	horrizontal, vertical = get_note_density(note_array, 2, 1)
	print(vertical, horrizontal)
	"""

	cardinalities = chord_cardinalites(note_array2chords(note_array))
	length = int((cardinalities[-1]["note_end"] - window_size))

	vertical = np.zeros(length)
	horrizontal = np.zeros(length)
	for index, window in enumerate(range(0, length, window_size)):
		temp1 = np.intersect1d(cardinalities[ np.where( cardinalities["onset"] < window + window_size )], cardinalities[ np.where( cardinalities["onset"] >= window )])
		temp2 = np.intersect1d(cardinalities[ np.where( cardinalities["onset"] < window)], cardinalities[ np.where( cardinalities["note_end"] >= window )])
		number_of_dif_onsets = len(list(map(np.unique, temp1["onset"]))) + len(list(map(np.unique, temp2["onset"])))
		horrizontal[index] = (len(temp1) +len(temp2))
		vertical[index] = (sum(temp1["cardinality"]) + sum(temp2["cardinality"]))/number_of_dif_onsets if number_of_dif_onsets !=0 else 0 
	return horrizontal, vertical


def get_dssm(part, note_array, window_size=1, median_filter_size = (3, 21)):
	"""Perform the Density analysis and output density SSM.

	Parameters
	----------
	part : partitura.score.Part
		Dummy attribute the partitura part.
	note_array : structured array
		The note array ordered on onsets.
	bar_in_beats : int
		The duration of a bar in beats relevant to the time signature.

	Returns:
	--------
	d_array : np.darray
		The density array per note in the note array.
	"""
	window = max(np.unique(note_array["ts_beats"]))*window_size
	h, v = get_note_density(note_array, window)
	features = ["dssm_novelty_curve", "normalize_horizontal_density", "normalize_vertical_density", "horizontal_density", "vertical_density"]
	X = np.vstack((h, v)).T
	row_sums = X.sum(axis=1)
	X = X / (row_sums[:, np.newaxis] + 1e-6)
	nov = apply_ssm_analysis(X, gaussian_filter_size=2*window+1, median_filter_size = median_filter_size)
	d_array = np.zeros((len(note_array), len(features)))
	for i, value in enumerate(nov):
		w_start, w_end = i*window, (i+1)*window
		d_array[np.where((note_array["onset_beat"] >= w_start) & (note_array["onset_beat"] < w_end))] = [value, X[i][0], X[i][1], h[i], v[i]]
	return d_array, features


def get_bars_from_density(note_array, window_size):
	"""From Note Density get bar number where Horrizontal density is locally min and Vertical Density is locally max.
	
	Parameters
	----------
	note_array : structured array
		A structured array containing onsets, durations and pitch of notes extracted with partitura from a xml score.
	window_size : int
		Usually the bar size relative to beats
	step : int
		Usually a beat.

	Returns
	-------
	horrizontal : numpy array
		An array containing the normalized note density of different note onsets
	vertical : numpy array

	Examples
	--------
	import partitura
	my_musicxml_file = partitura.EXAMPLE_MUSICXML

	note_array = partitura.musicxml_to_notearray(my_musicxml_file)
	print(note_array)
	horrizontal, vertical = get_note_density(note_array, 2, 1)
	print(vertical, horrizontal)
	"""
	horrizontal, vertical = get_note_density(note_array, window_size)
	xn = argrelextrema(horrizontal, np.less)[0].tolist()
	yn = argrelextrema(vertical, np.greater)[0].tolist()
	x = list()
	y = list()
	# for n in range(-1, 2):
	# 	print(n)
	# 	x += list(map(lambda i : i+n if i>n else 0, xn))
	# 	y += list(map(lambda i : i+n if i>n else 0, yn))
	# print(xn, yn)
	pivot_bars = np.intersect1d(xn, yn).tolist()
	return list(set(pivot_bars))





if __name__ == "__main__":
	import partitura
	my_musicxml_file = partitura.EXAMPLE_MUSICXML

	note_array = partitura.utils.ensure_notearray(partitura.load_musicxml(my_musicxml_file))
	# print(note_array)
	# horrizontal, vertical = get_note_density(note_array, 2, 1)
	# print(vertical, horrizontal)
	
	import os
	import matplotlib.pyplot as plt
	dirname = os.path.dirname(__file__)
	par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
	my_musicxml_file = os.path.join(par(par(dirname)), "samples", "xml", "mozart_piano_sonatas", "K279-1.musicxml")
	
	part = partitura.load_musicxml(my_musicxml_file)
	note_array = partitura.utils.ensure_notearray(part)
	d_array = get_dssm(part, note_array)
