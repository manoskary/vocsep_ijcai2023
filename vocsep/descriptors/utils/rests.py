import numpy as np

def hammer(note_array, bar_in_beats):
	'''Get Onset vertical chords followed by a rest.'''    
	hammer_pos = list()
	max_onset = np.max(note_array["onset_beat"])
	length = max_onset + np.max(note_array[np.where(note_array["onset_beat"] == max_onset)[0]]["duration_beat"]) - bar_in_beats
	step = 1 
	window_size = bar_in_beats
	vl_beat_pos = list()
	for index in range(0, int(length)+1, window_size): 
		chords = np.intersect1d(note_array[ np.where( note_array["onset_beat"] < index + window_size )], note_array[ np.where( note_array["onset_beat"] >= index )]) 
		if chords.size > 1:
			if np.all(chords['onset_beat']%1 == 0):
				on = np.max(chords["onset_beat"])
				if on%bar_in_beats < bar_in_beats - 1:
					hammer_pos.append(on)
	return hammer_pos


def rest_after_onset(note_array, bar_in_beats):
	'''Get Onset of Rest at the last beat.'''    
	rests = list()
	max_onset = np.max(note_array["onset_beat"])
	length = max_onset + np.max(note_array[np.where(note_array["onset_beat"] == max_onset)[0]]["duration_beat"]) - bar_in_beats
	step = 1 
	window_size = bar_in_beats
	vl_beat_pos = list()
	for index in range(0, int(length)+1, window_size): 
		chords = np.intersect1d(note_array[ np.where( note_array["onset_beat"] < index + window_size )], note_array[ np.where( note_array["onset_beat"] >= index )]) 
		onsets = chords[np.where(chords['onset_beat']%1 == 0)]
		if chords.size > 1 and onsets.size >1:
			if np.max(chords["onset_beat"]) == np.max(onsets["onset_beat"]):
				on = onsets[np.argmax(onsets["onset_beat"])]
				if on["onset_beat"]%bar_in_beats < bar_in_beats-1 and on["duration_beat"] <= bar_in_beats-1:
					rests.append(on["onset_beat"])
	return rests	




if __name__ == "__main__":
	import partitura
	import os

	dirname = os.path.dirname(__file__)
	par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
	my_musicxml_file = os.path.join(par(par(dirname)), "samples", "xml", "mozart_piano_sonatas", "K280-1.musicxml")
	
	note_array = partitura.utils.ensure_notearray(partitura.load_musicxml(my_musicxml_file))
	bar_in_beats = 3
	print(list(map(lambda x: (int(x / bar_in_beats) + 1, x%bar_in_beats), rest_after_onset(note_array, bar_in_beats))))