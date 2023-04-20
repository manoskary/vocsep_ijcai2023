from basismixer.performance_codec import tempo_by_average, tempo_by_derivative, to_matched_score
from sklearn.utils.random import sample_without_replacement
import numpy as np

def performance_analysis(note_array, performance_array, bar_in_beats, samples_per_window=10):
	max_onset = np.max(note_array["onset_beat"])
	length = max_onset + np.max(note_array[np.where(note_array["onset_beat"] == max_onset)[0]]["duration_beat"]) - bar_in_beats
	step = 1 
	window_size = bar_in_beats
	ws2 = (bar_in_beats)/2
	vl_beat_pos = list()
	X = np.zeros((int(length), 4, 4, samples_per_window))

	for index in range(0, int(length), step): 
		temp1 = np.intersect1d(np.where( note_array["onset_beat"] < index + window_size ), np.where( note_array["onset_beat"] >= index )) 
		temp2 = np.intersect1d(np.where( note_array["onset_beat"] < index), np.where( note_array["onset_beat"]+note_array["duration_beat"] >= index ))
		tnotes = np.union1d(temp1, temp2)
		for i in range(4):
			sample_indices = sample_without_replacement(len(tnotes), samples_per_window)
			# compute expressive features from performance array
			performance = performance_array[tnotes[sample_indices]]
			beat_period, velocity, timing, articulation_log = list(zip(*performance))
			# Add to output array
			X[index][i][0] = np.array(beat_period)
			X[index][i][1] = np.array(velocity)
			X[index][i][2] = np.array(timing)
			X[index][i][3] = np.array(articulation_log)
	return X



def tempo_curve(ppart, alignment, spart):
	note_array, _ = to_matched_score(spart, ppart, alignment)
	tempo_cu = tempo_by_average(note_array["onset"], note_array["p_onset"], note_array["duration"], note_array["p_duration"])
	return tempo_cu



if __name__ == "__main__":
	import partitura
	import os
	import matplotlib.pyplot as plt

	dirname = os.path.dirname(__file__)
	par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
	my_musicxml_file = os.path.join(par(par(dirname)), "samples", "match", "mozart_piano_sonatas", "K279-1.match")
	ppart, alignment, spart = partitura.load_match(my_musicxml_file, create_part=True)
	tempo_cu = tempo_curve(ppart, alignment, spart)
	plt.figure(figsize=(20, 8))
	plt.plot(tempo_cu, ':', label='Tempo Curve')
	plt.legend()
	plt.grid()
	plt.show()