import numpy as np
from struttura.descriptors.utils.note_density import note_array2chords
from struttura.utils import chord_to_intervalVector


def check_chromatic(noteA, noteB):
	if noteB["onset_beat"]+noteB["duration_beat"] == noteA["onset_beat"]:
		if noteA["pitch"] == noteB["pitch"]+1:
			return True
		elif noteA["pitch"] == noteB["pitch"]:
			return True
	return False


def chromatic_upward_bass(note_array):
	"""Detect Chromatic Upward movements on bass voice.

	Given the Cello part find chromatic upward movements where notes mayu occur consecutive times.
	"""
	chromatic_bass_beat_pos = list()
	for index in range(3, len(note_array)):
		r = 1
		cnotes = [note_array[index]["pitch"]]
		while check_chromatic(note_array[index-r+1], note_array[index-r]):
			if index-r == 0 or r>8:
				break
			cnotes.append(note_array[index-r]["pitch"])
			r += 1
		if len(set(cnotes)) >= 3:
			chromatic_bass_beat_pos.append(note_array[index]["onset_beat"])
	return chromatic_bass_beat_pos



def diminshed7(note_array):

	dimished_beat_pos = list()
	chords = note_array2chords(note_array)
	for c in chords:
		on, dur, pi = zip(*c)
		if chord_to_intervalVector(list(pi)) == [0, 0, 4, 0, 0, 2]:
			dimished_beat_pos.append(on[0] + max(dur))
	return dimished_beat_pos


def ped_note(list_of_na, bar_in_beats):
	"""Detect a note that lasts more than 1 measure.

	Given a single part note array.
	"""
	pedal_notes_beat_pos = list()
	for note_array in list_of_na:
		indices = np.where(note_array["duration_beat"] >= bar_in_beats)[0]
		for index in indices:
			pedal_notes_beat_pos.append(note_array[index]["onset_beat"]+note_array[index]["duration_beat"])
	return sorted(list(set(pedal_notes_beat_pos)))



def full_rest_and_unison(note_array):
	full_rest_beat_pos = list()
	unison_beat_pos = list()
	chords = note_array2chords(note_array)

	for index, c in enumerate(chords[1:], 1):
		onset_next = c[0][0]
		onset_previous = chords[index-1][0][0]
		if len(chords[index-1])>1:
			pitch = chords[index-1][0][2]%12
			rest_bool = [1 if n[1]+onset_previous < onset_next else 0 for n in chords[index-1]  ]
			uni_bool = [1 if n[2]%12 == pitch else 0 for n in chords[index-1] ]
			if rest_bool.count(1) == len(chords[index-1]):
				full_rest_beat_pos.append(onset_next)
			if uni_bool.count(1) == len(chords[index-1]):
				unison_beat_pos.append(onset_previous)
	return full_rest_beat_pos, unison_beat_pos



def key_estimation(note_array, bar_in_beats):

	total_duration = np.max(note_array["onset_beat"])
	index_range = np.int(total_duration + 1 - bar_in_beats*4)
	keys = np.empty(index_range, dtype='|S2')
	for index in range(index_range):
		# break array to every two measure
		start_idx = np.min(np.where(note_array["onset_beat"] >= index)[0])
		end_idx = np.max(np.where(note_array["onset_beat"] <= index + 4*bar_in_beats)[0])
		keys[index] = partitura.musicanalysis.estimate_key(note_array[start_idx:end_idx], method='krumhansl')
	# Filter tonalities by occurence
	unique_elements, frequency = np.unique(keys, return_counts=True)
	sorted_indexes = np.argsort(frequency)[::-1]
	sorted_by_freq = unique_elements[sorted_indexes]
	if sorted_by_freq.shape[0] > 1:
		main_tonality = sorted_by_freq[0]
		aux_tonality = sorted_by_freq[1]
	else:
		main_tonality = None
		aux_tonality = None
	tonality_changes_beat_position = list()
	for i in range(bar_in_beats, keys.shape[0] - bar_in_beats):
		if np.all(keys[i:i+int(bar_in_beats)] == keys[i]) and keys[i] != main_tonality:
			tonality_changes_beat_position.append(i+2*bar_in_beats)
			main_tonality = keys[i]
	return tonality_changes_beat_position


def rhythm_break(list_of_na):
	"""Detect 15 note consecutive rhythm breaks.

	Given a single part note array.
	"""
	rhythm_break_beat_pos = list()
	for note_array in list_of_na:
		for index in range(note_array.shape[0]-1, 10):
			cons = np.all(note_array[index-9:index]["duration_beat"] == note_array[index]["duration"])
			diff_on = np.all(note_array[index-9:index]["onset_beat"] != note_array[index]["onset_beat"])
			if cons and diff_on and note_array[index+1]["duration"] != note_array[index]["duration"]:
				rhythm_break_beat_pos.append(note_array[index+1]["onset_beat"]+note_array[index+1]["duration"])
			else:
				pass
	return sorted(list(set(rhythm_break_beat_pos)))


def triple_hammer(note_array):
	"""Detect triple hammer blow three same notes.

	Given a single part note array.
	"""
	hammer_beat_pos = list()
	for note_array in list_of_na:
		for index in range(4, len(note_array)-1):
			i = 0
			j = 1
			cond = True
			while cond and i<4 and index-j>=0:
				if note_array[index - i]["onset_beat"] == note_array[index-j]["onset_beat"] + note_array[index-j]["duration_beat"] and note_array[index - i]["duration_beat"] == note_array[index-j]["duration_beat"]:
					i+=1
					j+=1
				elif note_array[index - i]["onset_beat"] == note_array[index-j]["onset_beat"]:
					j+=1
				else:
					cond=False
			if cond and note_array[index]["onset_beat"] + note_array[index]["duration_beat"] < note_array[index+1]["onset_beat"]:
				hammer_beat_pos.append(note_array[index]["onset_beat"]+ note_array[index]["duration_beat"])
	return sorted(list(set(hammer_beat_pos)))


def triple_onset_rest(note_array, bar_in_beats):
	last_onset_pos = list()
	if bar_in_beats%2 ==0:
		indlen = 4
	else:
		indlen = 3
	for index in range(indlen, len(note_array)-1):
		i = 0
		j = 1
		cond = True
		while cond and i<indlen and index-j>=0:
			last_onset = note_array[index - i]["onset_beat"]
			last_duration = note_array[index - i]["duration_beat"]
			previous_onset = note_array[index-j]["onset_beat"]
			previous_duration = note_array[index-j]["duration_beat"]
			if last_onset%1 == 0 and previous_onset%1 == 0:
				if last_onset == previous_onset + previous_duration and last_duration == last_duration:
					i+=1
					j+=1
				elif note_array[index - i]["onset_beat"] == note_array[index-j]["onset_beat"]:
					j+=1
				else:
					cond=False
			else:
				cond=False
		if cond and note_array[index]["onset_beat"] + note_array[index]["duration_beat"] < note_array[index+1]["onset_beat"]:
			last_onset_pos.append(note_array[index]["onset_beat"])
	return last_onset_pos




def authentic_cadences(note_array, cello_note_array):
	auth_beat_pos = list()
	onsets_of_interest = list()
	for index in range(1, len(cello_note_array)):
		piA, piB = cello_note_array[index-1 : index+1]["pitch"]
		if (piB - piA)%12 == 5 and cello_note_array[index]["onset_beat"]%4 == 0:
			onsets_of_interest.append(cello_note_array[index]["onset_beat"])

	for onset in onsets_of_interest:
		x = np.take(note_array["pitch"], np.where(note_array["onset_beat"] == onset)).tolist()[0]
		x = chord_to_intervalVector(x)
		if x == [0, 0, 1, 1, 1, 0]:
			durations_of_interest = np.take(note_array["duration_beat"], np.where(note_array["onset_beat"] == onset)).tolist()[0]
			auth_beat_pos.append(onset + min(durations_of_interest))
	auth_beat_pos = sorted(list(set(auth_beat_pos)))
	return auth_beat_pos

def piece_end(note_array):
	x = note_array["onset_beat"]
	return [np.max(x)]



if __name__ == "__main__":
	import partitura
	my_musicxml_file = partitura.EXAMPLE_MUSICXML

	note_array = partitura.utils.ensure_notearray(partitura.load_musicxml(my_musicxml_file))
	# print(note_array)
	# horrizontal, vertical = get_note_density(note_array, 2, 1)
	# print(vertical, horrizontal)
	
	import os
	dirname = os.path.dirname(__file__)
	par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
	my_musicxml_file = os.path.join(par(par(dirname)), "samples", "xml", "mozart_piano_sonatas", "K331-1.musicxml")

	parts = partitura.load_musicxml(my_musicxml_file)
	# list_of_na = [partitura.utils.ensure_notearray(part) for part in parts]
	note_array = partitura.utils.ensure_notearray(parts)
	# cello_note_array = partitura.utils.ensure_notearray(parts[-1])
	
	print(triple_onset_rest(note_array, 6))
	# print(sorted(set(map(lambda x : int(x/4)+1, d))))