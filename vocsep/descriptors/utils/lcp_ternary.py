import numpy as np
import itertools
import random
from struttura.utils import RollingHash


def ternary_delta(x, y):
    n = x-y
    if n==0:
        return "O"
    if n>0:
        return "N"
    if n<0:
        return "P"

    
def voice_filtering(rt, voices, temps, cindex, len_chords):
    dt = np.dtype(int) 
    if not voices:
        temps[0] = rt
        voices[0] = np.zeros((len_chords, 2))
        voices[0][0] = rt[0], rt[1] # here to apply exception on duration
    else:
        bool_v = True
        candidates = [(note[0], note[1], abs(note[2] - rt[2]), key) for key, note in temps.items() if note[0]+note[1]==rt[0]]
        if candidates:
            key = min(candidates, key=lambda x:x[2])[3]
            temps[key] = rt
            voices[key][cindex] = rt[0], rt[1]
            bool_v = False
        if bool_v:
            epsilon = list()
            for key, temp in temps.items():
                if temp[0]+temp[1] < rt[0]:
                    epsilon.append((key, rt[0] - (temp[0]+temp[1])))
                    bool_v = False
            if not bool_v:
                key = min(epsilon, key = lambda t: t[1])[0]
                temps[key] = rt
                voices[key][cindex] = rt[0], rt[1]
            else:
                key = max(voices.keys())+1
                temps[key] = rt
                voices[key] = np.zeros((len_chords, 2))
                voices[key][cindex] = rt[0], rt[1]
    return voices, temps
    
    
def separate_chords(chords):
    voices = dict()
    temps = dict()
    for cindex, c in enumerate(chords):
        onset, dur, pitch = zip(*c)
        result = all(elem == dur[0] for elem in dur)
        rt = onset[0], dur[0], pitch[0]
        if result:
            voices, temps = voice_filtering(rt, voices, temps, cindex, len(chords)) 
        else:
            note_lists = [list(item[1]) for item in itertools.groupby(sorted(c), key=lambda x: x[1])]
            for l in note_lists:
                rt = l[0][0], l[0][1], l[-1][2]
                voices, temps = voice_filtering(rt, voices, temps, cindex, len(chords))
    return voices
    
def encode_line(note_list):
    """Encodes ternary rhythm in a single melodic/rhythmic line.
     
    Parameters
    ----------
    note_list : list()
        A list of note tuples (onset, dur). The list must comply with some constrains such as:
        non overlapping notes, 
        
    Returns
    -------
    ternary : numpy array()
        The ternary encoding of the note_list.
    """
    ternary = np.full((note_list.shape[0]), "-")    
    if note_list.shape[0] == 0:
        raise ValueError("The list is empty")
    if note_list.shape[0] == 1:
        print(note_list)
        raise ValueError("Insufficient elements in list")
    else:
        ternary[0] = "P" if note_list[0][1] != 0 else "-"
        for index in range(1, note_list.shape[0]):
            # check if notes are consecutive
            if note_list[index][0] != 0 and note_list[index][1] != 0:
                
                
                # Let's add a constrain to filter syncopations
                i = 1
                bv = False
                previous_onset = note_list[index-i][0]
                previous_duration = note_list[index-i][1]
                if previous_onset == 0 and previous_duration == 0:
                    for i in range(1, 5):
                        previous_onset = note_list[index-i][0]
                        previous_duration = note_list[index-i][1]                    
                        if previous_duration !=0:
                            break
                        if index-i == 0:
                            break
                    i+1
                    
                if previous_onset + previous_duration == note_list[index][0]:
                    ternary[index] = ternary_delta(previous_duration, note_list[index][1])
                # first check if the notes are consecutive or there is a rest between them. 
                elif previous_onset + previous_duration < note_list[index][0]:
                    ternary[index] = "P" # Syncopations are marked as "P" is not necessarily bad.
                # if notes are overlapping then the note_list is wrong
                else:
#                     print("Overlapping notes in Note list.")
                    pass
        return ternary
    
    
def encode_note_array(note_array):
    note_array = [(n["onset_beat"], n["duration_beat"], n["pitch"]) for n in note_array]
    chords = [list(item[1]) for item in itertools.groupby(sorted(note_array), key=lambda x: x[0])]
    voices = separate_chords(chords) 
#     ternary_dict = dict([(voice_num, encode_line(note_list)) for voice_num, note_list in voices.items()])
    ternary_array = np.array([encode_line(note_list) for note_list in voices.values()])
    return ternary_array



            
            
         
def length_increase(text, word_size, indices, max_occur, max_occur_indices):
    """Find maximum length for a given prefix in string such that the occurance of the word doesn't decrease.

    Parameters
    ----------
    text : str
        A given string in which we search reapeating substrings 
    word_size : int
        The size of the initial prefix
    indices : int
        The position of the prefix in the string
    max_occur : int
        The occurance of the prefix in the string

    Returns
    -------
    word_size-1: int
        The size of the maximal length of the subsequence that has a given prefix.
    """
    # Initiating Rolling Hashing arrays to optimize search.
    hasharray = RollingHash(text, word_size).selective_hasharray(indices)
    for index in max_occur_indices:
        word_hash = RollingHash(text[index : index + word_size], word_size).hasharray()
        word_occur = np.count_nonzero(hasharray == word_hash)
        if word_occur >= max_occur:
            return length_increase(text, word_size+1, indices, word_occur, max_occur_indices)
        else:
            pass
    return word_size-1
    

def max_length_checking(text, word_size, max_occur_indices, indices):
    """Find maximum length for a given prefix in string such that the occurance of the word doesn't decrease.

    Parameters
    ----------
    text : str
        A given string in which we search reapeating substrings 
    word_size : int
        The size of the initial prefix
    max_occur_indices : list(int)
        The initial indices where max occurance was recorded. This is a subset of the indices.
        We use that to reduce the multiplicity search.
    indices : list(int)
        The list of indices for which we need to create a hash in the hash array

    Returns
    -------
    text[indices[0] : indices[0]+word_size] : string
        The longest substring in the text which has the maximum occurance. This is computed recursively.
    """
    if len(indices) == 0:
        raise ValueError("Something went wrong")
    elif len(indices) == 1:
        return [text[indices[0] : indices[0]+word_size]], indices
    else:
        word_size = length_increase(text, word_size, indices, 0, max_occur_indices) # Execute only when first word changes
        hasharray = RollingHash(text, word_size).selective_hasharray(indices)        
        occur_dict = dict()

        for index in max_occur_indices:
            if index + word_size < len(text):
                word_hash = RollingHash(text[index : index + word_size], word_size).hasharray()
                word_occur = np.count_nonzero(hasharray == word_hash)
                occur_dict[index] = word_occur
        max_occur = max(occur_dict.values())
        nindices = [i for i, o in occur_dict.items() if o == max_occur]        
        if sorted(indices) == sorted(nindices):
            return [text[index : index+word_size] for index in nindices], nindices
        else:
            return max_length_checking(text, word_size, nindices, indices)

    
def pattern_searching(text, word_size):    
    """Find maximum length for a given prefix in string such that the occurance of the word doesn't decrease.

    Parameters
    ----------
    text : str
        A given string in which we search reapeating substrings 
    word_size : int

    Returns
    -------
    max_length_checking(text, word_size, max_occur_indices) : str
        The longest sequence with maximum occurance (Not to be confused with lcs).
    """
    searched_ind_length = 0
    max_occur_indices = list()
    selected_indices = list()
    random_index = 0
    max_occur = 1
    word_found = False
    hasharray = RollingHash(text, word_size).hasharray()
    index_list = [i for i in range(len(text) - word_size) if text[i]!="-"]
    # The searching occurs in two parts, first finding max occurance of small length substrings and then 
    # we find their possible max length so that the max occurance doesn't decrease.
    while any(index_list):        
        random_index = random.choice(index_list)
        word = text[random_index : random_index+word_size]            
        word_hash = RollingHash(word, word_size).hasharray()
        word_occur = np.count_nonzero(hasharray == word_hash)
        current_ind_pos = sorted(np.where(hasharray == word_hash)[0].tolist())
        searched_ind_length += len(current_ind_pos)
        for pos, j in enumerate(current_ind_pos):
            try:
                index_list.remove(j)
            except:
                pass
            # Filter overlapping indices
            if pos > 0:
                if current_ind_pos[pos] + word_size > current_ind_pos[pos-1]:
                    word_occur -= 1
        if max_occur == word_occur:
            max_occur_indices.append(random_index)
            selected_indices += current_ind_pos
        elif max_occur < word_occur:
            max_occur_indices.append(random_index)
            selected_indices += current_ind_pos
            max_occur = word_occur 
            index = random_index
        else:   
            pass
        word_found = True if searched_ind_length >= len(text) - word_size else False  

    # We filter the final list for overlapping.
    return max_length_checking(text, word_size, max_occur_indices, selected_indices)




                

if __name__ == '__main__':
    import partitura
    import os
    from matplotlib.pyplot import imshow

    dirname = os.path.dirname(__file__)
    par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
    my_musicxml_file = os.path.join(par(par(dirname)), "samples", "xml", "Prelude_in_C_minor_-_BWV_999_-_Bach.musicxml")
    part = partitura.load_musicxml(my_musicxml_file)
    # list_of_na = [partitura.utils.ensure_notearray(part) for part in parts]
    note_array = partitura.utils.ensure_notearray(part)
    
    string = ""
    for el in encode_note_array(note_array)[1]:
        string +=el
    text = string
    word_size = 7
    print(pattern_searching(text, word_size))
    # Have to group patterns to bars
    
    