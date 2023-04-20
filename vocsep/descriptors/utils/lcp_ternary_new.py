import numpy as np
import itertools
import random
import os, sys


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils import RollingHash


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
    """Returns the ternary encoding of a note_array in voices

    Parameters
    ----------
    note_array : structured array
        A partitura note arrray

    Returns
    -------
    ternary_array : numpy array
        The ternary encoding
    v : numpy array
        The voices, an array of voice x number of notes x onset and duration.
    
    """
    note_array = [(n["onset_beat"], n["duration_beat"], n["pitch"]) for n in note_array]
    chords = [list(item[1]) for item in itertools.groupby(sorted(note_array), key=lambda x: x[0])]
    voices = separate_chords(chords) 
#     ternary_dict = dict([(voice_num, encode_line(note_list)) for voice_num, note_list in voices.items()])
    v = np.array(list(voices.values()))
    ternary_array = np.array([encode_line(note_list) for note_list in voices.values()])
    return ternary_array, v



            
            
         
def length_increase(text, word_size, indices, max_occur, max_occur_indices):
    """Find the maximum length of words so that their occurance index doesn't decrease.

        Given a text and an initial word size alogn with positions of the candidate words with max occurance.
        we find the maximum length such that at least one of the prolongated words has max occurance.

    Parameters
    ----------
    text : str
        A given string in which we search reapeating substrings 
    word_size : int
        The size of the initial prefix
    indices : list()
        The position of all the prefix in the string
    max_occur : int
        The occurance of the prefix in the string
    max_occur_indices : list()
        The position of all candidates in the text, i.e. the different prefixes.

    Returns
    -------
    word_size-1: int
        The size of the maximal length of the subsequence that has a given prefix.
    """
    # Initiating Rolling Hashing arrays to optimize search.
    hasharray = RollingHash(text, word_size).selective_hasharray(indices)
    red_h = hasharray[np.nonzero(hasharray)]
    uniques, freq = np.unique(red_h, return_counts=True)
    if np.max(freq) >= max_occur:
        return length_increase(text, word_size+1, indices, np.max(freq), max_occur_indices)
    else:
        return word_size-1
    

def filter_overlap(indices, word_size):
    """ Filter overlapping prefixes in a Text.

    Parameters
    ----------
    indices : list(int)
        A list of the indices where prefixes start.

    word_size : int
        The size of the prefix.

    Returns
    -------
    filtered_indices : list()
        The list where indices are filters, we keep the ones that appeared first.
    """
    filtered_indices = list()
    indices = sorted(indices)
    j=1
    # To remove the overlapping indices we have to keep track of the indices we skip
    for i in range(len(indices)):
        if j == 1:
            j = 1
            cond = True
            while i+j+1 < len(indices) and cond:
                if indices[i] + word_size <= indices[i+j]:
                    filtered_indices.append(indices[i+j])
                    if indices[i] + word_size <= indices[i+j+1]:
                        cond = False
                    else:
                        j+1
                else:
                    j +=1
        else :
            j-=1
    return filtered_indices


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
    final_indices : list(list(int))
        The list of lists of grouped indices.
    word_size : int
        The maximal_word_size.
    """
    if len(max_occur_indices) == 0:
        raise ValueError("Something went wrong")
    elif len(max_occur_indices) == 1:
        print("Only one maximum occuring pattern found")
        return [text[max_occur_indices[0] : max_occur_indices[0]+word_size]], indices
    else:
        # we find their possible max length so that the max occurance doesn't decrease.
        # Ideally we want to find more than one max length
        word_size = length_increase(text, word_size, indices, 0, max_occur_indices) # Execute only when first word changes
        # Filter Indices for overlaps
        filtered_indices = filter_overlap(indices, word_size)
        # Build reduced hasharray
        hasharray = RollingHash(text, word_size).selective_hasharray(filtered_indices)
        red_h = hasharray[np.nonzero(hasharray)]
        uniques, freq = np.unique(red_h, return_counts=True)
        freq = freq[np.nonzero(freq>1)]
        uniques = uniques[np.nonzero(freq>1)]
        sort_ind = np.argsort(freq)[::-1]
        sorted_by_freq = uniques[sort_ind][:10]
        final_indices = [np.where(hasharray == h)[0].tolist() for h in sorted_by_freq]
        return final_indices, word_size



    
def pattern_searching(text, word_size=6):    
    """Find maximum length for a given prefix in string such that the occurance of the word doesn't decrease.

    Parameters
    ----------
    text : str
        A given string in which we search reapeating substrings 
    word_size : int
        An initial word size

    Returns
    -------
    final_indices : list(list(int))
        The list of lists of grouped indices.
    word_size : int
        The maximal_word_size.
    """
    # First filter the words that start with dash.
    index_list = [i for i in range(len(text) - word_size) if text[i]!="-"]
    # Build the selective hash array
    hasharray = RollingHash(text, word_size).selective_hasharray(index_list)
    # The searching occurs in two parts, first finding max occurance of small length substrings and then 
    # we find their possible max length so that the max occurance doesn't decrease.
    red_h = hasharray[np.nonzero(hasharray)]
    uniques, freq = np.unique(red_h, return_counts=True)
    freq = freq[np.nonzero(freq>2)]
    uniques = uniques[np.nonzero(freq>2)]
        # sort_ind = np.argwhere(freq == np.amax(freq)).flatten()
        # sorted_by_freq = uniques[sort_ind]
    sort_ind = np.argsort(freq)[::-1]
    sorted_by_freq = uniques[sort_ind]

    # We create a list with all found indices.
    selected_indices = [np.where(hasharray == h)[0].tolist() for h in sorted_by_freq]
    max_occur_indices = list(map(lambda x : x[0], selected_indices))
    selected_indices = sum(selected_indices, [])

    # We filter the final list for overlapping.
    return max_length_checking(text, word_size, max_occur_indices, selected_indices)

def lcp_to_onsets_single(note_array):
    enc, voc = encode_note_array(note_array)
    enc = ["".join(x) for x in enc]
    text = "".join(enc)
    patterns_indices, max_size = pattern_searching(text)
    patterns = [text[pa[0] : pa[0]+max_size] for pa in patterns_indices]
    print(patterns)
    hash_array = np.array([RollingHash(s, max_size).hasharray() for s in enc])
    ind = []
    for word in patterns:
        rows, cols = np.where(hash_array == hash(word))
        for i, v in enumerate(voc):
            if rows.any():
                on, dur = zip(*(v[cols[np.where(rows == i)[0]]]))
                ind += on
    return sorted(list(set(ind)))


def lcp_to_onsets(list_of_note_arrays):
    enc, voc = zip(*[encode_note_array(na) for na in note_array]) 
    enc = ["".join(x) for l in enc for x in l]
    text = "".join(enc)
    patterns_indices, max_size = pattern_searching(text)
    patterns = [text[pa[0] : pa[0]+max_size] for pa in patterns_indices]

    hash_array = np.array([RollingHash(s, max_size).hasharray() for s in enc])
    ind = dict(zip(list(range(len(voc))), [[] for _ in range(len(voc))]))
    for word in patterns:
        rows, cols = np.where(hash_array == hash(word))
        for i, v in enumerate(voc):
            on, dur = zip(*(v[cols[np.where(rows == i)[0]]]))
            ind[i] += on
    return sorted(list(set(sum(list(ind.values())))))


if __name__ == '__main__':
    import partitura
    import os
    from matplotlib.pyplot import imshow

    dirname = os.path.dirname(__file__)
    par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
    my_musicxml_file = os.path.join(par(par(dirname)), "samples", "xml", "k080-04.musicxml")
    part = partitura.load_musicxml(my_musicxml_file)[0]
    if isinstance(part, list):
        note_array = [partitura.utils.ensure_notearray(p) for p in part] 
        enc, voc = zip(*[encode_note_array(na) for na in note_array]) 
        enc = ["".join(x) for l in enc for x in l] 
        lengths = [len(text) for text in enc]
    else:
        print("Single part")
        note_array = partitura.utils.ensure_notearray(part)
        print(lcp_to_onsets_single(note_array))
    # for text in enc:
    #     patterns_indices, max_size = pattern_searching(text)
    #     patterns = [text[pa[0] : pa[0]+max_size] for pa in patterns_indices]
    #     hash_array = np.array([RollingHash(s, max_size).hasharray() for s in enc])
    #     for word in patterns:
    #         rows, cols = np.where(hash_array == hash(word))
    #         for u in np.unique(cols):
    #             voices[u] += np.where(cols = u)[0] 



    #     print(slicing_and_correct_index(lengths, voices, np.array(ind[0])))
    #     break
    # Have to group patterns to bars
    
    