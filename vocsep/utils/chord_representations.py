from itertools import combinations


def chord_to_intervalVector(midi_pitches, return_pc_class=False):
    '''Given a chord it calculates the Interval Vector.


    Parameters
    ----------
    midi_pitches : list(int)
        The midi_pitches, is a list of integers < 128.

    Returns
    -------
    intervalVector : list(int)
        The interval Vector is a list of six integer values.
    '''
    intervalVector = [0, 0, 0, 0, 0, 0]
    PC = set([mp%12 for mp in midi_pitches])
    for p1, p2 in combinations(PC, 2):
        interval = int(abs(p1 - p2))
        if interval <= 6:
            index = interval
        else:
            index = 12 - interval
        if index != 0:
            index = index-1
            intervalVector[index] += 1
    if return_pc_class:
        return intervalVector, list(PC)
    else:
        return intervalVector
