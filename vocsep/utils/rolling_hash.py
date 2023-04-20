import numpy as np

class RollingHash:
    '''A rolling hash for a window of constant length into a text,
        both specified at construction.
    '''

    def __init__(self, text, size_word):
        '''Set up a rolling hash for a window of size_word into text.'''
        self.text = text
        self.size_word = size_word
        if len(text) < size_word:
            self.hash = None
            return
        # rk = hash(text[:size_word])
        self.hash = np.zeros((len(text)+1 - size_word))
        # self.hash[0] = rk
        self.pos = -1
        self.z = 0
        self.window_start = 0 
        self.window_end = size_word
    
    def hasharray(self):
        x = self.hash
        if len(self.text) == self.size_word:
            return x[0]
        for i in range(len(self.text) - self.size_word + 1):
            if self.window_end < len(self.text):
                self.window_start += 1             
                self.window_end += 1
                x[self.window_start] = hash(self.text[self.window_start:self.window_end])
        return x

    def selective_hasharray(self, indices):
        x = self.hash
        if len(self.text) == self.size_word:
            return x[0]
        for i in indices:
            self.window_start = i
            self.window_end = i+self.size_word
            if self.window_end < len(self.text):                
                x[self.window_start] = hash(self.text[self.window_start:self.window_end])
        return x

    def window_text(self):
        '''Return current window text.'''
        return self.text[self.window_start:self.window_end]