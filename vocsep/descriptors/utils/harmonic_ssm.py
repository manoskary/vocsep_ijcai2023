from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import ndimage
from scipy.signal import argrelextrema
from vocsep.utils import chord_to_intervalVector


def intervalic_analysis(note_array, window_size=1):
	"""
	Note array 2 interval vectors with moving window.

	Parameters
	----------
	note_array : structured array.
		Partitura Part note array.
	window_size: float
		Expressed in portion of the time signature nominator.
		i.e. ts 4/4 and window_size=0.5 means a window of 2 beats.

	Returns
	-------
	X : np.array
		An array of interval vectors per window.
	"""
	X = np.zeros((len(note_array), 6))
	for i, note in enumerate(note_array):
		index = note["onset_beat"]
		window_length = window_size * note["ts_beats"]
		temp1 = np.intersect1d(note_array[ np.where( note_array["onset_beat"] < index + window_length )], note_array[ np.where( note_array["onset_beat"] >= index )])
		temp2 = np.intersect1d(note_array[ np.where( note_array["onset_beat"] < index)], note_array[ np.where( note_array["onset_beat"]+note_array["duration_beat"] >= index )])
		tnotes = np.union1d(temp1, temp2)
		X[i] = chord_to_intervalVector(tnotes["pitch"])
	return X


def analysis_to_SSM(note_array, window_size=1, return_analysis_matrix=False):
	"""
	Perform the intervalic analysis and output structure SSM.


	Parameters
	----------
	note_array : structured array
		The note array ordered on onsets
	window_size: float
		Expressed in portion of the time signature nominator.
		i.e. ts 4/4 and window_size=0.5 means a window of 2 beats.
	return_analysis_matrix : bool
		If True returns the interval analysis of the note array.

	Returns:
	--------
	S : np.array
		The self_similarity matrix, i.e. cosine similarity of the PCA of X, 
		where X is the intervalic analysis of the note_array.
	X : np.array (optional)
		The intervalic analysis of the note_array.

	Examples
	--------
	import partitura as pt
	note_array = pt.load_score(pt.EXAMPLE_MUSICXML).note_array(include_time_signature=True)
	S = analysis_to_SSM(note_array)
	"""

	X = intervalic_analysis(note_array, window_size)
	S = cosine_similarity(X)
	if return_analysis_matrix:
		return S, X
	else:
		return S


def filter_diag_sm(S, L=30):
	"""Path smoothing of similarity matrix by forward filtering along main diagonal

	Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

	Args:
		S: Similarity matrix (SM)
		L: Length of filter

	Returns:
		S_L: Smoothed SM
	"""
	N = S.shape[0]
	M = S.shape[1]
	S_L = np.zeros((N, M))
	S_extend_L = np.zeros((N + L, M + L))
	S_extend_L[0:N, 0:M] = S
	for pos in range(0, L):
		S_L = S_L + S_extend_L[pos:(N + pos), pos:(M + pos)]
	S_L = S_L / L
	return S_L


def filter_diag_mult_sm(S, L=1, tempo_rel_set=np.asarray([1]), direction=0):
	"""Path smoothing of similarity matrix by filtering in forward or backward direction
	along various directions around main diagonal
	Note: Directions are simulated by resampling one axis using relative tempo values

	Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

	Args:
		S: Self-similarity matrix (SSM)
		L: Length of filter
		tempo_rel_set: Set of relative tempo values
		direction: Direction of smoothing (0: forward; 1: backward)

	Returns:
		S_L_final: Smoothed SM
	"""
	N = S.shape[0]
	M = S.shape[1]
	num = len(tempo_rel_set)
	S_L_final = np.zeros((N, M))

	for s in range(0, num):
		M_ceil = int(np.ceil(M/tempo_rel_set[s]))
		resample = np.multiply(np.divide(np.arange(1, M_ceil+1), M_ceil), M)
		np.around(resample, 0, resample)
		resample = resample - 1
		index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
		S_resample = S[:, index_resample]

		S_L = np.zeros((N, M_ceil))
		S_extend_L = np.zeros((N + L, M_ceil + L))

		# Forward direction
		if direction == 0:
			S_extend_L[0:N, 0:M_ceil] = S_resample
			for pos in range(0, L):
				S_L = S_L + S_extend_L[pos:(N + pos), pos:(M_ceil + pos)]

		# Backward direction
		if direction == 1:
			S_extend_L[L:(N+L), L:(M_ceil+L)] = S_resample
			for pos in range(0, L):
				S_L = S_L + S_extend_L[(L-pos):(N + L - pos), (L-pos):(M_ceil + L - pos)]

		S_L = S_L / L
		resample = np.multiply(np.divide(np.arange(1, M+1), M), M_ceil)
		np.around(resample, 0, resample)
		resample = resample - 1
		index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)

		S_resample_inv = S_L[:, index_resample]
		S_L_final = np.maximum(S_L_final, S_resample_inv)

	return S_L_final



def forw_back_smoothing(S, L=20, tempo_rel_set=np.asarray([1])):
	S_forward = filter_diag_mult_sm(S, L, tempo_rel_set, direction=0)
	S_backward = filter_diag_mult_sm(S, L, tempo_rel_set, direction=1)
	S_final = np.maximum(S_forward, S_backward)
	return S_final


def threshold_matrix(S, thresh, strategy='absolute', scale=False, penalty=0, binarize=False):
	"""Treshold matrix in a relative fashion

	Notebook: C4/C4/C4S2_SSM-Thresholding.ipynb

	Args:
		S: Input matrix
		thresh: Treshold (meaning depends on strategy)
		strategy: Thresholding strategy ('absolute', 'relative', 'local')
		scale: If scale=True, then scaling of positive values to range [0,1]
		penalty: Set values below treshold to value specified
		binarize: Binarizes final matrix (positive: 1; otherwise: 0)
		Note: Binarization is applied last (overriding other settings)


	Returns:
		S_thresh: Thresholded matrix
	"""
	if np.min(S) < 0:
		raise Exception('All entries of the input matrix must be nonnegative')

	S_thresh = np.copy(S)
	N, M = S.shape
	num_cells = N * M

	if strategy == 'absolute':
		thresh_abs = thresh
		S_thresh[S_thresh < thresh] = 0

	if strategy == 'relative':
		thresh_rel = thresh
		num_cells_below_thresh = int(np.round(S_thresh.size*(1-thresh_rel)))
		if num_cells_below_thresh < num_cells:
			values_sorted = np.sort(S_thresh.flatten('F'))
			thresh_abs = values_sorted[num_cells_below_thresh]
			S_thresh[S_thresh < thresh_abs] = 0
		else:
			S_thresh = np.zeros([N, M])

	if scale:
		cell_val_zero = np.where(S_thresh == 0)
		cell_val_pos = np.where(S_thresh > 0)
		if len(cell_val_pos[0]) == 0:
			min_value = 0
		else:
			min_value = np.min(S_thresh[cell_val_pos])
		max_value = np.max(S_thresh)
		# print('min_value = ', min_value, ', max_value = ', max_value)
		if max_value > min_value:
			S_thresh = np.divide((S_thresh - min_value), (max_value - min_value))
			if len(cell_val_zero[0]) > 0:
				S_thresh[cell_val_zero] = penalty
		else:
			print('Condition max_value > min_value is voliated: output zero matrix')

	if binarize:
		S_thresh[S_thresh > 0] = 1
		S_thresh[S_thresh < 0] = 0
	return S_thresh


def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
	"""Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1]
	See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

	Notebook: C4/C4S4_NoveltySegmentation.ipynb

	Args:
		L: Parameter specifying the kernel size M=2*L+1
		var: Variance parameter determing the tapering (epsilon)

	Returns:
		kernel: Kernel matrix of size M x M
	"""
	taper = np.sqrt(1/2) / (L * var)
	axis = np.arange(-L, L+1)
	gaussian1D = np.exp(-taper**2 * (axis**2))
	gaussian2D = np.outer(gaussian1D, gaussian1D)
	kernel_box = np.outer(np.sign(axis), np.sign(axis))
	kernel = kernel_box * gaussian2D
	if normalize:
		kernel = kernel / np.sum(np.abs(kernel))
	return kernel


def compute_novelty_ssm(S, kernel=None, L=10, var=0.5, exclude=False):
	"""Compute novelty function from SSM [FMP, Section 4.4.1]

	Notebook: C4/C4S4_NoveltySegmentation.ipynb

	Args:
		S: SSM
		kernel: Checkerboard kernel (if kernel==None, it will be computed)
		L: Parameter specifying the kernel size M=2*L+1
		var: Variance parameter determing the tapering (epsilon)
		exclude: Sets the first L and last L values of novelty function to zero

	Returns:
		nov: Novelty function
	"""
	if kernel is None:
		kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
	N = S.shape[0]
	M = 2*L + 1
	nov = np.zeros(N)
	# np.pad does not work with numba/jit
	S_padded = np.pad(S, L, mode='constant')

	for n in range(N):
		# Does not work with numba/jit
		nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
	if exclude:
		right = np.min([L, N])
		left = np.max([0, N-L])
		nov[0:right] = 0
		nov[left:N] = 0

	return nov


def compute_time_lag_representation(S, circular=True):
	"""Computation of (circular) time-lag representation

	Notebook: C4/C4S4_StructureFeature.ipynb

	Args:
		S: Self-similarity matrix
		circular: computes circular version

	Returns:
		L: (Circular) time-lag representation of S
	"""
	N = S.shape[0]
	if circular:
		L = np.zeros((N, N))
		for n in range(N):
			L[:, n] = np.roll(S[:, n], -n)
	else:
		L = np.zeros((2*N-1, N))
		for n in range(N):
			L[((N-1)-n):((2*N)-1-n), n] = S[:, n]
	return L

def novelty_structure_feature(L, padding=True):
	"""Computation of the novelty function from a circular time-lag representation

	Notebook: C4/C4S4_StructureFeature.ipynb

	Args:
		L: Circular time-lag representation
		padding: Padding the result with the value zero

	Returns:
		nov: Novelty function
	"""
	N = L.shape[0]
	if padding:
		nov = np.zeros(N)
	else:
		nov = np.zeros(N-1)
	for n in range(N-1):
		nov[n] = np.linalg.norm(L[:, n+1] - L[:, n])

	nov = (nov - np.min(nov)) / (np.max(nov) - np.min(nov) + 1e-6)
	return nov


def peak_detection(nov, thresh = 0.3, window=10):
	peaks = list()
	nov = np.where(nov > thresh, nov, 0)
	for i in range(nov.shape[0]-window):
		matrix = nov[i : i+window]
		local = argrelextrema(matrix, np.greater)[0]
		if local.size > 1:
			peaks.append(np.argmax(nov[peaks]) + i )
		elif local.size == 1:
			peaks.append(local[0]+i)
		else:
			pass
	return list(set(peaks))


def get_hssm(part, note_array, window_size=1, thresh=0.15, median_filter_size = (3, 21), gaussian_filter_size=8):
	"""Return the harmonic (Int_Vec) SSM and a set of lines within it.

	Parameters
	----------
	part : partitura.score.Part
		A partitura part, in this function it is a dummy variable can also be given empty.
	note_array : structured array
		The notes of the score with their corresponding onset, duration and pitch.
	step : int
		The step for the interval vector window, also reflects the information of the SSM pixels.
	thresh : float
		The threshold for smoothing and thresholding of the cosine SSM.
	line_length : int
		Detect lines from above this pixel length (propotional to the step, i.e. 1 beat )
	line_gap : int
		Detect lines that have above this distance between them.
	reg_E : float
		Parameter of tensor PCA.
	learning_rate : float
		Parameter of tensor PCA.
	n_iter_max : float
		Parameter of tensor PCA.    

	Returns
	-------
	S_smooth : int
		The SSM after smoothing and thresholding.
	line : list
		The set of lines detected in the SSM with probabilistic hough transform.

	Example
	-------
	import partitura as pt

	note_array = pt.load_score(pt.EXAMPLE_MUSICXML).note_array(include_time_signature=True)
	hssm = get_hssm(note_array)
	"""
	S_cos = analysis_to_SSM(note_array, window_size)
	S_smooth = threshold_matrix(forw_back_smoothing(S_cos), thresh=thresh, strategy='relative')
	L = compute_time_lag_representation(S_smooth)
	L_filter = ndimage.median_filter(L, median_filter_size)
	L_filter = ndimage.gaussian_filter(L_filter, gaussian_filter_size)
	nov = novelty_structure_feature(L_filter)
	return nov


def apply_ssm_analysis(X, median_filter_size = (3, 21), gaussian_filter_size=6):
	S_cos = cosine_similarity(X)
	S_smooth = threshold_matrix(forw_back_smoothing(S_cos), thresh=0.05, strategy='relative')
	L = compute_time_lag_representation(S_smooth)
	L_filter = ndimage.median_filter(L, median_filter_size)
	L_filter = ndimage.gaussian_filter(L_filter, gaussian_filter_size)
	nov = novelty_structure_feature(L_filter)
	return nov




