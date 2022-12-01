import lib.xray.utils as pu

import numpy as np
import imageio
import os


class CLAHE():
	'''Contrast Limited Adaptive Histogram Equalization.
	In reality, we do a normalization before applying CLAHE, making it the N-CLAHE method, but in
	N-CLAHE the normalization is done using a log function, instead of a linear one, as we use here.
	'''

	def __init__(self, filename, results_path, window_size, clip_limit, n_iter):
		self.filename = filename
		self.results_path = results_path
		self.window_size = window_size
		self.clip_limit = clip_limit
		self.n_iter = n_iter

	def run(self):
		image = imageio.imread(self.filename)

		if len(image.shape) > 2:
			image = pu.to_grayscale(image)

		normalized_image = pu.normalize(np.min(image), np.max(image), 0, 255, image)

		equalized_image = self.clahe(normalized_image)
		imageio.imwrite(os.path.join(self.results_path, os.path.basename(self.filename)), equalized_image)

	def clahe(self, image):
		'''Applies the CLAHE algorithm in an image.

		Parameters:
			image: image to be processed.

		Returns a processed image.
		'''

		border = self.window_size // 2

		padded_image = np.pad(image, border, "reflect")
		shape = padded_image.shape
		padded_equalized_image = np.zeros(shape).astype(np.uint8)

		for i in range(border, shape[0] - border):
			for j in range(border, shape[1] - border):
				# Region to extract the histogram
				region = padded_image[i-border:i+border+1, j-border:j+border+1]
				# Calculating the histogram from region
				hist, bins = pu.histogram(region)
				# Clipping the histogram
				clipped_hist = pu.clip_histogram(hist, bins, self.clip_limit)
				# Trying to reduce the values above clipping
				for _ in range(self.n_iter):
					clipped_hist = pu.clip_histogram(hist, bins, self.clip_limit)
				# Calculating the CDF
				cdf = pu.calculate_cdf(hist, bins)
				# Changing the value of the image to the result from the CDF for the given pixel
				padded_equalized_image[i][j] = cdf[padded_image[i][j]]

		# Removing the padding from the image
		equalized_image = padded_equalized_image[border:shape[0] - border, border:shape[1] - border].astype(np.uint8)

		return equalized_image

	def clipped_histogram_equalization(self, region):
		'''Calculates the clipped histogram equalization for the given region.

		Parameters:
			region: array-like.

		Returns a dictionary with the CDF for each pixel in the region.
		'''

		# Building the histogram
		hist, bins = pu.histogram(region)
		n_bins = len(bins)

		# Removing values above clip_limit
		excess = 0
		for i in range(n_bins):
			if hist[i] > self.clip_limit:
				excess += hist[i] - self.clip_limit
				hist[i] = self.clip_limit

		## Redistributing exceding values ##
		# Calculating the values to be put on all bins
		for_each_bin = excess // n_bins
		# Calculating the values left
		leftover = excess % n_bins

		hist += for_each_bin
		for i in range(leftover):
			hist[i] += 1

		# Calculating probability for each pixel
		pixel_probability = hist / hist.sum()
		# Calculating the CDF (Cumulative Distribution Function)
		cdf = np.cumsum(pixel_probability)

		cdf_normalized = cdf * 255

		hist_eq = {}
		for i in range(len(cdf)):
			hist_eq[bins[i]] = int(cdf_normalized[i])

		return hist_eq
