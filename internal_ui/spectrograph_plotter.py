"""
The code for the two vertical lines is based on the bounding rectangle in the following repository:
https://github.com/ashokfernandez/wxPython-Rectangle-Selector-Panel
"""

from __future__ import division

import wx

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
plt.rcParams['toolbar'] = 'None'

from scipy.signal.windows import triang
import scipy.io.wavfile as wavfile


class SpecPlotter(wx.Panel):
	# The minimum amount of time to be displayed.  This is not used when loading data, only zooming.
	MIN_TIME_SHOWN = .5

	def __init__(self, parent, id=-1, dpi=None, **kwargs):
		"""
		TODO:
		 - Actually set self.colormap_bounds initial values based on what the ArgEstiamtor produces (need to check with
		  Connor which values I should be using)
		"""
		wx.Panel.__init__(self, parent, id=id, **kwargs)

		self.parent = parent

		self.figure = mpl.figure.Figure(dpi=dpi, figsize=(2, 2))

		# Create an axes, turn off the labels and add them to the figure
		self.axes = plt.Axes(self.figure, [0, 0, 1, 1])
		self.axes.set_axis_off()   #Not sure why this is here
		self.figure.add_axes(self.axes)

		self.canvas = FigureCanvas(self, -1, self.figure)

		# Load the logo .wav file and produce it's spectrogram
		self.cur_fs, self.cur_data = wavfile.read("resources/logo.wav")
		self.cur_spectrogram = self.axes.specgram(self.cur_data, NFFT=1024, Fs=self.cur_fs)[-1]

		# Set and store the x_lims for the logo spectrogram
		self.x_lims = (0, self.cur_spectrogram.get_extent()[1])
		self.axes.set_xlim(self.x_lims)

		self.time_lims = None

		self.x_scaling_fn = None  # Set to None so it won't be used when not created properly
		self.inverse_x_scaling_fn = None

		# Initialise the lines
		self.line0 = self.axes.axvline(-1, 0, 1, color='red')
		self.line1 = self.axes.axvline(-1, 0, 1, color='red')

		self.x0 = None
		self.x1 = None

		self.log_scale = False
		self.dynamic_colormap_range = True
		self.cmap = "viridis"

		self.colormap_bounds = (-150, 100)

		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.canvas, 1, wx.EXPAND)
		# sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
		self.SetSizer(sizer)

		self.canvas.mpl_connect('button_press_event', self.on_press)
		self.canvas.mpl_connect('button_release_event', self.on_release)
		self.canvas.mpl_connect('motion_notify_event', self.on_motion)

		self.pressed = False
		self.colormap_sample_fig = None

	def toggle_log_scale(self):
		"""
		TODO:
		 - Actually implement the redrawing of the log scaled spectrogram
		"""
		self.log_scale = not self.log_scale

		##### REDRAW SPECTROGRAM #####

	def toggle_dynamic_colormap_range(self, actually_toggle_value=True):
		if actually_toggle_value:
			self.dynamic_colormap_range = not self.dynamic_colormap_range

		if self.time_lims is None:
			start_time = 0
			args = {"windowSize": 1024, "overlap": 1/8}
		else:
			start_time = self.time_lims[0]
			args = self.parent.args

		# If it's toggling the colormap range when looking at the logo, set the start time to 0
		# start_time = 0 if self.time_lims is None else self.time_lims[0]
		# args = self.parent.args
		self.load_data(None, None, start_time=start_time, nfft=args['windowSize'],
					   noverlap=int(args['windowSize'] * args['overlap']))

		self.adjust_colormap_sample()

	def change_cmap(self, new_cmap):
		self.cmap = new_cmap
		self.toggle_dynamic_colormap_range(actually_toggle_value=False)

	def show_colormap_sample(self):
		if self.colormap_sample_fig is not None:
			plt.close(self.colormap_sample_fig)

		self.colormap_sample_fig = plt.figure(num="Current Colormap Sample", figsize=(9, 1.5))
		plt.imshow(np.array([self.colormap_bounds]), cmap=self.cmap)
		plt.gca().set_visible(False)
		cax = plt.axes([0.1, 0.2, 0.8, 0.6])
		plt.colorbar(orientation="horizontal", cax=cax)
		plt.show()

	def adjust_colormap_sample(self):
		if self.colormap_sample_fig is not None and plt.fignum_exists(self.colormap_sample_fig.number):
			self.show_colormap_sample()


	def clear(self):
		if self.cur_spectrogram is not None:
			self.cur_spectrogram.remove()

		self.clear_lines(redraw=False) #Not redrawing so that a blank white screen doesn't exist

		self.cur_spectrogram = None

		self.x_lims = None
		self.time_lims = None

		self.x_scaling_fn = None
		self.inverse_x_scaling_fn = None

	def x_axis_scaler(self, to_scale):
		if self.x_scaling_fn is not None:
			return list(map(self.x_scaling_fn, to_scale))
		else:
			raise Exception("No scaling function was created yet!")


	def load_data(self, data, sampling_freq, start_time=0, nfft=1024, noverlap=128, redraw=True):
		"""


		:param data: The waveform data to be loaded as an ndarray
		:param sampling_freq: The sampling frequency of the data given (in Hz)
		:param start_time: The time at the start of the data
		:param nfft: The number of data points used in each block for the FFT. A power 2 is most efficient.
		:param noverlap: The number of points of overlap between blocks

		TODO:
		 - Move the scaling function declerations to the init function (after ensuring it isn't being checked against
		 None anywhere)
		"""
		if data is not None:
			self.cur_data = data
			self.cur_fs = sampling_freq

		# If a spectrogram has been generated already (PRETTY SURE THIS IS IMPOSSIBLE NOW)
		if self.cur_spectrogram is not None:
			self.cur_spectrogram.remove()

		spec, _, _, self.cur_spectrogram = self.axes.specgram(self.cur_data, NFFT=nfft, Fs=self.cur_fs,
												  window=triang(nfft), noverlap=noverlap, cmap=self.cmap,
												  vmin=None if self.dynamic_colormap_range else self.colormap_bounds[0],
												  vmax=None if self.dynamic_colormap_range else self.colormap_bounds[1])

		if self.dynamic_colormap_range:
			self.colormap_bounds = (np.min(spec), np.max(spec))
			self.adjust_colormap_sample()

		if data is not None:
			# Store the new x axis limits of the spectrogram for scaling the x axis coordinates to time,
			# and for rescaling the image
			self.x_lims = (0, self.cur_spectrogram.get_extent()[1])
			self.axes.set_xlim(self.x_lims)

			self.time_lims = (start_time, start_time + (len(self.cur_data) - 1) / self.cur_fs)

			self.x_scaling_fn = lambda x: self.time_lims[0] + (x - self.x_lims[0]) * (
						self.time_lims[1] - self.time_lims[0]) / (self.x_lims[1] - self.x_lims[0])

			self.inverse_x_scaling_fn = lambda t: self.x_lims[0] + (t - self.time_lims[0]) * (
						self.x_lims[1] - self.x_lims[0]) / (self.time_lims[1] - self.time_lims[0])
		else:
			self.axes.set_xlim(self.x_lims)
			
		self.clear_lines(redraw=redraw)

	def clear_lines(self, redraw=True):
		"""
		Clears the lines (moves them outside of view)

		:param redraw: A boolean parameter specifying if the canvas should be redrawn
		"""
		self.x0 = None
		self.x1 = None

		self.line0.set_xdata(-1)
		self.line1.set_xdata(-1)

		if redraw:
			self.canvas.draw()

	def set_lines(self, t0, t1, redraw=True):
		self.x0 = self.inverse_x_scaling_fn(t0)
		self.x1 = self.inverse_x_scaling_fn(t1)
		self.line0.set_xdata(self.x0)
		self.line1.set_xdata(self.x1)

		# SHOULD self.pressed BE CHANGED??????????????????????
		if redraw:
			self.canvas.draw()

	def on_press(self, event):
		"""
		Callback to handle the mouse being clicked and held over the canvas
		"""
		# Check that a spectrogram has been generated
		if self.time_lims is not None:
			# Check the mouse press was actually on the canvas
			if event.xdata is not None and event.ydata is not None:
				# Upon initial press of the mouse record the origin and record the mouse as pressed
				self.pressed = True
				self.line0.set_linestyle('dashed')
				self.line1.set_linestyle('dashed')

				self.x0 = event.xdata

				self.line0.set_xdata(self.x0)

	def on_release(self, event):
		"""
		Callback to handle the mouse being released over the canvas
		"""
		# Check that the mouse was actually pressed on the canvas to begin with and this isn't a rouge mouse
		# release event that started somewhere else.   Also check that a spectrogram has been generated
		if self.pressed and self.time_lims is not None:

			# Upon release draw the rectangle as a solid rectangle
			self.pressed = False

			self.line0.set_linestyle('solid')
			self.line1.set_linestyle('solid')

			# Check the mouse was released on the canvas, and if it wasn't then just leave the width and
			# height as the last values set by the motion event
			if event.xdata is not None and event.ydata is not None:
				self.x1 = event.xdata
				self.line1.set_xdata(self.x1)

			if self.x0 == self.x1:
				self.clear_lines()
			else:
				self.canvas.draw()

			self.parent.segmenter.select_segment(clear_spectrograph_lines=False)

	def on_motion(self, event):
		"""
		Callback to handle the motion event created by the mouse moving over the canvas
		"""
		# If the mouse has been pressed draw an updated rectangle when the mouse is moved so
		# the user can see what the current selection is.  Only do this if a spectrogram has been generated.
		if self.time_lims is not None:
			if self.pressed:

				# Check the mouse was released on the canvas, and if it wasn't then just leave the width and
				# height as the last values set by the motion event
				if event.xdata is not None and event.ydata is not None:
					self.x1 = event.xdata

				self.line1.set_xdata(self.x1)

				self.canvas.draw()

			status_bar_time = None if event.xdata is None else self.x_scaling_fn(event.xdata)
			self.parent.set_statusbar_values(status_bar_time)

	def zoom_mouse(self, zoom, mouse_relative_coord=.5):
		"""
		Zoom into the spectrogram's x axis, centered around a given location (usually the mouse's location).
		
		:param zoom: A float value indicating how much to zoom by.  If zooming in, it's a positive value specifying
		 the ratio of the currently viewable spectrogram which won't be seen after zooming.  If zooming out, it's a
		 negative value indicating how much to zoom out by.  Specifically, for value 0 < x < 1,
		 calling zoom(x) followed by zoom(-x) will be an identity (do nothing)
		:param mouse_relative_coord: The relative location in the x axis to zoom in on (usually generated from
		 the mouse position).  This location is given relative to the maximum x axis value,  e.g. .75 would
		 indicate the mouse (or position to zoom in on) has 75% of the window on it's left and 25% on it's right
		:return: The newly set time limits as a tuple in the following format: (start_time, end_time)
		"""
		zooming_out = zoom < 0

		# Ensuring the zoom is less than one is done so that it doesn't zoom out to outside the plot.
		if zooming_out:
			zoom = min(-zoom, 1 - np.finfo(np.float32).eps)  # A minimal epsilon subtraction is done to prevent division by zero

		cur_view_dist = self.x_lims[1] - self.x_lims[0]

		cursor_x = self.x_lims[0] + mouse_relative_coord * cur_view_dist

		# Compute the distance (in the matplotlib canvas's coordinates) that will be either removed (by zooming in),
		# or added (zoomed out)
		if zooming_out:
			distance_zoomed = cur_view_dist / (1 - zoom) - cur_view_dist
		else:
			distance_zoomed = cur_view_dist * zoom
			min_x_dist_shown = self.MIN_TIME_SHOWN * (self.x_lims[1] - self.x_lims[0]) / (
						self.time_lims[1] - self.time_lims[0])
			if cur_view_dist - distance_zoomed < min_x_dist_shown:
				distance_zoomed = cur_view_dist - min_x_dist_shown

		left_zoom_ratio = (cursor_x - self.x_lims[0]) / cur_view_dist
		right_zoom_ratio = 1 - left_zoom_ratio

		# Define the new x limits, and if zooming out ensuring the new x limits don't go outside the view of the plot
		# (not setting to self.x_lims since that would alter the scaling functions)
		if zooming_out:
			spectro_x_bounds = self.cur_spectrogram.get_extent()[:2]
			x_lims = (max(self.x_lims[0] - left_zoom_ratio * distance_zoomed, spectro_x_bounds[0]),
					  min(self.x_lims[1] + right_zoom_ratio * distance_zoomed, spectro_x_bounds[1]))
		else:
			x_lims = (self.x_lims[0] + left_zoom_ratio * distance_zoomed,
					  self.x_lims[1] - right_zoom_ratio * distance_zoomed)


		return self.zoom(x_lims=x_lims)

	def zoom(self, time_lims=None, x_lims=None):
		"""
		Entirely zoom into the specified section of the spectrogram.

		:param time_lims: The time limits to zoom in on as a tuple: (start_time, end_time).
		 This is only used if x_lims is not specified
		:param x_lims: The limits in terms of the canvas to zoom in on as a tuple: (start_coord, end_coord)
		"""
		if x_lims is None:
			x_lims = self.inverse_x_scaling_fn(time_lims)

		self.time_lims = self.x_axis_scaler(x_lims)
		self.axes.set_xlim(x_lims)
		self.x_lims = (self.x0, self.x1) = x_lims  #######I PROBABLY SHOULDN'T SET self.x0 or self.x1##################################################

		# If either of the two lines are outside of the currently viewed window, clear them
		if self.x0 > self.line0.get_xdata() or self.line0.get_xdata() > self.x1 or \
				self.x0 > self.line1.get_xdata() or self.line1.get_xdata() > self.x1:
			self.clear_lines(redraw=False)
			self.parent.segmenter.select_segment(clear_spectrograph_lines=False)

		self.canvas.draw()

		return self.time_lims
