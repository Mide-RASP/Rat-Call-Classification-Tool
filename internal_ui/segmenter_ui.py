from __future__ import division

import wx
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.patches import Rectangle

from sortedcontainers import SortedDict

class Segmenter(wx.Panel):
	def __init__(self, parent, id=-1, dpi=None, **kwargs):
		wx.Panel.__init__(self, parent, id=id, **kwargs)

		self.parent = parent

		self.figure = mpl.figure.Figure(dpi=dpi, figsize=(2, 2))

		# Create an axes, turn off the labels and add them to the figure
		self.axes = [plt.Axes(self.figure, [0, 0, 1, 1])]
		self.axes[0].set_axis_off()   #Not sure why this is here
		self.figure.add_axes(self.axes[0])

		self.canvas = FigureCanvas(self, -1, self.figure)

		self.cur_file_index = 0 #THIS COULD BE REMOVED AND INSTEAD self.parent.cur_file_index COULD BE USED (BUT I'M THINKING KEEPING THE DEPENDENCIES MINIMAL BETWEEN THE CLASSES IS BEST)

		#####BELOW COMMENT IS NOT CORRECT ANYMORE###############
		# A dictionary mapping segments to the matplotlib.patches.Rectangle objects which represent them
		self.segments = [SortedDict()]

		self.selected_segment = None


		self.cur_file_time_lims = None

		# Given a value in time units, get it's coordinate relative to the matplotlib figure
		self.x_scaling_fn = None

		# Given a coordinate in the matplotlib figure, convert it to time units
		self.x_scaling_inverse_fn = None

		self.background_color = (1, 1, 1)  # White
		self.selected_color = (1, 0, 0)  # Red
		self.classified_color = (0, 0, 1)  # Blue
		self.unclassified_color = (1, 1, 0)  # Yellow

		# The following is nice for getting the color without an if statement
		# (e.g. self.label_specific_colors[is_classified])
		self.label_specific_colors = [self.unclassified_color, self.classified_color]

		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.canvas, 1, wx.EXPAND)
		self.SetSizer(sizer)

		self.Bind(wx.EVT_CHAR_HOOK, self.on_key_press)
		self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
		self.canvas.mpl_connect('motion_notify_event', self.on_mouse_movement)

		self.MinSize = wx.Size(-1, 25)

	def clear(self):
		"""
		NOTES:
			 - Pretty sure I don't need to loop through the file entirely
		"""
		for file_index, seg, _ in list(self):
			self.segments[file_index][seg][0].remove()
			del self.segments[file_index][seg]

		self.figure.delaxes(self.axes[self.cur_file_index])

		self.axes = [plt.Axes(self.figure, [0, 0, 1, 1])]
		self.axes[0].set_axis_off()  # Not sure why this is here
		self.figure.add_axes(self.axes[0])

		self.cur_file_index = 0
		self.segments = [SortedDict()]

		self.selected_segment = None

		self.x_scaling_fn = None
		self.x_scaling_inverse_fn = None

		self.canvas.draw()

	def __iter__(self, forwards=True):
		"""
		An iterator to go through all the segments stored, and give a tuple of relevant information.

		Each value yielded will be a tuple with the the following information:
		(file_index, (start_time, end_time), has_been_classified)
		"""
		increment = 2 * int(forwards) - 1
		for file_index, file_segments in zip(range(len(self.segments))[::increment], self.segments[::increment]):
			for seg, (_, classified) in file_segments.items()[::increment]:
				yield file_index, seg, classified

	def currently_empty(self):
		"""
		Returns True if there are no segments stored for the current file, and False if there are.
		"""
		return len(self.segments[self.cur_file_index]) == 0

	def switch_files(self, index, start_time, end_time, redraw=True):
		"""
		A method called when switching between files.

		:param index: The index to switch to
		:param start_time: The starting time of the file to be represented
		:param end_time: The ending time of the file to be represented
		"""
		self.axes[self.cur_file_index].set_xlim((0,1))

		# If the file trying to be switched to doesn't exist in the segmenter
		while len(self.segments) - 1 < index:
			self.segments.append(SortedDict())
			self.axes.append(plt.Axes(self.figure, [0, 0, 1, 1]))
			self.axes[-1].set_axis_off()

		# Reset the currently selected segment
		self.select_segment(redraw=False, clear_spectrograph_lines=False)

		self.figure.delaxes(self.axes[self.cur_file_index])

		self.cur_file_index = index

		self.figure.add_axes(self.axes[self.cur_file_index])

		if redraw:
			self.canvas.draw()

		self.set_scaling(start_time, end_time, True)

	def set_scaling(self, start_time, end_time, set_file_bounds=False):
		"""
		Set the scaling function and it's inverse.  Used when the segment of time being represented is changed.

		:param start_time: The starting time of the segment to be represented
		:param end_time: The ending time of the segment to be represented
		"""
		x_lims = self.axes[self.cur_file_index].get_xlim()

		self.x_scaling_fn = lambda t: x_lims[0] + (t - start_time) * (x_lims[1] - x_lims[0]) / (end_time - start_time)
		self.x_scaling_inverse_fn = lambda x: start_time + (x - x_lims[0]) * (end_time - start_time) / (
					x_lims[1] - x_lims[0])

		if set_file_bounds:
			self.cur_file_time_lims = (start_time, end_time)

	def remove_segment(self, segment):
		"""
		Removes the given segment from the Segmenter, and if it's been classified, remove that as well.

		:param segment: The segment to be removed (as a tuple of the start and end times of the segment)
		"""
		self.segments[self.cur_file_index][segment][0].remove()

		del self.segments[self.cur_file_index][segment]

		self.parent.remove_classification(segment)

		if segment == self.selected_segment:
			self.selected_segment = None

		self.parent.spec_panel.clear_lines()

		self.canvas.draw()

	def add_segment(self, start_time, end_time, classified=False, redraw=True):
		"""
		Adds a segment to the Segmenter.

		:param start_time: The starting time for the segment to be added
		:param end_time: The ending time for the segment to be added
		:param classified: True if the segment to be added has been classified, False if not
		:param redraw: True if the matplotlib canvas should be redrawn
		:return: True if the segment was successfully added, False if not
		"""
		if self.is_valid_addition(start_time, end_time):
			segment_rect = self.fill_segment(
				start_time,
				end_time,
				self.label_specific_colors[classified])

			self.segments[self.cur_file_index][(start_time, end_time)] = [segment_rect, classified]

			self.select_segment(redraw=redraw)
			return True
		else:
			return False

	def fill_segment(self, start_time, end_time, color):
		"""
		Fills the area between start_coord and end_coord (given in time units) with a Rectangle

		:param start_time: The starting time for the segment to be filled
		:param end_time: The ending time for the segment to be filled
		:param color: The color to make the filled segment (given as rgp triplet, but anything matplotlib
		accepts should work)
		:return: The matplotlib.patches.Rectangle object used to denote the segment
		"""
		scaled_start_coord = self.x_scaling_fn(start_time)
		scaled_end_coord = self.x_scaling_fn(end_time)
		BIG_ENOUGH_HEIGHT = 2 ############

		new_rect = Rectangle((scaled_start_coord, -.5),
							 scaled_end_coord-scaled_start_coord,
							 BIG_ENOUGH_HEIGHT,
							 facecolor=color)

		self.axes[self.cur_file_index].add_patch(new_rect)

		return new_rect

	def cur_segment_classified(self):
		"""
		Changes the selected segment from unclassified to classified.
		"""
		cur_segment = self.segments[self.cur_file_index][self.selected_segment]

		cur_segment[0].set_color(self.classified_color)
		cur_segment[1] = True

	def select_segment(self, segment=None, redraw=True, clear_spectrograph_lines=True):
		"""
		Selects the given segment.  If not specifying the segment, this function is used to deselect the currently
		selected segment.

		:param segment: The segment to be selected (as a tuple of the start and end times of the segment), or None if no
		segment should be selected
		:param redraw: A boolean value, specifying if the matplotlib canvas should be redrawn after the function is run
		:param clear_spectrograph_lines:  A boolean value, specifying if the lines in the spectrograph should be cleared
		"""
		if self.selected_segment is not None:
			# Set the currently selected segment color to the color it was prior to being selected
			cur_segment_info = self.segments[self.cur_file_index][self.selected_segment]
			cur_segment_info[0].set_color(self.label_specific_colors[cur_segment_info[1]])

		if segment is not None:
			self.selected_segment = segment
			self.segments[self.cur_file_index][segment][0].set_color(self.selected_color)

			self.parent.spec_panel.set_lines(*segment, redraw=redraw)
		else:
			self.selected_segment = None

			if clear_spectrograph_lines:
				self.parent.spec_panel.clear_lines()

		if redraw:
			self.canvas.draw()

	def on_mouse_click(self, event):
		"""
		A callback to be used when the mouse is clicked on the Segmenter panel.
		"""
		if self.x_scaling_fn is not None:
			selected_seg = self.get_segment(event.xdata)

			self.select_segment(selected_seg)

	def on_key_press(self, event):
		"""
		The callback used when a keystroke is registered
		"""
		keycode = event.GetKeyCode()
		if keycode == wx.WXK_DELETE and self.selected_segment is not None:
			self.remove_segment(self.selected_segment)
		else:
			self.parent.on_key_press(event)

	def is_valid_addition(self, start_time, end_time):
		"""
		Checks if adding the segment defined by the given start/end time can be added to the Segmenter.

		:param start_time: The starting time for the segment to be added
		:param end_time: The ending time for the segment to be added
		:return: True if the segment can be added, False if not
		"""
		intervals = [[self.cur_file_time_lims[0]] * 2] + \
					list(self.segments[self.cur_file_index]) + \
					[[self.cur_file_time_lims[1]] * 2]


		# Other break conditions could be used here to leave the loop faster, but the speed benifit should be super
		# minimal so I'm leaving it as is for now
		for j in xrange(len(intervals) - 1):
			if intervals[j][1] <= start_time and intervals[j + 1][0] >= end_time:
				return True
		return False

	@staticmethod
	def validate_and_order_segments(segments, file_start_time, file_end_time):
		"""
		Verify if the given segments are valid and return the segments ordered properly (sorted by start time,
		and switching a segments start/end times if they end before they start).  Reasons for being invalid include
		having zero	width (e.g. [1,1]), or having segments which overlap.

		:param segments: A list of lists (or tuples) representing the segments to be validated.
		e.g. if there were two segments, one starting at 1 and ending at 2 and another starting at 3 and ending at 4
		the following would be given: [[1, 2], [3, 4]]
		:return: None if the given segments are invalid, or the segments ordered properly

		NOTES:
		- Now that sorted dictionaries are used, some of this might not be needed
		"""
		if not len(segments):
			return segments

		# Check that no segments have start times equal to their end times, and swaps start and end times for segments
		# with end times before their start times
		for seg in segments:
			if seg[0] == seg[1]:
				return None
			elif seg[0] > seg[1]:
				seg.reverse()

			# Check that all segments are within the file being viewed
			if seg[0] < file_start_time or seg[1] > file_end_time:
				return None

		sorted_segs = sorted(segments)

		# Check that no segments are overlapping
		if any(sorted_segs[j][1] > sorted_segs[j+1][0] for j in xrange(len(segments) - 1)):
			return None

		return sorted_segs

	def load_segments(self, segments, classified=False):
		"""
		Loads a set of segments into the Segmenter.

		:param segments: A list of lists (or tuples) representing the segments to be loaded.
		e.g. if there were two segments, one starting at 1 and ending at 2 and another starting at 3 and ending at 4
		the following would be given: [[1, 2], [3, 4]]
		:param classified: A value indicating if the given segments have been classified.  It's ether a bool,
		or a list of bools the same size as 'segments'.

		NOTES:
		- Now that sorted dictionaries are used, some of this might not be needed
		"""
		ordered_segments = self.validate_and_order_segments(segments, *self.cur_file_time_lims)
		if ordered_segments is None:
			raise Exception("The segments to be loaded are not valid.  Reasons for being invalid include having "
							"zero segment width (e.g. [1,1]), having segments outside of the bounds for the file, "
							"or having segments which overlap!")
		elif isinstance(classified, list) and len(classified) != len(ordered_segments):
			raise Exception("The length of 'segments' must be equal to the length of 'classified'!")
		else:
			if isinstance(classified, bool):
				classified = [classified] * len(segments)

			for seg, is_classified in zip(segments, classified):
				self.add_segment(seg[0], seg[1], is_classified, redraw=False)

			self.canvas.draw()

	def get_segment(self, x_coord):
		"""
		Gets the segment containing the given x_coord, or None if no segment is defined there

		:param x_coord: The coordinate in the matplotlib canvas
		:return: The segment that the given x_coord is contained within, or None if it isn't contained within a segment
		"""
		scaled_x_coord = self.x_scaling_inverse_fn(x_coord)
		for seg in self.segments[self.cur_file_index]:
			if seg[0] < scaled_x_coord < seg[1]:  # THIS MAKES NO SEGMENT OWN IT'S BOUNDARIES (may not be ideal)
				return seg

		return None

	def on_mouse_movement(self, event):
		"""
		A callback used when the mouse is moved (over the Segmenter's panel).
		"""
		if self.x_scaling_fn is not None:
			self.parent.set_statusbar_values(None if event.xdata is None else self.x_scaling_inverse_fn(event.xdata))

	def fill_with_empty_info(self, num_files):
		"""
		NOTES:
			- ASSUMES THIS OBJECT WAS JUST CREATED
		"""
		self.segments = [SortedDict() for _ in range(num_files)]
		self.axes = self.axes + [plt.Axes(self.figure, [0, 0, 1, 1]) for _ in range(num_files-1)]

	def zoom(self, start_time, end_time):
		self.axes[self.cur_file_index].set_xlim(self.x_scaling_fn(np.array([start_time, end_time])))
		self.set_scaling(start_time, end_time)

		self.canvas.draw()

