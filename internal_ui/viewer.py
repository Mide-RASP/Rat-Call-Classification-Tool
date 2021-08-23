from __future__ import division

import wx
import os

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use('wxagg')
import matplotlib.pyplot as plt

import itertools
import json
import shutil
import csv

from sortedcontainers import SortedDict
from collections import defaultdict


# from raspy_tools import export_dialog, export_utils  # EVENTUALLY ALL CODE HOUSED IN RaspLab MUST BE MOVED TO THE Raspy PROJECT#########
import export_utils_copy as export_utils

from raspy.core import Rasper
from raspy.runRaspy import maxFileSize, splitLargeFile
from raspy.argEstimator import estimateArgs

from spectrograph_plotter import SpecPlotter
from segmenter_ui import Segmenter
from ui_helper import YesNoDialog, IntegerEntryDialog, ColormapMinMaxDialog


progress_style = (wx.PD_CAN_ABORT | wx.PD_APP_MODAL | wx.PD_ESTIMATED_TIME ########TAKEN FROM run_wrapper
                  | wx.PD_REMAINING_TIME)


class MainWindow(wx.Frame):
	"""
	Rasper could likely take advantage of librosa or something similar for optimization

	TODO:
	 - Have naming convensions for scaling functions be consistent between segmenter and spectrogram plotter
	 (I think they're currently exactly the opposite)
	 - Have a function call to determine if any data is loaded rather than a manual check (I believe it's being
	 done through the spectrogram panel)
	 - Have some way of viewing the frequency information (likely just for the mouse cursor's location)
	"""
	DEFAULT_ZOOM_RATIO = .1

	ID_RENDER_RASPY = wx.NewId()
	ID_RASPY_SEGMENT = wx.NewId()
	ID_NEXT_FILE = wx.NewId()
	ID_PREVIOUS_FILE = wx.NewId()
	ID_GOTO_FILE = wx.NewId()
	ID_GOTO_SEGMENT = wx.NewId()
	ID_ZOOM_SEGMENT = wx.NewId()
	ID_TOGGLE_LOG_SCALE = wx.NewId()
	ID_SAVE_FILE = wx.NewId()
	ID_OPEN_DIR = wx.NewId()
	ID_DYNAMIC_COLORMAP_RANGE = wx.NewId()
	ID_SET_CMAP_BOUNDS = wx.NewId()
	ID_OPEN_CMAP_SAMPLE = wx.NewId()

	timeScalar = 1.0 / (10 ** 6)  # Taken from RaspLab's viewer.py file

	def __init__(self, parent, title):
		"""
		TODO:
		 - Fill in dummy strings (ones with all caps) with actual content
		"""
		self.dirname = ""
		self.folder_name = ""
		self.call_classifications = []

		self.root_filename = ""
		self.file_list = []

		self.data_list = []

		self.cum_file_lens = []  # A list of the length in seconds of each WAV file loaded

		self.cur_file_index = 0

		self.samp_freq = None
		self.rasper = None
		self.args = None

		# A "-1" in the size parameter instructs wxWidgets to use the default size.
		# In this case, we select 200px width and the default height.
		wx.Frame.__init__(self, parent, title=title, size=(200, -1))
		self.spec_panel = SpecPlotter(self)
		self.status_bar = self.CreateStatusBar(3)  # A Statusbar in the bottom of the window

		# Setting up the menu.
		self.filemenu = wx.Menu()
		menu_open_file = self.filemenu.Append(wx.ID_OPEN, "Open New &File", " Open a .wav file to edit")
		menu_open_dir = self.filemenu.Append(self.ID_OPEN_DIR, "&Open Existing Run", " Open an existing directory created by this program")
		menu_run = self.filemenu.Append(self.ID_RENDER_RASPY, "&Run Classification", " Runs the Raspy algorithm")
		# menu_segment = self.filemenu.Append(self.ID_RASPY_SEGMENT, "&Segment File", " Runs the Raspy segmenter on the data")
		menu_open_next = self.filemenu.Append(self.ID_NEXT_FILE, "&Next File\t-->", " Moves to the next file")
		menu_open_previous = self.filemenu.Append(self.ID_PREVIOUS_FILE, "&Previous File\t<--",
												  " Moves to the previous file")
		menu_goto_file_index = self.filemenu.Append(self.ID_GOTO_FILE, "&Goto File Index",
												  " Moves to file at the specified index")
		menu_goto_call_index = self.filemenu.Append(self.ID_GOTO_SEGMENT, "Goto Call/Segment &Index",
												  " Zoom in on a call/segment specified by index")
		menu_save = self.filemenu.Append(self.ID_SAVE_FILE, "Save as &CSV\tCtrl+S",
										 "Saves the segments in any of the open files")
		menu_exit = self.filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

		self.fix_menu_items()
		self.filemenu.FindItemById(self.ID_RENDER_RASPY).Enable(False)
		# self.filemenu.FindItemById(self.ID_RASPY_SEGMENT).Enable(False)
		self.filemenu.FindItemById(self.ID_GOTO_FILE).Enable(False)

		self.colormap_menu = wx.Menu()
		self.cmap_menu_dict = {}
		colormap_options = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gist_earth', 'Greys']
		for cmap_name in colormap_options:
			self.cmap_menu_dict[cmap_name] = self.colormap_menu.Append(
				wx.ID_ANY,
				cmap_name,
				"Set the colormap to be %s"%cmap_name,
				kind=wx.ITEM_CHECK)

			self.Bind(wx.EVT_MENU, lambda x, n=cmap_name: self.cmap_changed(n), self.cmap_menu_dict[cmap_name])

		self.cmap_menu_dict['viridis'].Check()


		self.view_menu = wx.Menu()
		colormap_selector_menu = self.view_menu.Append(wx.ID_ANY, "&Colormap Selection", self.colormap_menu)
		menu_dynamic_colormap_range = self.view_menu.Append(self.ID_DYNAMIC_COLORMAP_RANGE, "Dynamic Colormap Range",
															"Toggle using the minimum and maximum values in whatever spectrogram you view as the bounds for the colormap", kind=wx.ITEM_CHECK)
		menu_set_colormap_bounds = self.view_menu.Append(self.ID_SET_CMAP_BOUNDS, "&Set Colormap Bounds",
														 "Set the bounds for static colormap scaling")
		# menu_log_scale = self.view_menu.Append(self.ID_TOGGLE_LOG_SCALE, "LOG_SCALE", "MORE DETAILED DESCRIPTION",
		# 									   kind=wx.ITEM_CHECK)
		menu_cmap_samp = self.view_menu.Append(self.ID_OPEN_CMAP_SAMPLE, "Display Colormap Sample",
											   "Open a window with the colormap with values plotted",
											   kind=wx.ITEM_CHECK)
		menu_zoom_segment = self.view_menu.Append(self.ID_ZOOM_SEGMENT, "&Zoom Into Segment\tZ",
												 " Zooms into a segment which is selected.")

		menu_dynamic_colormap_range.Check()

		# Creating the menubar.
		menu_bar = wx.MenuBar()
		menu_bar.Append(self.filemenu, "&File")  # Adding the "filemenu" to the menu_bar
		menu_bar.Append(self.view_menu, "&View")
		self.SetMenuBar(menu_bar)  # Adding the MenuBar to the Frame content.

		# Events.
		self.Bind(wx.EVT_MENU, lambda e: self.on_open(opening_dir=False), menu_open_file)
		self.Bind(wx.EVT_MENU, lambda e: self.on_open(opening_dir=True), menu_open_dir)
		self.Bind(wx.EVT_MENU, lambda e: self.export_as_csv(), menu_save)
		self.Bind(wx.EVT_MENU, lambda e: self.on_exit(), menu_exit)
		self.Bind(wx.EVT_MENU, lambda e: self.run_algorithm(), menu_run)
		self.Bind(wx.EVT_MENU, lambda e: self.goto_file_index(self.cur_file_index + 1), menu_open_next)
		self.Bind(wx.EVT_MENU, lambda e: self.goto_file_index(self.cur_file_index - 1), menu_open_previous)
		self.Bind(wx.EVT_MENU, lambda e: self.prompt_and_set_file_index(), menu_goto_file_index)
		self.Bind(wx.EVT_MENU, lambda e: self.prompt_for_segment_zoom(), menu_goto_call_index)
		self.Bind(wx.EVT_MENU, lambda e: self.zoom_into_segment(self.segmenter.selected_segment, self.cur_file_index),
				  menu_zoom_segment)
		# self.Bind(wx.EVT_MENU, lambda e: self.spec_panel.toggle_log_scale(), menu_log_scale)
		self.Bind(wx.EVT_MENU, lambda e: self.spec_panel.show_colormap_sample(), menu_cmap_samp)
		self.Bind(wx.EVT_MENU, lambda e: self.spec_panel.toggle_dynamic_colormap_range(), menu_dynamic_colormap_range)
		self.Bind(wx.EVT_MENU, lambda e: self.prompt_and_set_colormap_bounds(), menu_set_colormap_bounds)
		# self.Bind(wx.EVT_MENU, lambda e: self.run_segmentation(), menu_segment)

		self.Bind(wx.EVT_MOUSEWHEEL, self.mouse_scrolled)

		self.Bind(wx.EVT_CHAR_HOOK, self.on_key_press)

		self.Bind(wx.EVT_CLOSE, lambda e: self.on_exit())

		self.sizer2 = wx.BoxSizer(wx.HORIZONTAL)
		self.buttons = []

		self.call_types = ["Complex", "Upward Ramp", "Downward Ramp", "Flat", "Short", "Split", "Step Up", "Step Down",
					  "Multi-Step", "Trill", "Flat/Trill Combination", "Trill with Jumps", "Inverted U",
					  "Composite", "22-kHz Call", "Other/Unknown"]

		for j, call_type in enumerate(self.call_types):
			self.buttons.append(wx.Button(self, -1, call_type))
			self.sizer2.Add(self.buttons[j], 1, wx.EXPAND)

			self.Bind(wx.EVT_BUTTON, lambda x, j=j: self.classification_chosen(self.call_types[j]), self.buttons[j])

		self.segmenter = Segmenter(self)


		# Use some sizers to see layout options
		self.sizer = wx.BoxSizer(wx.VERTICAL)
		self.sizer.Add(self.spec_panel, 1, wx.EXPAND)
		self.sizer.Add(self.segmenter, 0, wx.EXPAND)
		self.sizer.Add(self.sizer2, 0, wx.EXPAND)

		# Layout sizers
		self.SetSizer(self.sizer)
		self.SetAutoLayout(1)
		self.sizer.Fit(self)

		self.Maximize()
		self.Show()
		
	def cmap_changed(self, new_cmap):
		prev_cmap = self.spec_panel.cmap
		if prev_cmap != new_cmap:
			self.cmap_menu_dict[prev_cmap].Check(check=False)
			self.cmap_menu_dict[new_cmap].Check()

			self.spec_panel.change_cmap(new_cmap)



	def loading_new_file(self):
		# Promt to save existing information
		save_dlg = YesNoDialog(self,
							   yes_fn=self.export_as_csv,
							   no_fn=lambda: None,
							   text="Would you like to save your work before opening a new workspace?",
							   title="Save Before Opening")

		# If the X button or escape key aren't used to close the dialog
		if save_dlg.ShowModal() != wx.ID_ABORT:
			# Wipe all information stored in the segmenter, and wipe it's graphics
			self.segmenter.clear()

			# Wipe all information stored in the spectrogram panel, and wipe it's graphics
			self.spec_panel.clear()

			# Wipe all information stored in this class
			self.clear()

			return True
		return False

	def clear(self):
		self.dirname = ""
		self.folder_name = ""
		self.call_classifications = []

		self.root_filename = ""

		self.file_list = []
		self.data_list = []
		self.cum_file_lens = []

		self.cur_file_index = 0

		self.samp_freq = None
		self.rasper = None
		self.args = None

	def run_segmentation(self):
		if self.segmenter.currently_empty():
			if self.rasper is None:
				self.rasper = Rasper(fileName="%s\\%s"%(self.folder_name, self.root_filename),
									 arr = [],
									 # arr=np.concatenate(self.data_list),   ##99% SURE THIS IS FINE TO REMOVE
									 source=None, #??????????????????????
									 fs=self.samp_freq)

			with wx.ProgressDialog(
					title="Segmentation Progress",
					message='Segmenting %s'%self.file_list[self.cur_file_index].split("\\")[-1],
					parent=self,
					style=progress_style) as progressBar:

				self.rasper.callback = export_utils.RaspyProgressTracker(progressBar)
				self.rasper.fileData = self.data_list[self.cur_file_index]  # CHECK WITH CONNOR THIS IS OKAY TO DO

				try:
					times = self.rasper.isolateCalls(**{k: v for k, v in self.args.items() if k != "syllableGap"})[0]

					time_offset = self.cum_file_lens[self.cur_file_index]

					self.segmenter.load_segments([(ary[0]+time_offset, ary[-1]+time_offset) for ary in times])
				except export_utils.StopExecution:
					pass

		else:
			wx.MessageBox("Segmentation was not done since segments already exist on this file!",
						  "Segmentation Warning", wx.OK, self)

	def run_algorithm(self):
		if self.segmenter.currently_empty():
			if self.rasper is None:
				self.rasper = Rasper(fileName="%s\\%s"%(self.folder_name, self.root_filename),
									 arr=[],
									 # arr=np.concatenate(self.data_list),   ##99% SURE THIS IS FINE TO REMOVE
									 source=None,  # ??????????????????????
									 fs=self.samp_freq)

			with wx.ProgressDialog(
					title="Segmentation Progress",
					message='Segmenting %s' % self.file_list[self.cur_file_index].split("\\")[-1],
					parent=self,
					style=progress_style) as progressBar:
				# export_dialog.RaspyExportDialog(root=self) ##############THIS INIT FUNCTION WAS CHANGED IN A WAY THAT NEEDS CHANGING BACK ONCE EVERYTHING ELSE IS FIGURED OUT##############
				# temp = export_dialog.RaspyExportDialog.getExport(root=self,
				# 												 args=self.args)  ##############THIS INIT FUNCTION WAS CHANGED IN A WAY THAT NEEDS CHANGING BACK ONCE EVERYTHING ELSE IS FIGURED OUT##############

				self.rasper.fullFileName = self.file_list[self.cur_file_index]
				self.rasper.fileData = self.data_list[self.cur_file_index]  # CHECK WITH CONNOR THIS IS OKAY TO DO
				self.rasper.callback = export_utils.RaspyProgressTracker(progressBar)

				report_builder = self.rasper.classify(**self.args)

				cur_time_offset = self.cum_file_lens[self.cur_file_index]
				segs = []
				for call in self.rasper.calls:
					start_time = cur_time_offset + call.times[0]
					end_time = cur_time_offset + call.times[-1]

					if not self.insert_classification(start_time, end_time, call.callType):
						raise Exception(
							"A classification tried to be inserted but failed!  The classification was: (%f,%f):%s" % (
							start_time, end_time, call.callType))

					segs.append((start_time, end_time))
				self.segmenter.load_segments(segs, classified=True)  # only using this for it's saftey checks (otherwise would insert calls 1 at a time)
		else:
			wx.MessageBox("Classification was not done since segments already exist on this file!",
						  "Classification Warning", wx.OK, self)

	def fix_menu_items(self):
		"""
		Enables or disables menu options depending on if they should be pressable
		"""
		next_file_item = self.filemenu.FindItemById(self.ID_NEXT_FILE)
		if next_file_item.IsEnabled():
			if self.cur_file_index >= len(self.data_list) - 1:
				next_file_item.Enable(False)
		else:
			if self.cur_file_index < len(self.data_list) - 1:
				next_file_item.Enable(True)

		previous_file_item = self.filemenu.FindItemById(self.ID_PREVIOUS_FILE)
		if previous_file_item.IsEnabled():
			if self.cur_file_index <= 0:
				previous_file_item.Enable(False)
		else:
			if self.cur_file_index > 0:
				previous_file_item.Enable(True)

	def on_key_press(self, event):
		if self.spec_panel.time_lims is not None:
			keycode = event.GetKeyCode()

			if event.ControlDown() and keycode == wx.WXK_RIGHT:
				self.zoom_on_next_segment(True)
			elif event.ControlDown() and keycode == wx.WXK_LEFT:
				self.zoom_on_next_segment(False)
			elif keycode == wx.WXK_RIGHT and self.cur_file_index < len(self.data_list) - 1:
				self.goto_file_index(self.cur_file_index + 1)
			elif keycode == wx.WXK_LEFT and self.cur_file_index != 0:
				self.goto_file_index(self.cur_file_index - 1)
			elif event.ControlDown() and keycode == ord('S'): #Not sure why it's not 's'
				self.export_as_csv()
			elif keycode == wx.WXK_UP:
				new_time_limits = self.spec_panel.zoom_mouse(self.DEFAULT_ZOOM_RATIO)
				self.segmenter.zoom(*new_time_limits)
			elif keycode == wx.WXK_DOWN:
				new_time_limits = self.spec_panel.zoom_mouse(-self.DEFAULT_ZOOM_RATIO)
				self.segmenter.zoom(*new_time_limits)
			elif keycode == ord('Z'):
				if self.segmenter.selected_segment is not None: #####I DON'T THINK THIS IS NEEDED ANYMORE##########
					self.zoom_into_segment(self.segmenter.selected_segment, self.cur_file_index)

	def mouse_scrolled(self, event):
		if self.spec_panel.time_lims is not None:
			relative_x_coord = event.GetLogicalPosition(wx.WindowDC(self))[0] / self.GetSize()[0]
			zoom_ratio = self.DEFAULT_ZOOM_RATIO * event.GetWheelRotation() / event.GetWheelDelta()
			self.segmenter.zoom(*self.spec_panel.zoom_mouse(zoom_ratio, relative_x_coord))

	def on_exit(self):
		if self.spec_panel.time_lims is not None:
			save_dlg = YesNoDialog(self,
								  yes_fn=self.export_as_csv,
								  no_fn=lambda: None,
								  text="Would you like to save your work before closing?",
								  title="Save Before Exit")

			if save_dlg.ShowModal() == wx.ID_ABORT:
				return

		self.Destroy()

	def prompt_and_set_colormap_bounds(self):
		if self.spec_panel.dynamic_colormap_range:
			wx.MessageBox("Colormap range selection can only be done when dynamic colormap range is disabled!",
						  "Colormap Range Warning", wx.OK, self)
		else:
			cur_start, cur_end = self.spec_panel.colormap_bounds
			dlg = ColormapMinMaxDialog(self, cur_start, cur_end)
			if dlg.ShowModal() == wx.ID_OK:
				self.spec_panel.colormap_bounds = dlg.get_min_and_max()
				self.spec_panel.toggle_dynamic_colormap_range(actually_toggle_value=None)

	def goto_file_index(self, index, redraw=True):
		self.cur_file_index = index
		self.load_cur_file_spectrogram(redraw=redraw)
		self.segmenter.switch_files(index, self.spec_panel.time_lims[0], self.spec_panel.time_lims[1], redraw=redraw)
		self.fix_menu_items()
		self.status_bar.SetStatusText(self.file_list[self.cur_file_index], 2)

	def estimate_args(self, data):
		with wx.ProgressDialog(
				title="Argument Estimation",
				message="Calculating file parameters",
				parent=self) as progressBar:

			self.args = estimateArgs(data, self.samp_freq,
									 callback=export_utils.RaspyArgsProgressTracker(progressBar))
			return self.args

	def load_file_info(self, file_name, samp_freq=None):
		stored_freq, array = wavfile.read(file_name)
		array = array.astype(np.float32)

		if samp_freq is None:
			self.samp_freq = stored_freq

		abs_file_name = os.path.abspath(file_name)
		self.folder_name = os.path.splitext(abs_file_name)[0]
		# core_name = os.path.basename(self.folder_name)

		# make results folder
		if os.path.isdir(self.folder_name):
			for i in itertools.count(1):
				new_folder_name = '{} ({})'.format(self.folder_name, i)
				if not os.path.isdir(new_folder_name):
					break
			os.rename(self.folder_name, new_folder_name)
		os.mkdir(self.folder_name)

		args_file_name = self.folder_name + '.json'

		# get the parameters
		if not os.path.exists(args_file_name):
			# generate parameters from data
			fs, data = wavfile.read(abs_file_name, True)

			self.estimate_args(data)
			# save to file
			with open(args_file_name, 'w') as argsFile:
				json.dump(self.args, argsFile, indent=4)
		else:
			# load these files args
			with open(args_file_name, 'r') as argsFile:
				self.args = json.load(argsFile)

		# copy parameters file into results folder
		args_file_copy_name = os.path.join(self.folder_name, 'parameters.json')
		shutil.copy(args_file_name, args_file_copy_name)

		if os.path.getsize(abs_file_name) > maxFileSize + 128:
			self.data_list, _, self.file_list = zip(
				*splitLargeFile(abs_file_name, array, self.args, sampling_frequency=self.samp_freq))
		else:
			self.data_list = [array]
			self.file_list = [file_name]

		self.cur_file_index = 0

		self.status_bar.SetStatusText(self.file_list[0], 2)

		self.call_classifications = [SortedDict() for _ in self.data_list]

		file_lens = [(len(d) - 1) / self.samp_freq for d in self.data_list[:-1]]

		self.cum_file_lens = np.cumsum(np.r_[0, file_lens])

		self.fix_menu_items()

	def load_directory_info(self, directory):
		"""
		TODO:
		 - Write comments for what files this is looking for in the given directory
		 - Check if csv file checking needs the .endswith call
		"""
		self.folder_name = directory

		filename_helper_fn = lambda f: "%s\\%s" % (directory, f)

		csv_filename = None
		wav_filenames = []
		file_indices = []
		for filename in os.listdir(directory):
			if filename.endswith(".wav"):
				wav_filenames.append(filename_helper_fn(filename))
				file_indices.append(int(filename.split()[-1][:-4]))
			elif filename.endswith("segments_info.csv"):
				csv_filename = filename_helper_fn(filename)
			elif filename.endswith(".json"):
				# load the files args
				with open(filename_helper_fn(filename), 'r') as argsFile:
					self.args = json.load(argsFile)

		wav_filenames = np.asarray(wav_filenames)
		file_indices = np.asarray(file_indices)

		self.file_list = wav_filenames[file_indices.argsort()] #MIGHT NEED TO CONVERT TO A LIST

		self.data_list = [wavfile.read(filename)[1].astype(np.float32) for filename in self.file_list]

		file_lens = [(len(d) - 1) / self.samp_freq for d in self.data_list[:-1]]
		self.cum_file_lens = np.cumsum(np.r_[0, file_lens])

		self.call_classifications = [SortedDict() for _ in self.data_list]
		self.segmenter.fill_with_empty_info(len(self.data_list))

		if self.args is None:
			# generate parameters from data
			self.estimate_args(np.concatenate(self.data_list))
			# save to file
			with open(filename_helper_fn("parameters.json"), 'w') as argsFile:
				json.dump(self.args, argsFile, indent=4)

		if csv_filename is not None:
			self.load_from_csv(csv_filename)

		self.cur_file_index = 0
		self.segmenter.cur_file_index = 0

		self.status_bar.SetStatusText(self.file_list[0], 2)


	def on_open(self, opening_dir):
		if len(self.data_list):
			if not self.loading_new_file():
				return

		if opening_dir:
			file_dlg = wx.DirDialog(self, "Choose directory")
		else:
			file_dlg = wx.FileDialog(self, "Choose a file", self.dirname, "", "*.wav", wx.FD_OPEN)

		if file_dlg.ShowModal() == wx.ID_OK:
			if opening_dir:
				self.root_filename = "" ##############################################
				self.dirname = file_dlg.GetPath()
			else:
				self.root_filename = file_dlg.GetFilename()
				self.dirname = file_dlg.GetDirectory()  # PRETTY SURE I DON'T NEED TO MAKE THIS A CLASS VARIABLE
				f = os.path.join(self.dirname, self.root_filename)

			freq_dlg = IntegerEntryDialog(
				self,
				"Enter the sampling frequency in Hz (values less than 1 indicate the value provided by the WAV files should be used)",
				title="Sampling Frequency Specification",
				default=250000)

			freq_dlg.ShowModal()
			if freq_dlg.get_value() > 0:
				self.samp_freq = freq_dlg.get_value()

			if opening_dir:
				self.load_directory_info(self.dirname)
			else:
				self.load_file_info(f, self.samp_freq)

			self.filemenu.FindItemById(self.ID_RENDER_RASPY).Enable(True)
			# self.filemenu.FindItemById(self.ID_RASPY_SEGMENT).Enable(True)
			self.filemenu.FindItemById(self.ID_GOTO_FILE).Enable(True)

			self.load_cur_file_spectrogram()

			self.segmenter.set_scaling(*self.spec_panel.time_lims, set_file_bounds=True)

		file_dlg.Destroy()  # PRETTY SURE THIS ISN'T NEEDED

	def load_cur_file_spectrogram(self, redraw=True):
		if self.args is None:
			raise Exception("No args have been calculated when loading spectrogram!")

		self.spec_panel.load_data(
			self.data_list[self.cur_file_index],
			self.samp_freq,
			start_time=self.cum_file_lens[self.cur_file_index],
			nfft=self.args['windowSize'],
			noverlap=int(self.args['windowSize'] * self.args['overlap']),
			redraw=redraw)

	def insert_classification(self, start_time, end_time, classification, dry_run=False):
		"""
		THIS COMMENT NEEDS TO BE UPDATED

		Attempts to insert a new classification into the stored classifications.  It forces self.call_classifications
		to maintain a sorted order, and prevents any overlap between classifications

		:return: True if the classification was inserted successfully, False if not
		"""
		intervals = [[float('-inf'), float('-inf')]] + \
					list(self.call_classifications[self.cur_file_index]) + \
					[[float('inf'), float('inf')]]

		# Other break conditions could be used here to leave the loop faster, but the speed benifit should be super
		# minimal so I'm leaving it as is for now
		for j in xrange(len(intervals) - 1):
			if intervals[j][1] <= start_time and intervals[j + 1][0] >= end_time:
				if not dry_run:
					self.call_classifications[self.cur_file_index][(start_time, end_time)] = classification
				return True
		return False

	def classification_chosen(self, classification):
		if self.segmenter.selected_segment is not None:
			if self.segmenter.selected_segment not in self.call_classifications[self.cur_file_index]:
				self.segmenter.cur_segment_classified()

			self.call_classifications[self.cur_file_index][self.segmenter.selected_segment] = classification

			self.segmenter.select_segment()
		elif self.spec_panel.x1 is not None:  # If no box is selected then do nothing and notify that nothing is done
			start_loc = min(self.spec_panel.x0, self.spec_panel.x1)
			end_loc = max(self.spec_panel.x0, self.spec_panel.x1)

			start_time, end_time = self.spec_panel.x_axis_scaler((start_loc, end_loc))

			if self.insert_classification(start_time, end_time, classification, dry_run=True):
				if self.segmenter.add_segment(start_time, end_time, True):
					self.call_classifications[self.cur_file_index][(start_time, end_time)] = classification

			# Remove the selection lines
			self.spec_panel.clear_lines()

	def remove_classification(self, segment):
		self.call_classifications[self.cur_file_index].pop(segment, None)

	def get_classification(self, time_coord):
		"""
		Gets the classification specified at a given time.

		:param time_coord: The time at which to find the classification
		:return: The classification, or None if no classification at the given time exists
		"""
		for times, classification in self.call_classifications[self.cur_file_index].items():
			if times[0] < time_coord < times[1]: ######boundaries don't belong to classification
				return classification
		return None

	def set_statusbar_values(self, time_coord):
		"""

		:param time_coord: The time coordinate (offset by the file's starting time)

		TODO:
		 - Have the values disappear when not on an element that indicates time values

		OPTIMIZATION NOTES:
			- This could be optimised, and would probably save a good chunk of computational power (but not really
			speed things up too much)
				- Avoid Search:
					- Store the upper and lower bounds for the segment (or lack of segment) currently hovered over by the mouse
					- Before iterating over the classifications, first check if the location is within the stored bounds,
					and if yes, there's no need for classification iteration.   If no, update the new bounds
				- Optimize Search:
					- Could do a binary search on the segments since they're in sorted order
		"""
		if time_coord is None:
			self.status_bar.SetStatusText("")
			self.status_bar.SetStatusText("", 1)
		else:
			self.status_bar.SetStatusText(
				"%.3f (%.3f)" % (time_coord, time_coord - self.cum_file_lens[self.cur_file_index]))

			cur_class = self.get_classification(time_coord)
			if cur_class is None:
				self.status_bar.SetStatusText("", 1)
			else:
				self.status_bar.SetStatusText(cur_class, 1)

	def prompt_and_set_file_index(self):
		max_index = len(self.data_list) - 1

		file_index_dlg = IntegerEntryDialog(
			self,
			"Enter the the index of the file you'd like to go to (with range [%d, %d])" % (0, max_index),
			min=0,
			max=max_index,
			title="File Index Specification",
			default=self.cur_file_index)

		file_index_dlg.ShowModal()

		self.goto_file_index(file_index_dlg.get_value())

	def export_as_csv(self, output_filename=None):
		"""
		TODO:
		 - Handle the situation where the file trying to be written already exists from a previous save
		"""
		if output_filename is None:
			output_filename = "%s\\segments_info.csv" % self.folder_name

		with open(output_filename, 'wb') as f:
			writer = csv.writer(f, delimiter=',')
			writer.writerow([
				"File",
				"File Index",
				"Sub File Index",
				"Call Index",
				"Abs. Start Time",
				"Abs. End Time",
				"Rel. Start Time",
				"Rel. End Time",
				"Call Length",
				"Call Type",
				"File Start Time"])

			call_counter = 0
			file_call_counter = defaultdict(int)
			for j, (file_index, seg, is_classified) in enumerate(self.segmenter):
				writer.writerow([
					self.file_list[file_index],
					file_index,
					file_call_counter[file_index],
					call_counter,
					seg[0],
					seg[1],
					seg[0] - self.cum_file_lens[file_index],
					seg[1] - self.cum_file_lens[file_index],
					seg[1] - seg[0],
					self.call_classifications[file_index][seg] if seg in self.call_classifications[file_index] else "",
					self.cum_file_lens[file_index]])

				call_counter += 1
				file_call_counter[file_index] += 1

	def load_from_csv(self, filename):
		with open(filename) as f:
			reader = csv.reader(f)

			#Skip the header
			reader.next()

			prev_file_index = -1
			for row in reader:
				row[1] = int(row[1])
				row[4] = float(row[4])
				row[5] = float(row[5])

				self.cur_file_index = row[1]
				self.segmenter.cur_file_index = row[1]

				if prev_file_index != row[1]:
					prev_file_index = row[1]
					self.segmenter.set_scaling(
						self.cum_file_lens[row[1]],
						self.cum_file_lens[row[1]] + (len(self.data_list[row[1]]) - 1) / self.samp_freq,
						True)

				has_classification = row[9] != ""

				if not self.segmenter.add_segment(row[4], row[5], has_classification, redraw=False):
					raise Exception("The segment (%s, %s) being imported is not valid!" % (row[4], row[5]))

				if has_classification:
					if not self.insert_classification(row[4], row[5], row[9]):
						raise Exception("Classifications being imported are not valid!")

			self.segmenter.canvas.draw()

	def zoom_into_segment(self, segment, file_index):
		"""
		Fully zooms into a segment.

		:param segment: The segment to zoom into as a tuple: (start_time, end_time)
		:param file_index: The index for the file where the segment is located
		"""
		if segment is not None:
			if self.cur_file_index != file_index:
				self.goto_file_index(file_index, redraw=False)

			self.segmenter.select_segment(segment, redraw=False)

			self.spec_panel.zoom(time_lims=segment)
			self.segmenter.zoom(*segment)

	def zoom_on_next_segment(self, forward=True, selected_segment=None):
		"""
		TODO:
		 - Make this so it's not horribly inefficient (was a test that turned into the actual implementation)
		"""
		if selected_segment is None:
			selected_segment = self.segmenter.selected_segment

		if selected_segment is not None:
			zoom_next = False
			for file_index, seg, _ in self.segmenter.__iter__(forward):
				if zoom_next:
					self.zoom_into_segment(seg, file_index)
					return
				if seg == self.segmenter.selected_segment:
					zoom_next = True

	def zoom_on_segment_index(self, index):
		for j, (file_index, seg, _) in enumerate(self.segmenter):
			if j == index:
				self.zoom_into_segment(seg, file_index)
				return

	def prompt_for_segment_zoom(self):
		max_index = len(list(self.segmenter)) - 1

		if max_index >= 0:
			call_index_dlg = IntegerEntryDialog(
				self,
				"Enter the the index of the call/segment you'd like to zoom in on (with range [%d, %d])" % (
				0, max_index),
				min=0,
				max=max_index,
				title="Segment Index Specification",
				default=self.cur_file_index)

			call_index_dlg.ShowModal()

			self.zoom_on_segment_index(call_index_dlg.get_value())
		else:
			wx.MessageBox("Segment zooming cannot be done if no segments have been defined!",
						  "Segment Zoom Warning", wx.OK, self)


if __name__ == "__main__":
	app = wx.App(False)
	frame = MainWindow(None, "Rat-Chat Internal Classification Tool")
	app.MainLoop()
