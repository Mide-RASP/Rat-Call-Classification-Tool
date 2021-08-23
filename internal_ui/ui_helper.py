from __future__ import division

import wx
from wx.lib.intctrl import IntCtrl
from wx.lib.agw import floatspin

class IntegerEntryDialog(wx.Dialog):
	def __init__(self, parent, text, min=None, max=None, default=0, title="", id=-1, **kwargs):
		"""
		:param text: The text to display in the dialog box (usually explaining what input is desired)
		:param min: The minimum value to accept, or None for an unbounded minimum (inclusive)
		:param max: The maximum value to accept, or None for an unbounded maximum (inclusive, how wxPython does it)
		:param default: The default value for the dialog
		:param title: The title for the Dialog box
		"""
		wx.Dialog.__init__(self, parent, id, title, size=(400, 125), **kwargs)

		self.min = float("-inf") if min is None else min
		self.max = float("inf") if max is None else max

		self.label = wx.StaticText(self, wx.ID_ANY, text)

		self.int_control = IntCtrl(self, -1, default, min=min, max=max)

		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.label, 6, wx.EXPAND | wx.ALL, border=10)
		sizer.Add(self.int_control, 1, wx.EXPAND| wx.ALL, border=10)

		self.SetSizer(sizer)

		self.Bind(wx.EVT_CHAR_HOOK, self.on_key_press)

		self.Centre()

	def on_key_press(self, event):
		"""
		TODO:
		 - Deal with the issues involving wxPython's IntCtrl (e.g. pressing ctrl+backspace)
		"""
		if event.GetKeyCode() == wx.WXK_RETURN or event.GetKeyCode() == wx.WXK_NUMPAD_ENTER:
			if self.min <= self.get_value() <= self.max:
				self.Close()
			else:
				self.label.SetLabel(self.label.GetLabel() + " (Double check boundaries: [%s, %s])"%(self.min, self.max))
				self.label.Wrap(370) #Should be 380 but that's not visually consistant
		else:
			event.Skip()

	def get_value(self):
		return self.int_control.GetValue()


class YesNoDialog(wx.Dialog):

	NO_BUTTON_ID = wx.NewId()

	def __init__(self, parent, yes_fn, no_fn, text, title="", id=-1, **kwargs):
		"""
		:param yes_fn: A function taking no parameters, which is run if the "Yes" button is pressed.  It runs directly
		prior to the dialog closing
		:param no_fn: The same as yes_fn but for the "No" button
		:param text: The text to display in the dialog box (usually explaining what input is desired)
		:param title: The title for the Dialog box
		"""
		wx.Dialog.__init__(self, parent, id, title, size=(250, 125), **kwargs)

		label = wx.StaticText(self, wx.ID_ANY, text)

		yes_button = wx.Button(self, wx.ID_OK, "Yes")
		no_button = wx.Button(self, self.NO_BUTTON_ID, "No")

		self.Bind(wx.EVT_BUTTON, self.callback_creator(yes_fn, wx.ID_OK), id=wx.ID_OK)
		self.Bind(wx.EVT_BUTTON, self.callback_creator(no_fn, wx.ID_OK), id=self.NO_BUTTON_ID)

		operation_aborted_callback = self.callback_creator(lambda: None)
		self.Bind(wx.EVT_CLOSE, operation_aborted_callback)
		self.Bind(wx.EVT_CHAR_HOOK,
				  lambda e: operation_aborted_callback(e) if e.GetKeyCode() == wx.WXK_ESCAPE else e.Skip())

		sizer1 = wx.BoxSizer(wx.HORIZONTAL)
		sizer1.Add(yes_button, 1, wx.EXPAND | wx.ALL, border=10)
		sizer1.Add(no_button, 1, wx.EXPAND | wx.ALL, border=10)

		sizer2 = wx.BoxSizer(wx.VERTICAL)
		sizer2.Add(label, 6, wx.EXPAND | wx.ALL, border=10)
		sizer2.Add(sizer1, 1, wx.EXPAND)

		self.SetSizer(sizer2)

		self.Centre()

	def callback_creator(self, fn, end_modal_val=wx.ID_ABORT):
		def callback(e):
			fn()
			self.EndModal(end_modal_val)
		return callback


class ColormapMinMaxDialog(wx.Dialog):
	def __init__(self, parent, cur_min, cur_max, title="Colormap Min/Max Selection", num_digits=3, id=-1, **kwargs):
		wx.Dialog.__init__(self, parent, id, title, size=(300, 125), **kwargs)

		start_label = wx.StaticText(self, wx.ID_ANY, "Min Colormap Value")
		end_label = wx.StaticText(self, wx.ID_ANY, "Max Colormap Value")

		min_spin_val = -1e4
		max_spin_val = 1e4
		spin_increment = 10 ** - num_digits
		self.start_float_spin = floatspin.FloatSpin(self, -1, pos=(50, 50), min_val=min_spin_val, max_val=max_spin_val,
											   increment=spin_increment, value=cur_min, agwStyle=floatspin.FS_LEFT)
		self.end_float_spin = floatspin.FloatSpin(self, -1, pos=(50, 50), min_val=min_spin_val, max_val=max_spin_val,
											 increment=spin_increment, value=cur_max, agwStyle=floatspin.FS_LEFT)

		for spin in (self.start_float_spin, self.end_float_spin):
			spin.SetFormat("%f")
			spin.SetDigits(num_digits)

		save_button = wx.Button(self, wx.ID_OK, "Save")

		start_sizer = wx.BoxSizer(wx.VERTICAL)
		start_sizer.Add(start_label, 0, wx.CENTER)
		start_sizer.Add(self.start_float_spin, 0, wx.CENTER | wx.EXPAND)

		end_sizer = wx.BoxSizer(wx.VERTICAL)
		end_sizer.Add(end_label, 0, wx.CENTER)
		end_sizer.Add(self.end_float_spin, 0, wx.CENTER | wx.EXPAND)

		start_end_sizer = wx.BoxSizer(wx.HORIZONTAL)
		start_end_sizer.Add(start_sizer, 0, wx.CENTER)
		start_end_sizer.AddSpacer(20)
		start_end_sizer.Add(end_sizer, 0, wx.CENTER)

		main_sizer = wx.BoxSizer(wx.VERTICAL)
		main_sizer.AddSpacer(10)
		main_sizer.Add(start_end_sizer, 0, wx.CENTER)
		main_sizer.AddSpacer(10)
		main_sizer.Add(save_button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, border=26)
		main_sizer.AddSpacer(10)

		end_modal_fn = lambda e: self.EndModal(wx.ID_ABORT)
		self.Bind(wx.EVT_CLOSE, end_modal_fn)
		self.Bind(wx.EVT_CHAR_HOOK, lambda e: end_modal_fn(None) if e.GetKeyCode() == wx.WXK_ESCAPE else e.Skip())

		self.SetSizer(main_sizer)

		self.Centre()

	def get_min_and_max(self):
		return self.start_float_spin.GetValue(), self.end_float_spin.GetValue()
