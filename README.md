# Rat-Call-Classification-Tool

## Description:
This tool is used for the manual classification and exploration of ultrasonic rat vocalizations, and has an intuitive user interface for doing this.  It's primary use case is to generate a set of manual classificiations of recordings of rat audio, so that machine learning models could be trained on the data.

## Machine Learning Classifier
In order to classify rat calls using Machine Learning, good data is required.  Unfortunately there's no abundance of manually classified rat audio, so to help mitigate this, a transfer learning approach is utilized.  Using a [pre-trained ResNet-50 network](https://tfhub.dev/google/humpback_whale/1) designed to identify whale sounds in underwater audio recorings as our source model, it outperforms any other approach we're aware of on the same limited dataset.

## Raspy Classification Tool Quick-Start Guide 
This section is intended to make it easier to get started using this tool, but it only covers a portion of the things that can be done.  After going through this section, exploring what else you can do with this software should be easy!

### Tool Anatomy:

![Image Of Tool](https://i.imgur.com/hsyHqBw.png)

 | *Name* | *Explanation* 
:---: | :---: | ---
A | Spectrogram | A spectrogram of the data contained in the currently viewed .wav file (see item G) 
B | Bounding Lines | The two vertical red lines indicate the boundaries for the segment (in time) currently selected.  It can be specified by clicking, dragging, then releasing the cursor on the spectrogram, or by clicking an existing segment in the Segmenter (item C). 
C | Segmenter | A view of the classifications which have been made already.  A segment can be selected by clicking on one of the colored blocks, and a selected segment can be identified by becoming red, and by the bounding lines being moved to the segments boundaries. 
D | Call Buttons | Buttons used to classify a selected segment (either selected through the Bounding Lines or the Segmenter) 
E | Time Viewer | Shows the time at the location where the cursor (mouse) currently is.  It’s format is as follows:  *(time_relative_to_entire_file_start) time_relative_to_viewed_file_start*
F | Classification Viewer | Shows the classification of the call where the mouse currently is (or nothing if not hovering over a classified call) 
G | Current File | Displays the .wav file currently being viewed. 


### First Use Step-By-Step Guide:

1. Click on the ‘File’ menu on the top left of the window, then select ‘Open New File’ 
2. Select a .wav file to open, then specify the sampling rate.  If the file is large, it will split it into multiple chunks, then load the spectrogram for the first one (will load the entire spectrogram if it’s a small file). 
3. Click on the spectrogram, drag your mouse to the left or right, then release the click.  You should now see two vertical red lines. 
4. Click on the call button labeled ‘Flat’.  You should now see a blue box under the spectrogram (in the Segmenter).  To see what it’s classification is, hover over it (or above it in the spectrogram) and the classification (Flat) should be seen in the center bottom of the window (in the Classification Viewer). 
5. Now select the classification by clicking on the blue box in the Segmenter, then press the delete key to remove it. 
6. Go back to the ‘File’ menu, then select ‘Run Classification’. 
7. After it’s done running, you should see the Segmenter become populated with various call classifications.  Now select one of them (not the first call), by clicking on one of the blue boxes in the Segmenter. 
8. Press the ‘z’ key, which will zoom into the selected call.   
9. Press Ctrl + left arrow key, which will then zoom in on the previous call. 
10. Zoom out, either by using your mouse’s scroll wheel, or by pressing the up arrow key. 
11. Press the right arrow key to view the next portion of the split .wav file.bode  
12. Press The X button  to close the window then save your work when prompted. 
13. The classification information (and various other relevant data) can then be found in the same directory as the original .wav file selected, within a folder with the same name as the original selected .wav file 

### Hotkey Cheat-Sheet:

*Hotkey* | *What it does* 
:---: | ---
Ctrl + S | Saves the classification data as a CSV (overriding previously saved data) 
Scroll Wheel | Zooms into the spectrogram (centered wherever your mouse is located) 
Up/Down Arrows | Zooms into the spectrogram (centered at the center of the spectrogram) 
Left/Right Arrows | Moves between the files (generated when the initial .wav file is split) 
Z | Zooms into the segment/call currently selected 
Ctrl + Left/Right Arrows  | Zooms into the next or previous segment/call (relative to the call currently selected) 
Delete | Removes the classification currently selected 

## Data:
All the manually classified data used in this project is publically available in the following repository: [Rat-Vocalization-Data](https://github.com/Mide-RASP/Rat-Vocalization-Data)

## Support:

This work is supported by the US Army Medical Research and Materiel Command under Contract Number W81XWH18C014224.

The views, opinions and/or findings contained in this report are those of the author(s) and should not be construed as an official Department of the Army position, policy or decision unless so designated by other documentation

In conducting research using animals, the investigator(s) adhered to the Animal Welfare Act Regulations and other Federal statutes relating to animals and experiments involving animals and the principles set forth in the current version of the Guide for Care and Use of Laboratory Animals, National Research Council.
