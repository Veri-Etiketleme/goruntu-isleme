from tkinter import filedialog
from detect_image import detecImage
from detect_video import detecVideo
from tkinter import *

root = Tk()
root.title("Object detection")

root.filename = filedialog.askopenfilename(initialdir='./..', title='choose image', filetypes=(('jpg', '*.jpg'), ('png', '*.png'), ('jpeg','*.jpeg'), ('mp4', '*.mp4'), ('mov', '*mov'), ('mkv', '*mkv')))

ext = (root.filename).split('.')[-1]

if ext in ['mp4', 'mov', 'mkv']:
    detecVideo(root.filename)
else:
    print(root.filename)
    root.destroy()
    detecImage(root.filename)

