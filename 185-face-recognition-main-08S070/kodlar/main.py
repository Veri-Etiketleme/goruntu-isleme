from PyQt4 import QtCore, QtGui
import qdarkstyle
from threading import Thread
from collections import deque
from datetime import datetime
import time
import sys
import cv2
import imutils
import time
import os
import warnings

# 人臉偵測 face_recognize.py
from face_recognize import  face_rec

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get its own network to predict face_id
class CameraWidget(QtGui.QWidget):
    def __init__(self, width, height, stream_link, face_net, aspect_ratio=False, parent=None, deque_size=1):
        super(CameraWidget, self).__init__(parent)

        # face_recognition物件，人臉偵測 → 人臉辨識 → (口罩辨識 → 年紀偵測)
        self.face_net = face_net
        
        # Initialize deque used to store frames read from the stream
        self.deque = deque(maxlen=deque_size)

        # slight offset is needed since PyQt layouts have a built in padding, so add offset to counter the padding 
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio

        # The number of camera
        self.camera_stream_link = stream_link

        # Flag to check if camera is valid/working
        self.online = False
        self.capture = None
        self.video_frame = QtGui.QLabel()

        # Camera start streaming
        self.load_network_stream()

        # Start frame grabbing at background
        self.get_frame_thread = Thread(target=self.get_frame)
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_frame)
        self.timer.start(.5)
       
    def load_network_stream(self):
        """Verifies stream link and open new stream if valid"""

        def load_network_stream_thread():
            if self.verify_network_stream(self.camera_stream_link):
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True
                print('Camera_{} starts streaming...'.format(self.camera_stream_link))
            else:
                print("Camera_{} is offline!".format(self.camera_stream_link))
        
        self.start_streaming = Thread(target=load_network_stream_thread)
        self.start_streaming.setDaemon(True) # When main thread close, this thread will close too.
        self.start_streaming.start()

    def verify_network_stream(self, link):
        """Attempts to receive a frame from given link"""

        cap = cv2.VideoCapture(link)
        if cap.isOpened():
            cap.release()
            return True
        else:
            return False

    def get_frame(self):
        """Reads frame push into deque"""

        while True:
            try:
                if self.capture.isOpened() and self.online:
                    # Read next frame from stream and insert into deque
                    status, frame = self.capture.read()
                    if status:
                        self.deque.append(frame)
                    else:
                        self.capture.release()
                        self.online = False
                else:
                    # Attempt to reconnect
                    print('Attempting to reconnect camera_{}'.format(self.camera_stream_link))
                    self.load_network_stream()
                    self.pause(2)
                self.pause(.001)
            except AttributeError:
                pass

    def pause(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtGui.QApplication.processEvents()

    def set_frame(self):
        """Sets pixmap image to video frame"""

        if not self.online:
            self.pause(1)
            return
        
        if self.deque and self.online:
            # Grab latest frame
            try:
                frame = self.deque.pop()
            except IndexError:
                pass

            # True : Resize to fit the screen width  &&  Keep frame aspect ratio(長寬比)
            # False: Force resize
            if self.maintain_aspect_ratio:
                frame = imutils.resize(frame, width=self.screen_width)
            else:
                frame = cv2.resize(frame, (self.screen_width, self.screen_height))
            
            # Face detection and recognition
            self.frame = self.face_net.detect_img(frame)
            
            # Add timestamp at the right top corner
            cv2.rectangle(self.frame, (self.screen_width-190,0), (self.screen_width,50), color=(0,0,0), thickness=-1)
            cv2.putText(self.frame, datetime.now().strftime('%H:%M:%S'), (self.screen_width-185,37), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), lineType=cv2.LINE_AA)

            # Convert to pixmap and set to video frame
            self.img = QtGui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            self.pix = QtGui.QPixmap.fromImage(self.img)
            self.video_frame.setPixmap(self.pix)
            
    def get_video_frame(self):
        return self.video_frame
    
# Create camera widget
def appendCamera(screen_width, screen_height, ml, face_net):
    # 調整攝影機數量，0、1是webcam 下面camera2是rtsp寫法(須自己更改成符合的)
    camera0 = 0
    # camera1 = 1
    # username = 'Your camera username!'
    # password = 'Your camera password!'
    '''
    camera2 = 'rtsp://{}:{}@192.168.1.47:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    camera3 = 'rtsp://{}:{}@192.168.1.40:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    camera4 = 'rtsp://{}:{}@192.168.1.44:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    camera5 = 'rtsp://{}:{}@192.168.1.42:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    camera6 = 'rtsp://{}:{}@192.168.1.46:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    camera7 = 'rtsp://{}:{}@192.168.1.41:554/cam/realmonitor?channel=1&subtype=0'.format(username, password)
    '''

    # Create camera widgets
    print('Creating Camera Widgets...')
    zero = CameraWidget(screen_width//2, screen_height//2, camera0, face_net)
    # one = CameraWidget(screen_width//2, screen_height//2, camera1)
    '''
    two = CameraWidget(screen_width//3, screen_height//3, camera2)
    three = CameraWidget(screen_width//3, screen_height//3, camera3)
    four = CameraWidget(screen_width//3, screen_height//3, camera4)
    five = CameraWidget(screen_width//3, screen_height//3, camera5)
    six = CameraWidget(screen_width//3, screen_height//3, camera6)
    seven = CameraWidget(screen_width//3, screen_height//3, camera7)
    '''
    
    # 將設定完的視窗大小以及鏡頭畫面做順序的安排
    print('Adding widgets to layout...')
    ml.addWidget(zero.get_video_frame(),0,0,1,1)
    # ml.addWidget(one.get_video_frame(),0,1,1,1)
    '''
    ml.addWidget(two.get_video_frame(),0,2,1,1)
    ml.addWidget(three.get_video_frame(),1,0,1,1)
    ml.addWidget(four.get_video_frame(),1,1,1,1)
    ml.addWidget(five.get_video_frame(),1,2,1,1)
    ml.addWidget(six.get_video_frame(),2,0,1,1)
    ml.addWidget(seven.get_video_frame(),2,1,1,1)
    '''

# Action method, Exit program 
def exit_application(): 
    print("Close the app.")
    sys.exit(1)

# Create application windows, append widget, add event listener
def main():
    # face_recognition物件，人臉偵測 → 人臉辨識 → 口罩辨識 → 年紀偵測
    face_net = face_rec()

    app = QtGui.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt())
    app.setStyle(QtGui.QStyleFactory.create("Cleanlooks"))

    mw = QtGui.QMainWindow()
    mw.setWindowTitle('Camera GUI')
    mw.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint) # Hide UI border frame, always on top of desktop
    # mw.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint) # Gives the window a thin dialog border on Windows.

    btn = QtGui.QPushButton('Quit')
    btn.setToolTip('離開')
    btn.clicked.connect(exit_application)

    cw = QtGui.QWidget()
    ml = QtGui.QGridLayout()
    cw.setLayout(ml)
    mw.setCentralWidget(cw)
    mw.showMaximized()

    # Dynamically determine screen width/height
    screen_width = QtGui.QApplication.desktop().screenGeometry().width()
    screen_height = QtGui.QApplication.desktop().screenGeometry().height()
    print("螢幕大小: {}x{}".format(screen_width, screen_height))
    
    # Append Camera Widget
    appendCamera(screen_width, screen_height, ml, face_net)

    # Add leaving button to layout
    ml.addWidget(btn)
    
    # Add keyboard listener
    QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Q'), mw, exit_application)
    mw.show()
    
    if(sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec_()

if __name__ == '__main__':
    main()