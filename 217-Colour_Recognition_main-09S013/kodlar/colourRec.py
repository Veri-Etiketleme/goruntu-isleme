import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

test_image = cv2.imread('colour.jpeg')

# print(plt.imshow(test_image))

index=["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('dataset/colors.csv', names=index, header=None)

print(csv.head())

clicked = False
r = g = b = xpos = ypos = 0

def recognize_color(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d&lt==minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

def mouse_click(x,y,event,param,flags):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos,clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = test_image[y,x]
        b = int(b)
        g = int(g)
        r = int(r)


cv2.namedWindow("Colour Recognition App")
cv2.setMouseCallback("Colour Recognition App", mouse_click)

while (1):
    cv2.imshow("Colour Recognition App", test_image)

    if (clicked):
        # cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle
        cv2.rectangle(test_image, (20, 20), (250, 60), (b, g, r), -1)

        # Creating text string to display( Color name and RGB values )
        text = recognize_color(r, g, b) + "R=" + str(r) + "G=" + str(g) + "B" + str(b)

        # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(test_image, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # For very light colours we will display text in black colour
        if (r + g + b >= 600):
            cv2.putText(test_image, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

        # Break the loop when user hits 'esc' key
        if cv2.waitKey(20) & 0xFF == 27:
            break

cv2.destroyAllWindows()

