import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = [] # objekti koje ce da prepoznaje
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# print(classes)

def detImage(img):
    height, width, _ = img.shape

    # prebacujemo sliku u format koji odgovara algoritmu
    # treba da se prebaci u okvire 416x416
    # vrednosti svakog piksela se normalizuju - deli se sa 255

    # cv2.dnn.blobFromImage(image, scaleFactor, size, mean, swapRB, crop)
    blob = cv2.dnn.blobFromImage(img, 1/255,(416, 416), (0,0,0), swapRB = True, crop = False)

    # RGB kanali
    # for b in blob:
        # for n, img_blob in enumerate(b):
            # cv2.imshow(str(n), img_blob)

    # postavljamo blob slike na ulaz
    net.setInput(blob)


    output_layer_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layer_names)

    boxes = [] # okviri
    confidecnes = [] # sigurnost s kojom je objkat na slici
    class_ids = [] # predvidjeni objekat

    for output in layerOutputs: 
        for detection in output:
            # svaka detekcija ima 85 parametara
            # prva 4 elementa oznacavaju okvir, x_coord_centra, y_coord_centra, width, height, vrednosti su normalizovane
            # peti sadrzi pouzadnost
            # ostalih 80 su predikcije klasa - verovatnoca svake
            scores = detection[5:] # verovatnoca svake klase
            class_id = np.argmax(scores) # nadjemo na kom mestu je najveca verovatnoca
            confidence = scores[class_id] # gledamo koja je verovatnoca njaveca
            if confidence > 0.5: # ako je verovatnoca veca od ovoga onda nam je objekat zanimljiv
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # gornji levi ugao
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidecnes.append((float(confidence)))
                class_ids.append(class_id)

    # print(len(boxes))
    indexes = cv2.dnn.NMSBoxes(boxes, confidecnes, 0.5, 0.4) # uklanja redudantne okvire
    # print(indexes.flatten())

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 2)) # random boja za svvaki okvir

    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]]) + " " + str(round(confidecnes[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 5)
        cv2.putText(img, label, (x, y+20), font, 2, colors[i], 2)

    cv2.imshow('Image', img)
    
def detecImage(image_path):
    img = cv2.imread(image_path)
    detImage(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
