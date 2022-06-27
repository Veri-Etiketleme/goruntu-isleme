![Banner_IA2](images/banner-dimensiones.png)

# Real-time Face Mask Detection


La pandemia del COVID-19 inició en el año 2019 y ha llegado a afectar gravemente a la mayoría de países, perjudicando nuestra vida cotidiana, interrumpiendo el comercio y movimientos mundiales; usar una mascarilla se ha convertido en una normalidad debido a que según estudios realizados se ha llegado a comprobar que la propagación de este virus es por del canal de aire, por ello el uso de esta es indispensable pero mantener el control en sitios densamente poblados se requeriría de mucho personal para mantener el control lo cual sería bastante costoso es por esto se propone realizar la detección del tapabocas por medio de Inteligencia Artificial y cámaras de seguridad.[Vídeo](https://www.youtube.com/watch?v=_bMRN4Xwu6M)   - [Diapositivas](https://www.canva.com/design/DAEYHR-TSQ8/L8Su5uWWSTb6DLIR-of6ow/view?utm_content=DAEYHR-TSQ8&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink)

### Dataset
Se hizo uso de un dataset disponible en kaggle ["Face Mask Detection Dataset"](https://www.kaggle.com/omkargurav/face-mask-dataset?select=data) el cual consta de 7553 imágenes RGB en 2 carpetas con máscara y sin máscara. Las imágenes se nombran como etiqueta con máscara y sin máscara. Las imágenes de rostros con máscara son 3725 y las imágenes de rostros sin máscara son 3828.

![imagendata](images/imagendata.png)

### Clasificación

1. [Convolutional  Neural Network](Classification.ipynb) : Se implementaron dos redes CNN, la primera de ellas recibe las imagenes en RGB y un tamaño de 50 X 50, y la segunda recibe las imagenes en grises con un tamaño de 50 X 50.
2. [Transfer Learning](Classification.ipynb) : Se entreo con la Resnet 50, VGG -19,  MobileNetV2, teniento mejores resultados con la Resnet 50 y la VGG19, la primera de estas se entreno con imagenes de 50 X 50 y la segunda con imagenes de 100 X 100.
3. [Haar Features](Haar_features.ipynb): Se calcularon las haar Features para 200 imagenes de mascarilla como positivas y 200 imagenes de paisajes como negativas, posteriormente se hizo la clasificación con Adaboost.
3. [Detección de mascarillas en tiempo real](ClassificationRealTime.ipynb): la primera tarea fue detectar rostros en cada fotograma del video para ello utilizamos el clasificador Haarcascade de OpenCV para la detección de rostros.Posteriormente a cada uno de los rostros le pasabamos la MobileNetV2 para la clasificación.

### Validación 

Para la validación se hizo uso de otro dataset de kaggle [Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection?fbclid=IwAR1Iz0mqagxjPcDUadYDA4Hj9uD2bSLhDw2dvXIyibxS6gns3wAueLr2XYY), este dataset cuenta con 853 imágenes y tres clases con mascarilla, sin mascarilla y mascarilla usada incorrectamente, la cual no se tomó en cuenta.

![imagenvalidation](images/validate_data.png)


### Resultados

Los mejores resultados se obtuvieron con la red Resnet50 y la MobileNetV2, por ello se decidió usar la MobileNetV2 para la detección en tiempo real debido a que esta aprende menos parametros lo que la hace más optima para ser implementada en sitios públicos o en cualquier dispositivo.


![resulst](images/video__1_.gif)
