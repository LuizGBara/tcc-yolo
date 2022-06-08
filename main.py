import numpy as np
import argparse
import time
import cv2
import os


def yoloScan(img, conf):
    yolopath = os.path.join('yolo-coco')
    #yolopath = os.path.join('yolo-fruits')
    weightsPath = os.path.sep.join([yolopath, "yolov3.weights"])
    configPath = os.path.sep.join([yolopath, "yolov3.cfg"])
    # weightsPath = ch = os.path.sep.join([yolopath, "yolov3_custom_last.weights"])
    # configPath = os.path.sep.join([yolopath, "yolov3_custom.cfg"])
    labelsPath = os.path.sep.join([yolopath, "coco.names"])
    #labelsPath = os.path.sep.join([yolopath, "obj.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    threshold = 0.6
    #passa para o yolo o endereço dos diretorios do arquivo de config e pesos
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # carrega img
    image = cv2.imread(img)
    #salva tamanhos da imagem
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    #tranforma imagem om blob para servir como input da rede
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    #inicia contador de tempo
    start = time.time()
    #chama funcao do yolo passando paramentro armazenados no .net
    layerOutputs = net.forward(ln)
    #para contador de timer
    end = time.time()
    #print do tempo
    print("YOLO levou {:.6f} seconds".format(end - start))
    # inicializando variaveis
    caixas = []
    certezas = []
    classes = []
    encontradas = {}
    for output in layerOutputs:
        #loop pegando cada deteccao
        for detection in output:
            # extrindo classes encontrada e nivel de certezas
            scores = detection[5:]
            classID = np.argmax(scores)
            certeza = scores[classID]
            #remove intens con certeza abaixo da determinada
            if certeza > conf:
                #determina o tamanho da caixa de acordo com o centro da imagem tamnhaso da deteccao
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                #utiliza centro e tamanho para determinar os cantos da caixa
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # gera listas com o valores calculados
                caixas.append([x, y, int(width), int(height)])
                certezas.append(float(certeza))
                classes.append(classID)
    for val in classes:
        if val not in encontradas:
            encontradas[LABELS[val]] = classes.count(val)
    print(str(encontradas))
    print(str(classes))

    # remove caixas sobrepostas
    idxs = cv2.dnn.NMSBoxes(caixas, certezas, conf, threshold)
    # se exeiste pelo menos uma deteçao
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extraindo cordenadas das caixas
            (x, y) = (caixas[i][0], caixas[i][1])
            (w, h) = (caixas[i][2], caixas[i][3])

            #fazendo crop da imagem somente com o caixa da deteccao
            crop_img = image[y:y + h, x:x + w]

            cv2.imwrite(os.path.join('imgs/' + img + '%d.jpg' % i), crop_img)

    #cv2.imwrite(os.path.join(img + '_mod.jpg'), image)
    return encontradas, img+'_mod.jpg'

if __name__ == '__main__':
    conf = 0
    img = yoloScan('unknown.png', conf)
