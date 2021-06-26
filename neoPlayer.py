import os
import numpy as np
import pyaogmaneo as neo
import cv2

enc = neo.ImageEncoder()
h = neo.Hierarchy()

enc.initFromFile("enc.oenc")
h.initFromFile("h.ohr")

imgSize = enc.getVisibleSize(0)
actionSize = h.getInputSize(1)

print("Image size: {0}".format(imgSize))

actionCIs = (actionSize[0] * actionSize[1]) * [ actionSize[2] // 2 ]

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 512, 512)

while True:
    enc.reconstruct(h.getPredictionCIs(0))

    img = np.array(enc.getReconstruction(0)).astype(np.uint8).reshape(imgSize)
    frame = np.swapaxes(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0, 1)

    cv2.imshow("Result", frame)

    k = cv2.waitKey(1)

    if k & 0xFF == ord('q'):
        break

    actionCIs[0] = 1

    if k & 0xFF == ord('a'):
        actionCIs[0] = 0
    elif k & 0xFF == ord('d'):
        actionCIs[0] = 2

    h.step([ h.getPredictionCIs(0), actionCIs ], False)

