import numpy as np

def create_anchors():

    anchors = []

    a1 = [0.15, 0.2]
    a2 = [0.2, 0.3]
    a3 = [0.3, 0.6]
    a4 = [0.4, 0.75]
    a5 = [0.5, 0.85]

    for y in np.linspace(0, 1, 14):
        for x in np.linspace(0, 1, 14):
            anchors.append((y-a1[1]/2, x-a1[0]/2, y+a1[1]/2, x+a1[0]/2))
            anchors.append((y-a2[1]/2, x-a2[0]/2, y+a2[1]/2, x+a2[0]/2))
            anchors.append((y-a3[1]/2, x-a3[0]/2, y+a3[1]/2, x+a3[0]/2))
            anchors.append((y-a4[1]/2, x-a4[0]/2, y+a4[1]/2, x+a4[0]/2))
            anchors.append((y-a5[1]/2, x-a5[0]/2, y+a5[1]/2, x+a5[0]/2))

    return np.array(anchors)
