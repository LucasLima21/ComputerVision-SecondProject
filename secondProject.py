import numpy as np
import cv2


def grayScale(image): #converter para escala de cinza
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
def bgrScale(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def smoothing(image):
    return cv2.GaussianBlur(image, (9,9), 0)
    
def cannyMethodEdges(image): # para identificar a borda com canny, nao preciso da iamgem binarizada, nem suavizada!
    return cv2.Canny(image, 30, 200)

def binarization(image):
    _, binarized = cv2.threshold(smoothing(image), 100, 255, cv2.THRESH_BINARY)
    return binarized
    
def contours(image):
    binarized = binarization(image)
    contoured, _ = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contoured

def drawImageContours(image):
    copyGrayImage = np.copy(image)
    contoured = contours(copyGrayImage)
    cv2.drawContours(copyGrayImage, contoured, -1, (0, 0, 0), 2)
    cv2.imwrite('contoured.png', copyGrayImage)
    return copyGrayImage



def showResults(images):
    
    result = np.vstack([np.hstack([images[0], images[1], images[2]]), np.hstack([images[3], images[4], images[5]])])
    # result = np.concatenate((images[0], images[1]), axis=0)
    cv2.imshow('Resultados', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def startAlgorithm():
    image = cv2.imread('pikeman.jpg')
    image = image[::3, ::3] #diminui a imagem
    grayImage = grayScale(image)  
    
    showResults([grayImage, grayImage, smoothing(grayImage), 
    binarization(grayImage), cannyMethodEdges(grayImage), drawImageContours(grayImage)])
    


startAlgorithm()
