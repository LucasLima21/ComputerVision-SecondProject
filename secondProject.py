"""
Universidade do Estado do Amazonas - Escola Superior de Tecnologia
Alunos: Karla Félix, Lucas Lima, Victor Lopes
Visão Computacional
Segunda Atividade
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    copyGrayImage = grayScale(image)
    contoured = contours(copyGrayImage)
    cv2.drawContours(image, contoured, -1, (0, 255, 0), 2)
    cv2.imwrite('contoured.png', image)
    return image



def showResults(images):
    figure = plt.figure(figsize=(12, 6))
    titles = ['Original Image', 'Gray Scale', 'Gaussian Blur', "Binarization", "Canny", "Contours"]
    rows = 2
    columns = 3
    for i in range(rows * columns):
        plt.subplot(rows, columns, i+1)
        if(i == 0 or i == 5):
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(titles[i])

        else:
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            plt.title(titles[i])
        
    figure.tight_layout(pad=0.2)
    plt.show()
    

def startAlgorithm():
    image = cv2.imread('pikeman.jpg')
    # image = image[::3, ::3] #diminui a imagem
    copyOriginalImage = np.copy(image)
    grayImage = grayScale(image)  
    showResults([copyOriginalImage, grayImage, smoothing(grayImage), 
    binarization(grayImage), cannyMethodEdges(grayImage), drawImageContours(image)])
    

startAlgorithm()
