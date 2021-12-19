import numpy as np
import cv2
from time import sleep
from PIL import Image
import matplotlib.pyplot as plt
import glob

def main():
    FClass = sorted(glob.glob("/home/kaneko/models/research/deeplab/IoUtestimg/Class/*"))
    Fimg = sorted(glob.glob("/home/kaneko/models/research/deeplab/IoUtestimg/1000/*"))
    Sum = 0
    for i in range(0,15):
        print(FClass[i])
        print(Fimg[i])
        image = cv2.imread(Fimg[i]) # ファイル読み込み
        image1 = cv2.imread(FClass[i]) # ファイル読み込み
        image2 = cv2.resize(image1, dsize=(513, 384))
        Sum = Sum + sub(image, image2)
        print(Sum)
        print(i+1)
   
    MIoU = Sum / 15
    print(MIoU)   

def sub(image, image2):

    # BGRでの色抽出
#af　茎の色
    bgrLower = np.array([0, 100, 0])    # 抽出する色の下限
    bgrUpper = np.array([50, 150, 50])    # 抽出する色の上限
    bgrResult1 = bgrExtraction(image, bgrLower, bgrUpper)
    sleep(1)
#af 葉っぱの色
#    bgrLower1 = np.array([0, 0, 120])    # 抽出する色の下限
#    bgrUpper1 = np.array([50, 50, 140])    # 抽出する色の上限
#    bgrResult = bgrExtraction(image, bgrLower1, bgrUpper1)
#    cv2.imshow('BGR_test2', bgrResult)
#    sleep(1)

    bgrLower = np.array([200, 0, 0])    # 抽出する色の下限
    bgrUpper = np.array([255, 50, 50])    # 抽出する色の上限
    bgrResult2 = bgrExtraction(image2, bgrLower, bgrUpper)
    sleep(1)


    blended = cv2.addWeighted(src1=bgrResult1,alpha=0.7,src2=bgrResult2,beta=0.3,gamma=0)
#    cv2.imshow('1 + 2', blended)

    hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
    bin_img = cv2.inRange(hsv, (0, 10, 0), (255, 255, 255))
#    cv2.imshow('1mask OR 2mask', bin_img) 
    
    hsv = cv2.cvtColor(bgrResult1, cv2.COLOR_BGR2HSV)
    bin_img2 = cv2.inRange(hsv, (0, 10, 0), (255, 255, 255))
#    cv2.imshow('1mask', bin_img2)

    hsv = cv2.cvtColor(bgrResult2, cv2.COLOR_BGR2HSV)
    bin_img3 = cv2.inRange(hsv, (0, 10, 0), (255, 255, 255))
#    cv2.imshow('2mask', bin_img3)

    img_msk1 = cv2.cvtColor(bgrResult1, cv2.COLOR_BGR2GRAY)
    img_msk2 = cv2.cvtColor(bgrResult2, cv2.COLOR_BGR2GRAY) 
    img_AND = cv2.bitwise_and(img_msk1, img_msk2)
#    cv2.imshow('1mask AND 2mask', img_AND)   
    
   
    th,img_up = cv2.threshold(img_AND, 1, 255, cv2.THRESH_BINARY) 
    
#    cv2.imshow('1mask AND 2mask 2', img_up)


    pixel_number = np.size(img_up)
    pixel_sum = np.sum(img_up)
    white_pixel_number1 = pixel_sum/255
    print("Area of Intersection(AND) ピクセル数",white_pixel_number1)
    
    pixel_number = np.size(bin_img)
    pixel_sum = np.sum(bin_img)
    white_pixel_number2 = pixel_sum/255
    print("Area of Union(OR) ピクセル数",white_pixel_number2)

    IoU = white_pixel_number1 / white_pixel_number2
    print("IoU", IoU)
    return IoU
  
  #  cv2.waitKey(0)
  #  cv2.destroyAllWindows()



# BGRで特定の色を抽出する関数
def bgrExtraction(image, bgrLower, bgrUpper):
    img_mask = cv2.inRange(image, bgrLower, bgrUpper) # BGRからマスクを作成
    result = cv2.bitwise_and(image, image, mask=img_mask) # 元画像とマスクを合成
    return result

if __name__ == '__main__':
    main()
