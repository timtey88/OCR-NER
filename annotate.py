import cv2
import random

scale = 0.5
circles = []
counter = 0
counter2 = 0
point1 = ()
point2 = ()
myPoints = []
myColor = []

def mousePoints(event, x, y, flags, params):
    global counter, point1, point2, counter2, circles, myColor
    if event == cv2.EVENT_LBUTTONDOWN:
        if counter == 0:
            point1 = (int(x // scale), int(y // scale))
            counter += 1
            myColor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif counter == 1:
            point2 = (int(x // scale), int(y // scale))
            entity_type = input('Enter entity type: ')
            entity_name = input('Enter entity name: ')
            myPoints.append([point1, point2, entity_type, entity_name])
            counter = 0
        circles.append([x, y, myColor])
        counter2 += 1

img = cv2.imread('dataset/custom_declaration_2.png')
img = cv2.resize(img, (0, 0), None, scale, scale)

while True:
    # Display points
    for x, y, color in circles:
        cv2.circle(img, (x, y), 3, color, cv2.FILLED)
    cv2.imshow("Original Image", img)
    cv2.setMouseCallback("Original Image", mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(myPoints)
        break

cv2.destroyAllWindows()

"""
ROI for custom_declaration_1.png
        [510, 42, 628, 68, 'text', 'date'],
        [512, 72, 626, 94, 'text', 'flight_number'],
        [188, 234, 336, 250, 'text', 'name'],
        [76, 346, 92, 360, 'box', 'no_goods'],
        [74, 382, 90, 400, 'box', 'goods'],
        [76, 444, 90, 460, 'box', 'tobacco'],
        [74, 464, 90, 480, 'box', 'alcohol'],
        [76, 496, 90, 514, 'box', 'medical'],
        [74, 518, 88, 532, 'box', 'others'],
        [308, 442, 458, 456, 'text', 'tobacco'],
        [310, 474, 458, 488, 'text', 'alcohol'],
        [308, 492, 458, 510, 'text', 'medical'],
        [310, 510, 458, 530, 'text', 'others'],
        [76, 594, 90, 612, 'box', 'no_goods'],
        [76, 628, 92, 642, 'box', 'goods'],
        [76, 692, 90, 708, 'box', 'tobacco'],
        [76, 712, 92, 730, 'box', 'alcohol'],
        [76, 744, 92, 760, 'box', 'medical'],
        [76, 766, 90, 780, 'box', 'others'],
        [312, 688, 458, 704, 'text', 'tobacco'],
        [308, 722, 458, 736, 'text', 'alcohol'],
        [310, 740, 458, 756, 'text', 'medical'],
        [310, 756, 460, 776, 'text', 'others']
"""
"""
ROI for custom_declaration_2.png
        [(220, 232), (638, 252), 'text', 'name'], 
        [(218, 314), (288, 332), 'text', 'date'], 
        [(568, 310), (654, 334), 'text', 'flight_number']
"""