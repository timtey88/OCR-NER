import cv2

image = img_cv.copy()
level = 'page'

for _, x, y, w, h, c in df[['level', 'left', 'top', 'width', 'height', 'conf']].values:
    if level == 'page':
        if 1 == 1:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        else:
            continue
    elif level == 'block':
        if 1 == 2:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            continue
    elif level == 'para':
        if 1 == 3:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            continue
    elif level == 'line':
        if 1 == 4:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            continue
    elif level == 'word':
        if 1 == 5:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            continue

cv2.imshow("bounding box", image)
cv2.waitKey()
cv2.destroyAllWindows()
