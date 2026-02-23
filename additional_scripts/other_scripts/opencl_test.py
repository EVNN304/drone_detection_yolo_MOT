import cv2

# Загрузка изображения с использованием UMat
image = cv2.imread('3.jpg', cv2.IMREAD_COLOR)
umat_image = cv2.UMat(image)

# Применение фильтра Гаусса с использованием UMat
blurred_image = cv2.GaussianBlur(umat_image, (5, 5), 0)

# Преобразование обратно в Mat для отображения или сохранения
result_image = blurred_image.get()

# Отображение результата
cv2.imshow('Blurred Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
