# import os
#
#
#
#
# def func(path_img, path_lbl, pth_dst_img):
#     lst_img = os.listdir(path_img)
#     lst_txt = os.listdir(path_lbl)
#     c = 0
#     for i, k in enumerate(lst_txt):
#         if k[:-3]+"jpg" in lst_img:
#             #print(f"ITS_OK: {k[:-3]+'jpg'}")
#             pass
#         else:
#             print(f"NOT OK: {k[:-3]+'jpg'}")
#             c += 1
#             #os.rename(path_img+k[:-3]+"jpg", pth_dst_img+k[:-3]+"jpg")
#     print(c)
#
#
# func()


# import os
# import shutil
# import random
#
# src_images_dir = 'path/to/images'  # Путь к папке с изображениями
# src_labels_dir = 'path/to/labels'  # Путь к папке с аннотациями
# dst_images_dir = 'path/to/moved_images'  # Папка для перемещённых изображений
# dst_labels_dir = 'path/to/moved_labels'  # Папка для перемещённых аннотаций
# truck_class_id = '1'  # Класс грузовиков
# prob_remove = 0.5  # Вероятность переместить грузовик (50%)
#
# os.makedirs(dst_images_dir, exist_ok=True)
# os.makedirs(dst_labels_dir, exist_ok=True)
#
# for label_file in os.listdir(src_labels_dir):
#     if not label_file.endswith('.txt'):
#         continue
#     label_path = os.path.join(src_labels_dir, label_file)
#
#     with open(label_path, 'r') as f:
#         lines = f.readlines()
#
#     # Проверяем есть ли грузовик в аннотации
#     has_truck = any(line.startswith(truck_class_id + ' ') for line in lines)
#
#     if has_truck and random.random() < prob_remove:
#         # Перемещаем файл разметки
#         shutil.move(label_path, os.path.join(dst_labels_dir, label_file))
#         # Перемещаем соответствующее изображение (.jpg)
#         image_file = label_file.replace('.txt', '.jpg')
#         image_path = os.path.join(src_images_dir, image_file)
#         if os.path.exists(image_path):
#             shutil.move(image_path, os.path.join(dst_images_dir, image_file))


import os



def func(path_img, path_lbl, path_dest_lbl):
    lst_img = os.listdir(path_img)
    lst_lbl = os.listdir(path_lbl)

    for i, k in enumerate(lst_img):
        name_hash = k[:-3]
        if name_hash + "txt" in lst_lbl:
            os.rename(path_lbl+name_hash + "txt", path_dest_lbl + name_hash + "txt")


func()


