import numpy as np
from multiprocessing import shared_memory

class SharedFrameBuffer:
    """Двойной буфер для одного кадра (создается в главном процессе)"""
    def __init__(self, frame_h, frame_w, name_prefix="shm_frame"):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.frame_size = frame_h * frame_w * 3
        self.shm_0 = shared_memory.SharedMemory(create=True, size=self.frame_size)
        self.shm_1 = shared_memory.SharedMemory(create=True, size=self.frame_size)
        self.name_0 = self.shm_0.name
        self.name_1 = self.shm_1.name
        print(f"✅ SharedFrameBuffer создан: {frame_w}x{frame_h} ({self.frame_size/1024/1024:.1f} МБ)")

    def close(self):
        try:
            self.shm_0.close(); self.shm_0.unlink()
            self.shm_1.close(); self.shm_1.unlink()
        except Exception as e:
            print(f"⚠️ Ошибка освобождения shm: {e}")

class SharedFrameWriter:
    """Писатель для одного кадра (используется в дочерних процессах)"""
    def __init__(self, frame_h, frame_w, shm_name_0, shm_name_1, free_queue, ready_queue):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.shm_0 = shared_memory.SharedMemory(name=shm_name_0)
        self.shm_1 = shared_memory.SharedMemory(name=shm_name_1)
        self.free_queue = free_queue
        self.ready_queue = ready_queue
        self.write_idx = 0

    def write_frame(self, frame):
        self.free_queue.get()
        if self.write_idx == 0:
            shm_array = np.ndarray((self.frame_h, self.frame_w, 3), dtype=np.uint8, buffer=self.shm_0.buf)
        else:
            shm_array = np.ndarray((self.frame_h, self.frame_w, 3), dtype=np.uint8, buffer=self.shm_1.buf)
        np.copyto(shm_array, frame)
        self.ready_queue.put(self.write_idx)
        self.write_idx = 1 - self.write_idx

    def close(self):
        try: self.shm_0.close(); self.shm_1.close()
        except: pass

class SharedFrameReader:
    """Читатель для одного кадра"""
    def __init__(self, frame_h, frame_w, shm_name_0, shm_name_1, free_queue, ready_queue):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.shm_0 = shared_memory.SharedMemory(name=shm_name_0)
        self.shm_1 = shared_memory.SharedMemory(name=shm_name_1)
        self.free_queue = free_queue
        self.ready_queue = ready_queue

    def read_frame(self):
        read_idx = self.ready_queue.get()
        if read_idx == 0:
            shm_array = np.ndarray((self.frame_h, self.frame_w, 3), dtype=np.uint8, buffer=self.shm_0.buf)
        else:
            shm_array = np.ndarray((self.frame_h, self.frame_w, 3), dtype=np.uint8, buffer=self.shm_1.buf)
        frame = shm_array.copy()
        self.free_queue.put(1)
        return frame

    def close(self):
        try: self.shm_0.close(); self.shm_1.close()
        except: pass

class SharedCropsBuffer:
    """Буфер для передачи множества кропов (создается в главном процессе)"""
    def __init__(self, num_crops, crop_h, crop_w, name_prefix="shm_crops"):
        self.num_crops = num_crops
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.crop_size = crop_h * crop_w * 3
        self.total_size = num_crops * self.crop_size
        self.shm = shared_memory.SharedMemory(create=True, size=self.total_size)
        self.name = self.shm.name
        print(f"✅ SharedCropsBuffer создан: {num_crops} кропов × {crop_w}x{crop_h} ({self.total_size/1024/1024:.1f} МБ)")

    def close(self):
        try:
            self.shm.close(); self.shm.unlink()
        except Exception as e:
            print(f"⚠️ Ошибка освобождения crops shm: {e}")

class SharedCropsWriter:
    """Писатель для кропов"""
    def __init__(self, num_crops, crop_h, crop_w, shm_name):
        self.num_crops = num_crops
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.crop_size = crop_h * crop_w * 3
        self.shm = shared_memory.SharedMemory(name=shm_name)

    def write_crops(self, crops_list):
        for i, crop in enumerate(crops_list):
            if i >= self.num_crops:
                break
            offset = i * self.crop_size
            shm_array = np.ndarray((self.crop_h, self.crop_w, 3), dtype=np.uint8,
                                   buffer=self.shm.buf[offset:offset+self.crop_size])
            np.copyto(shm_array, crop)

    def close(self):
        try: self.shm.close()
        except: pass

class SharedCropsReader:
    """Читатель для кропов"""
    def __init__(self, num_crops, crop_h, crop_w, shm_name):
        self.num_crops = num_crops
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.crop_size = crop_h * crop_w * 3
        self.shm = shared_memory.SharedMemory(name=shm_name)

    def read_crops(self):
        crops = []
        for i in range(self.num_crops):
            offset = i * self.crop_size
            shm_array = np.ndarray((self.crop_h, self.crop_w, 3), dtype=np.uint8,
                                   buffer=self.shm.buf[offset:offset+self.crop_size])
            crops.append(shm_array.copy())
        return crops

    def close(self):
        try: self.shm.close()
        except: pass

# === НОВЫЙ КЛАСС ДЛЯ NEURO COLLECTOR ===
class SharedFrameWithDetsBuffer:
    """Буфер для передачи кадра + детекций (создается в главном процессе)"""
    def __init__(self, frame_h, frame_w, name_prefix="shm_frame_dets"):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.frame_size = frame_h * frame_w * 3
        self.shm = shared_memory.SharedMemory(create=True, size=self.frame_size)
        self.name = self.shm.name
        print(f"✅ SharedFrameWithDetsBuffer создан: {frame_w}x{frame_h} ({self.frame_size/1024/1024:.1f} МБ)")

    def close(self):
        try:
            self.shm.close(); self.shm.unlink()
        except Exception as e:
            print(f"⚠️ Ошибка освобождения shm: {e}")

class SharedFrameWithDetsWriter:
    """Писатель для кадра + детекций"""
    def __init__(self, frame_h, frame_w, shm_name):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.shm = shared_memory.SharedMemory(name=shm_name)

    def write_frame(self, frame):
        shm_array = np.ndarray((self.frame_h, self.frame_w, 3), dtype=np.uint8, buffer=self.shm.buf)
        np.copyto(shm_array, frame)

    def close(self):
        try: self.shm.close()
        except: pass

class SharedFrameWithDetsReader:
    """Читатель для кадра + детекций"""
    def __init__(self, frame_h, frame_w, shm_name):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.shm = shared_memory.SharedMemory(name=shm_name)

    def read_frame(self):
        shm_array = np.ndarray((self.frame_h, self.frame_w, 3), dtype=np.uint8, buffer=self.shm.buf)
        return shm_array.copy()

    def close(self):
        try: self.shm.close()
        except: pass


class SharedSimpleFrameBuffer:
    """Простой буфер для одного кадра (создается в главном процессе)"""
    def __init__(self, frame_h, frame_w, name_prefix="shm_simple"):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.frame_size = frame_h * frame_w * 3
        self.shm = shared_memory.SharedMemory(create=True, size=self.frame_size)
        self.name = self.shm.name
        print(f"✅ SharedSimpleFrameBuffer создан: {frame_w}x{frame_h} ({self.frame_size/1024/1024:.1f} МБ)")

    def close(self):
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception as e:
            print(f"️ Ошибка освобождения shm: {e}")

class SharedSimpleFrameWriter:
    """Писатель для простого буфера"""
    def __init__(self, frame_h, frame_w, shm_name):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.shm = shared_memory.SharedMemory(name=shm_name)

    def write_frame(self, frame):
        shm_array = np.ndarray((self.frame_h, self.frame_w, 3), dtype=np.uint8, buffer=self.shm.buf)
        np.copyto(shm_array, frame)

    def close(self):
        try:
            self.shm.close()
        except:
            pass

class SharedSimpleFrameReader:
    """Читатель для простого буфера"""
    def __init__(self, frame_h, frame_w, shm_name):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.shm = shared_memory.SharedMemory(name=shm_name)

    def read_frame(self):
        shm_array = np.ndarray((self.frame_h, self.frame_w, 3), dtype=np.uint8, buffer=self.shm.buf)
        return shm_array.copy()

    def close(self):
        try:
            self.shm.close()
        except:
            pass