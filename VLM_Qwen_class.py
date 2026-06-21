import os
import cv2
import torch
import time
import numpy as np
import multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForImageTextToText
from shm_utils import SharedFrameReader

os.environ["HF_HUB_OFFLINE"] = "1"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

class VLM_Qwen:
    def __init__(self, q_to_qwen: mp.Queue,
             shm_mot_to_vlm_name_0=None, shm_mot_to_vlm_name_1=None,
             shm_mot_to_vlm_free_queue=None, shm_mot_to_vlm_ready_queue=None,
             device="cuda:1",
             model_id="Qwen/Qwen2.5-VL-3B-Instruct",
             crop_size=288):
        self.q_to_qwen = q_to_qwen
        self.model_id = model_id
        self.prompt = """Проанализируй изображение и определи военную технику. Ответь СТРОГО по структуре:
ТИП: [танк / БМП / БТР / САУ / дрон / самолёт / вертолёт / корабль / другое / не военная техника]
МОДЕЛЬ: [конкретная модель, например: M1 Abrams, Т-72, Т-72Б3, Т-90М, Leopard 2A6, Challenger 2, Байрактар TB2, Shahed-136, или если на картинке птица то тоже указать, если дрон не самолетного типа указать мультиротор или "не определена"]
ПРИЗНАКИ: [кратко перечисли 2-3 визуальных признака]
УВЕРЕННОСТЬ: [высокая / средняя / низкая]
Отвечай кратко."""
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.crop_size = crop_size

        self.shm_mot_to_vlm_name_0 = shm_mot_to_vlm_name_0
        self.shm_mot_to_vlm_name_1 = shm_mot_to_vlm_name_1
        self.shm_mot_to_vlm_free_queue = shm_mot_to_vlm_free_queue
        self.shm_mot_to_vlm_ready_queue = shm_mot_to_vlm_ready_queue

    def set_prompt(self, prompt: str): self.prompt = prompt

    def load_models(self):
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, local_files_only=True)
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=torch.float16, device_map={"": self.device}, trust_remote_code=True, local_files_only=True).eval()
            print(f"✅ VLM загружен на {self.device}")
            return True, True
        except Exception as e:
            print(f"ERRR_VLM:{e.args}")
            return False, False

    def crop_object(self, frame, bbox, crop_size=288):
        frame_h, frame_w = frame.shape[:2]
        half = crop_size // 2
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        crop_x1 = cx - half; crop_y1 = cy - half; crop_x2 = cx + half; crop_y2 = cy + half
        crop_canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        src_x1 = max(0, crop_x1); src_y1 = max(0, crop_y1); src_x2 = min(frame_w, crop_x2); src_y2 = min(frame_h, crop_y2)
        dst_x1 = src_x1 - crop_x1; dst_y1 = src_y1 - crop_y1; dst_x2 = dst_x1 + (src_x2 - src_x1); dst_y2 = dst_y1 + (src_y2 - src_y1)
        if src_x2 > src_x1 and src_y2 > src_y1:
            crop_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
        return crop_canvas

    def analyze(self, image_pil, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_pil}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image_pil], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def parse_vlm_response(self, response):
        result = {"type": "не определено", "model": "не определена", "confidence": "неизвестно", "raw": response}
        try:
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith("ТИП:"): result["type"] = line.split(":", 1)[1].strip()
                elif line.startswith("МОДЕЛЬ:"): result["model"] = line.split(":", 1)[1].strip()
                elif line.startswith("УВЕРЕННОСТЬ:"): result["confidence"] = line.split(":", 1)[1].strip()
        except Exception: pass
        return result

    def wrap_text_pil(self, text, font, max_width):
        words = text.split(' '); lines = []; current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line); text_width = bbox[2] - bbox[0]
            if text_width <= max_width: current_line.append(word)
            else:
                if current_line: lines.append(' '.join(current_line))
                current_line = [word]
        if current_line: lines.append(' '.join(current_line))
        return lines

    def create_display_frame(self, frame_bgr, text, index, bboxes=None):
        h, w = frame_bgr.shape[:2]
        try: font_main = ImageFont.truetype(FONT_PATH, 18); font_small = ImageFont.truetype(FONT_PATH, 14)
        except IOError: font_main = ImageFont.load_default(); font_small = ImageFont.load_default()
        all_lines = []
        for paragraph in text.split('\n'):
            paragraph = paragraph.strip()
            if not paragraph: all_lines.append(''); continue
            wrapped = self.wrap_text_pil(paragraph, font_main, w - 40); all_lines.extend(wrapped)
        sample_bbox = font_main.getbbox('Agy'); line_height = (sample_bbox[3] - sample_bbox[1]) + 10
        header_height = 25; top_padding = 10; bottom_padding = 15
        text_area_height = header_height + top_padding + len(all_lines) * line_height + bottom_padding
        display_frame = np.ones((h + text_area_height, w, 3), dtype=np.uint8) * 255
        display_frame[:h, :w] = frame_bgr
        pil_img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        header_text = f"[Frame {index}]"
        draw.text((15, h + 5), header_text, font=font_small, fill=(0, 128, 0))
        y_text = h + header_height + top_padding
        for line in all_lines:
            draw.text((15, y_text), line, font=font_main, fill=(0, 0, 0)); y_text += line_height
        result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result_bgr

    def draw_bboxes(self, frame, bboxes):
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"#{i+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def run_process(self, daemon=True):
        prc = mp.Process(target=self.main_func, args=(), daemon=daemon)
        prc.start()
        return prc

    def main_func(self):
        processor, model = self.load_models()
        if not (processor and model): print("❌ VLM не загрузился"); return
        print("🚀 VLM процесс запущен, жду задачи через Shared Memory...")
        shm_reader = None
        frame_idx = 0
        while processor and model:
            if self.q_to_qwen.empty(): time.sleep(0.01); continue
            task = self.q_to_qwen.get()
            if task is None: break
            frame_idx = task['frame_idx']; bboxes = task['bboxes']
            frame_h = task.get('frame_h', 1080); frame_w = task.get('frame_w', 1920)
            try:
                if shm_reader is None:
                    shm_reader = SharedFrameReader(frame_h, frame_w, self.shm_mot_to_vlm_name_0, self.shm_mot_to_vlm_name_1, self.shm_mot_to_vlm_free_queue, self.shm_mot_to_vlm_ready_queue)
                    print(f"✅ VLM: подключён к Shared Memory от MOT ({frame_w}x{frame_h})")
                frame = shm_reader.read_frame()
                print(f"\n Анализ кадра {frame_idx}: {len(bboxes)} объектов")
                results = []
                for i, bbox in enumerate(bboxes):
                    print(f"   📦 Объект #{i+1}... {self.crop_size}", end=" ", flush=True)
                    crop = self.crop_object(frame, bbox, crop_size=self.crop_size)
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    answer = self.analyze(crop_pil, self.prompt)
                    parsed = self.parse_vlm_response(answer); parsed['bbox'] = bbox; results.append(parsed)
                    print(f"{parsed['type']} / {parsed['model']}")
                type_counts = {}
                for r in results: obj_type = r.get("type", "не определено"); type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                lines = [f"═══ ИТОГО НА КАДРЕ: {len(results)} объектов ═══", ""]; lines.append("СВОДКА:")
                for obj_type, count in type_counts.items(): lines.append(f"  • {obj_type}: {count} шт.")
                lines.append(""); lines.append("ДЕТАЛИ:")
                for i, r in enumerate(results, 1):
                    lines.append(f"─── Объект #{i} ───"); lines.append(f"  ТИП: {r.get('type', 'не определено')}")
                    lines.append(f"  МОДЕЛЬ: {r.get('model', 'не определена')}"); lines.append(f"  УВЕРЕННОСТЬ: {r.get('confidence', 'неизвестно')}"); lines.append("")
                final_report = "\n".join(lines)
                frame_with_bboxes = self.draw_bboxes(frame.copy(), bboxes)
                display_frame = self.create_display_frame(frame_with_bboxes, final_report, frame_idx, bboxes)
                info_text = f"Frame: {frame_idx} | Objects: {len(bboxes)}"
                cv2.putText(display_frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Qwen2.5-VL Viewer", display_frame)
                cv2.waitKey(1)
            except Exception as e:
                print(f"❌ Ошибка в VLM: {e}")
                import traceback; traceback.print_exc()
        cv2.destroyAllWindows()