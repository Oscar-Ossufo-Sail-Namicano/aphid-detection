import cv2
from kivy.app import App
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivymd.theming import ThemeManager
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivymd.uix.boxlayout import MDBoxLayout

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path
        )

    def show_file_manager(self, file_type):
        self.file_type = file_type
        self.file_manager.show('/')  # Pode definir um caminho inicial diferente

    def exit_manager(self, *args):
        self.file_manager.close()

    def select_path(self, path):
        self.exit_manager()
        if self.file_type == 'video':
            self.manager.current = 'video_detection_screen'
            self.manager.current_screen.load_video(path)
        elif self.file_type == 'image':
            self.manager.current = 'image_detection_screen'
            # Agende a chamada de detect_image para o próximo ciclo do loop de eventos
            Clock.schedule_once(lambda dt: self.manager.current_screen.detect_image(path))

    def real_time_detection(self):
        self.manager.current = 'real_time_screen'

class VideoDetectionScreen(Screen):
    video_image = ObjectProperty(None)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.effect = None
        self.net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()

    def load_video(self, path):
        self.capture = cv2.VideoCapture(path)
        if self.capture.isOpened():
            Clock.schedule_interval(self.update_frame, 1.0 / 30.0) # Ajuste o FPS conforme necessário
        else:
            self.show_error_popup("Erro ao carregar o vídeo.")

    def update_frame(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                classIds, scores, boxes = self.model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)
                for (classId, score, box) in zip(classIds, scores, boxes):
                    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                  color=(0, 255, 0), thickness=2)
                    text = '%s: %.2f' % (self.classes[classId], score)
                    cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color=(0, 255, 0), thickness=1)

                buf = cv2.flip(frame, 0).tostring()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.video_image.texture = texture
            else:
                self.capture.release()
                self.capture = None
                self.show_info_popup("Fim do vídeo.")
                self.manager.current = 'main_screen'

    def stop_video(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        self.manager.current = 'main_screen'

    def show_error_popup(self, text):
        dialog = MDDialog(text=text)
        dialog.open()

    def show_info_popup(self, text):
        dialog = MDDialog(text=text)
        dialog.open()
    def go_back_to_main(self):
        self.manager.current = 'main_screen'

class ImageDetectionScreen(Screen):
    image_display = ObjectProperty(None)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()

    def detect_image(self, path):
        img = cv2.imread(path)
        if img is not None:
            classIds, scores, boxes = self.model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
            for (classId, score, box) in zip(classIds, scores, boxes):
                cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                              color=(0, 255, 0), thickness=2)
                text = '%s: %.2f' % (self.classes[classId], score)
                cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color=(0, 255, 0), thickness=2)

            buf = cv2.flip(img, 0).tostring()
            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_display.texture = texture
        else:
            self.show_error_popup("Erro ao carregar a imagem.")

    def go_back(self):
        self.image_display.texture = None
        self.manager.current = 'main_screen'

    def show_error_popup(self, text):
        dialog = MDDialog(text=text)
        dialog.open()

class RealTimeScreen(Screen):
    real_time_image = ObjectProperty(None)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()

    def on_enter(self):
        self.start_real_time()

    def on_leave(self):
        self.stop_real_time()

    def start_real_time(self):
        self.capture = cv2.VideoCapture(0)  # Use 0 para a câmera padrão
        if self.capture.isOpened():
            Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        else:
            self.show_error_popup("Não foi possível acessar a câmera.")

    def go_back_to_main(self):
        self.manager.current = 'main_screen'

    def update_frame(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                classIds, scores, boxes = self.model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)
                for (classId, score, box) in zip(classIds, scores, boxes):
                    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                                  color=(0, 255, 0), thickness=2)
                    text = '%s: %.2f' % (self.classes[classId], score)
                    cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color=(0, 255, 0), thickness=1)

                buf = cv2.flip(frame, 0).tostring()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.real_time_image.texture = texture
            else:
                self.stop_real_time()
                self.show_info_popup("Câmera desconectada.")
                self.manager.current = 'main_screen'

    def stop_real_time(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        Clock.unschedule(self.update_frame)
        self.real_time_image.texture = None
        self.manager.current = 'main_screen'

    def show_error_popup(self, text):
        dialog = MDDialog(text=text)
        dialog.open()

    def show_info_popup(self, text):
        dialog = MDDialog(text=text)
        dialog.open()

class RootApp(MDApp):
    theme_cls = ThemeManager()

    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main_screen'))
        sm.add_widget(VideoDetectionScreen(name='video_detection_screen'))
        sm.add_widget(ImageDetectionScreen(name='image_detection_screen'))
        sm.add_widget(RealTimeScreen(name='real_time_screen'))
        return sm

if __name__ == '__main__':
    RootApp().run()
