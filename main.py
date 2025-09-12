"""
Aplitaivo desenvolviddo por Oscar Namicano no ambito 
do programa de graduacao em Agropecuaria com habilitacoes em extensao rural
"""

import cv2
from kivy.app import App
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty, ListProperty
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.textfield import MDTextField
import matplotlib.pyplot as plt

NE = 2   # Nível de Equilíbrio
NC = 10  # Nível de Controle
NDE = 15 # Nível de Dano Económico

from kivymd.uix.snackbar import Snackbar
import os, sys

def resource_path(relative_path):
    """Compatível com .exe gerado pelo PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

cfg = resource_path("yolov4-custom.cfg")
weights = resource_path("yolov4-custom_best.weights")
names = resource_path("classe.names")


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            preview=True
        )
        self.folder_mode = False
        self.dialog = None
        self.input_field = None

    def show_file_manager(self, file_type):
        self.folder_mode = False
        self.file_type = file_type
        self.file_manager.show('/')

    def show_folder_manager(self):
        self.folder_mode = True
        self.file_manager.show('/')

    def exit_manager(self, *args):
        self.file_manager.close()

    def select_path(self, path):
        self.exit_manager()
        if self.folder_mode:
            self.manager.current = 'image_detection_screen'
            Clock.schedule_once(lambda dt: self.manager.current_screen.detect_folder(path))
        else:
            if self.file_type == 'video':
                self.manager.current = 'video_detection_screen'
                self.manager.current_screen.load_video(path)
            elif self.file_type == 'image':
                self.manager.current = 'image_detection_screen'
                Clock.schedule_once(lambda dt: self.manager.current_screen.detect_image(path))

    def real_time_detection(self):
        # popup para pedir índice ou IP
        if not self.dialog:
            self.input_field = MDTextField(
                hint_text="Digite índice da cámera (0,1,...) ou IP/URL",
                text="0"
            )
            self.dialog = MDDialog(
                title="Selecionar Fonte de Cámera",
                type="custom",
                content_cls=self.input_field,
                buttons=[
                    MDRaisedButton(
                        text="CANCELAR",
                        on_release=lambda x: self.dialog.dismiss()
                    ),
                    MDRaisedButton(
                        text="OK",
                        on_release=lambda x: self.confirm_camera_source()
                    ),
                ],
            )
        self.dialog.open()

    def confirm_camera_source(self):
        source = self.input_field.text.strip()
        self.dialog.dismiss()
        try:
            source = int(source)
        except ValueError:
            pass
        real_time_screen = self.manager.get_screen('real_time_screen')
        real_time_screen.start_real_time(source)
        self.manager.current = 'real_time_screen'


class VideoDetectionScreen(Screen):
    video_image = ObjectProperty(None)
    detections = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        with open(names, 'r') as f:
            self.classes = f.read().splitlines()

    def load_video(self, path):
        self.capture = cv2.VideoCapture(path)
        if self.capture.isOpened():
            self.detections.clear()
            Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        else:
            self.show_error_popup("Erro ao carregar o vídeo.")

    def update_frame(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                classIds, scores, boxes = self.model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)
                self.detections.append(len(classIds))
                for (classId, score, box) in zip(classIds, scores, boxes):
                    cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,255), 2)
                    cv2.putText(frame, f'afideo: {score:.2f}', (box[0], box[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.video_image.texture = texture
            else:
                self.capture.release()
                self.capture = None
                self.show_info_popup("Fim do vídeo.")
                self.manager.current = 'main_screen'

    def generate_report(self):
        if not self.detections:
            self.show_info_popup("Nenhuma detecção feita ainda.")
            return
        tempos = list(range(len(self.detections)))
        plt.clf()
        plt.plot(tempos, self.detections, label='Afídeos Detectados')
        plt.axhline(y=NE, color='green', linestyle='--', label='Nível de Equilíbrio')
        plt.axhline(y=NC, color='orange', linestyle='--', label='Nível de Controle')
        plt.axhline(y=NDE, color='red', linestyle='--', label='Nível de Dano Económico')
        plt.xlabel('Quadros')
        plt.ylabel('Afídeos Detectados')
        plt.title('Relatório Parcial - Vídeo')
        plt.legend()
        plt.show()

    def show_error_popup(self, text):
        MDDialog(text=text).open()

    def show_info_popup(self, text):
        MDDialog(text=text).open()

    def go_back_to_main(self):
        if self.capture:
            self.capture.release()
        self.manager.current = 'main_screen'


class ImageDetectionScreen(Screen):
    image_display = ObjectProperty(None)
    detections = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        with open(cfg, 'r') as f:
            self.classes = f.read().splitlines()
        self.processing_folder = False

    def detect_image(self, path):
        self.processing_folder = False
        self.detections.clear()
        img = cv2.imread(path)
        if img is not None:
            classIds, scores, boxes = self.model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
            self.detections.append(len(classIds))
            for (classId, score, box) in zip(classIds, scores, boxes):
                cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,255), 2)
                cv2.putText(img, f'afideo: {score:.2f}', (box[0], box[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            buf = cv2.flip(img, 0).tobytes()
            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_display.texture = texture
        else:
            self.show_error_popup("Erro ao carregar a imagem.")

    def detect_folder(self, folder_path):
        self.processing_folder = True
        self.detections.clear()
        self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0

        if not self.images:
            self.show_error_popup("Nenhuma imagem encontrada na pasta.")
            return

        self.process_next_image()

    def process_next_image(self, dt=0):
        if self.current_index >= len(self.images):
            self.show_info_popup("Processamento da pasta concluído.")
            return

        path = self.images[self.current_index]
        img = cv2.imread(path)
        if img is None:
            self.current_index += 1
            Clock.schedule_once(self.process_next_image, 0.5)
            return

        classIds, scores, boxes = self.model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
        self.detections.append(len(classIds))
        for (classId, score, box) in zip(classIds, scores, boxes):
            cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,255), 2)
            cv2.putText(img, f'afideo: {score:.2f}', (box[0], box[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        buf = cv2.flip(img, 0).tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_display.texture = texture

        self.current_index += 1
        Clock.schedule_once(self.process_next_image, 1.5)

    def generate_report(self):
        if not self.detections:
            self.show_info_popup("Nenhuma detecção feita ainda.")
            return

        plt.clf()

        if self.processing_folder:
            amostras = list(range(1, len(self.detections) + 1))
            plt.plot(amostras, self.detections, label='Afídeos Detectados')
            plt.axhline(y=NE, color='green', linestyle='--', label='Nível de Equilíbrio')
            plt.axhline(y=NC, color='orange', linestyle='--', label='Nível de Controle')
            plt.axhline(y=NDE, color='red', linestyle='--', label='Nível de Dano Económico')
            plt.xlabel('Número de amostra (folha)')
            plt.ylabel('Número de Afídeos (colónias)')
            plt.title('Níveis de Dano Económico por Afídeos')
            plt.legend()
        else:
            plt.bar(['Imagem'], self.detections, color='blue', label='Afídeos Detectados')
            plt.axhline(y=NE, color='green', linestyle='--', label='NE')
            plt.axhline(y=NC, color='orange', linestyle='--', label='NC')
            plt.axhline(y=NDE, color='red', linestyle='--', label='NDE')
            plt.ylabel('Afídeos Detectados')
            plt.title('Relatório - Imagem')
            plt.legend()

        plt.show()
    
    def go_back(self):
        self.manager.current = 'main_screen'
    
    def show_error_popup(self, text):
        MDDialog(text=text).open()

    def show_info_popup(self, text):
        MDDialog(text=text).open()


class RealTimeScreen(Screen):
    real_time_image = ObjectProperty(None)
    detections = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        with open(names, 'r') as f:
            self.classes = f.read().splitlines()

    def start_real_time(self, source=0):
        self.detections.clear()
        self.capture = cv2.VideoCapture(source)
        if self.capture.isOpened():
            Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        else:
            self.show_error_popup("Não foi possível acessar a cámera ou stream.")

    def update_frame(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                classIds, scores, boxes = self.model.detect(frame, confThreshold=0.5, nmsThreshold=0.4)
                self.detections.append(len(classIds))
                for (classId, score, box) in zip(classIds, scores, boxes):
                    cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,255), 2)
                    cv2.putText(frame, f'afideo: {score:.2f}', (box[0], box[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.real_time_image.texture = texture
            else:
                self.stop_real_time()
                self.show_info_popup("Cámera desconectada.")
                self.manager.current = 'main_screen'

    def stop_real_time(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        Clock.unschedule(self.update_frame)
        self.real_time_image.texture = None

    def generate_report(self):
        if not self.detections:
            self.show_info_popup("Nenhuma detecção feita ainda.")
            return
        tempos = list(range(len(self.detections)))
        plt.clf()
        plt.plot(tempos, self.detections, label='Afídeos Detectados')
        plt.axhline(y=NE, color='green', linestyle='--', label='NE')
        plt.axhline(y=NC, color='orange', linestyle='--', label='NC')
        plt.axhline(y=NDE, color='red', linestyle='--', label='NDE')
        plt.xlabel('Tempo')
        plt.ylabel('Afídeos Detectados')
        plt.title('Relatório Parcial - Tempo Real')
        plt.legend()
        plt.show()

    def go_back_to_main(self):
        self.manager.current = 'main_screen'

    def show_error_popup(self, text):
        MDDialog(text=text).open()

    def show_info_popup(self, text):
        MDDialog(text=text).open()


class RootApp(MDApp):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main_screen'))
        sm.add_widget(VideoDetectionScreen(name='video_detection_screen'))
        sm.add_widget(ImageDetectionScreen(name='image_detection_screen'))
        sm.add_widget(RealTimeScreen(name='real_time_screen'))
        return sm


if __name__ == '__main__':
    RootApp().run()