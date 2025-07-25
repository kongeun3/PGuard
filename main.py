# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////

import sys
import os
import platform

import torch

# IMPORT / GUI AND MODULES AND WIDGETS
# ///////////////////////////////////////////////////////////////
from modules import *
from widgets import *
os.environ["QT_FONT_DPI"] = "96" # FIX Problem for High DPI and Scale above 100%

# IMPORT / USER CLASSES
# ///////////////////////////////////////////////////////////////
from modules.ai_functions import AIClass
from modules.image_functions import ImageClass

# SET AS GLOBAL WIDGETS
# ///////////////////////////////////////////////////////////////
widgets = None

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # DEFINITE PROJECT GLOBAL VARIABLES
        # ///////////////////////////////////////////////////////////////
        # GUI Variables
        self.home = os.getcwd()
        self.ControlKey = False

        # AI Variables
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.score_thr = 0.7
        self.text_prompt = "manhole"

        self.image_src_folder = ""
        self.image_save_folder = os.path.join(self.home, "init", "results")
        # image: 이미지 이름. dino_bbox: 탐지된 바운딩 박스의 리스트, dino_score: 탐지된 바운딩 박스의 스코어 리스트, resnet: 0-개구부 아님, 1-정상 개구부, 2-불량 개구부
        self.ai_result_dict = {"image": [], "image_path": [], "dino_bbox": [], "dino_score": [], "resnet": []}

        # Image Variables
        self.fileModel = QFileSystemModel()

        self.plot_image_path = ""

        self.scale = 1.0
        self.pixmap = None
        self.pixmap_item = None

        # CALL USER CUSTOM FUNCTIONS
        # ///////////////////////////////////////////////////////////////
        self.ai = AIClass(self.ui, self)
        self.imgcls = ImageClass(self.ui, self)

        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title = "PyDracula - Modern GUI"
        description = "P-Guard: Prompt based automated opening Guard System"
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # TOGGLE MENU
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # SET UI DEFINITIONS
        # ///////////////////////////////////////////////////////////////
        UIFunctions.uiDefinitions(self)

        # BUTTONS CLICK
        # ///////////////////////////////////////////////////////////////

        # LEFT MENUS
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_calculate.clicked.connect(self.buttonClick)
        widgets.btn_show.clicked.connect(self.buttonClick)

        # EXTRA LEFT BOX
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)
        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # EXTRA RIGHT BOX
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)
        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # SHOW APP
        # ///////////////////////////////////////////////////////////////
        self.show()

        # SET CUSTOM THEME
        # ///////////////////////////////////////////////////////////////
        useCustomTheme = False
        themeFile = "themes\py_dracula_light.qss"

        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)

        # SET HOME PAGE AND SELECT MENU
        # ///////////////////////////////////////////////////////////////
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))

        # EXTRA WHEEL EVENT
        # ///////////////////////////////////////////////////////////////
        self.ui.scrollAreaImage.wheelEvent = self.wheelEventScroll


    # BUTTONS CLICK
    # Post here your functions for clicked buttons
    # ///////////////////////////////////////////////////////////////
    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # SHOW HOME PAGE
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW CALCULATE PAGE
        if btnName == "btn_calculate":
            widgets.stackedWidget.setCurrentWidget(widgets.calculate)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW SHOWING PAGE
        if btnName == "btn_show":
            widgets.stackedWidget.setCurrentWidget(widgets.show) # SET PAGE
            UIFunctions.resetStyle(self, btnName) # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet())) # SELECT MENU


    # RESIZE EVENTS
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')

    def keyPressEvent(self, event):
        if event.key() == 16777249: # Ctrl key
            self.ControlKey = True
    
    def keyReleaseEvent(self, event):
        if event.key() == 16777249: # Ctrl key
            self.ControlKey = False
    
    def wheelEventScroll(self, event):
    
        self.mouseWheelAngleDelta = event.angleDelta().y() # -> 1 (up), -1 (down)

        if self.ControlKey:

            if self.mouseWheelAngleDelta > 0: 
                self.scale *= 1.1
                width_future = int(widgets.mainImageViewer.geometry().width() * 1.1)
                height_future = int(widgets.mainImageViewer.geometry().height() * 1.1)
            else : 
                self.scale /= 1.1
                width_future = int(widgets.mainImageViewer.geometry().width() / 1.1)
                height_future = int(widgets.mainImageViewer.geometry().height() / 1.1)
            
            self.oldPos = widgets.mainImageViewer.mapFromGlobal(QCursor.pos())
            cursor_x = self.oldPos.x()  
            cursor_y = self.oldPos.y()

            cursor_x = np.clip(cursor_x, 0, widgets.mainImageViewer.geometry().width())
            cursor_y = np.clip(cursor_y, 0, widgets.mainImageViewer.geometry().height())

            cursor_x = cursor_x / widgets.mainImageViewer.geometry().width()
            cursor_y = cursor_y / widgets.mainImageViewer.geometry().height()

            _width_diff = width_future - widgets.scrollAreaImage.geometry().width()
            _height_diff = height_future - widgets.scrollAreaImage.geometry().height() 

            set_hor_max = _width_diff + 18 if _width_diff > 0 else 0 # check padd value for scrollArea
            set_ver_max = _height_diff + 18 if _height_diff > 0 else 0 # 

            widgets.scrollAreaImage.horizontalScrollBar().setRange(0, set_hor_max) 
            widgets.scrollAreaImage.verticalScrollBar().setRange(0, set_ver_max) 
            
            if widgets.scrollAreaImage.verticalScrollBar().maximum() > 0: 
                setvalueY = int(cursor_y*set_ver_max)
                widgets.scrollAreaImage.verticalScrollBar().setValue(setvalueY)

            if widgets.scrollAreaImage.horizontalScrollBar().maximum() > 0: 
                setvalueX = int(cursor_x*set_hor_max)
                widgets.scrollAreaImage.horizontalScrollBar().setValue(setvalueX)

            widgets.mainImageViewer.setFixedSize(self.scale * self.pixmap.size())
            widgets.mainImageViewer.fitInView(self.pixmap_item)
            widgets.mainImageViewer.setVisible(False)
            
            widgets.mainImageViewer.setVisible(True)
        
        else : 
            scroll_value = widgets.scrollAreaImage.verticalScrollBar().value()
            widgets.scrollAreaImage.verticalScrollBar().setValue(scroll_value - self.mouseWheelAngleDelta)

if __name__ == "__main__":
    icon_path = os.path.join(os.getcwd(), "init", "icon", "pguard.png")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path))
    window = MainWindow()
    sys.exit(app.exec_())
