import sys
import matplotlib
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QStackedWidget,
    QHBoxLayout,
    QLabel,
)
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import Qt
from screens import NewtonRaphsonScreen, GoldenSectionScreen, SymbolicCalcScreen

matplotlib.use("QtAgg")


class WelcomeScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Layout principal centrado
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(25)

        # --- Sección Superior ---
        top_layout = QHBoxLayout()
        top_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.setSpacing(20)

        logo_label = QLabel()
        logo_label.setFixedSize(118, 177)
        pixmap = QPixmap("static/img/logo.svg")
        logo_label.setPixmap(
            pixmap.scaled(
                118,
                177,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Títulos a la derecha
        titles_layout = QVBoxLayout()
        titles_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)

        subtitle_font = QFont()
        subtitle_font.setPointSize(20)

        title_label = QLabel("Facultad de Ciencias Matemáticas")
        title_label.setFont(title_font)

        subtitle_label = QLabel("Escuela de Investigación Operativa")
        subtitle_label.setFont(subtitle_font)

        titles_layout.addWidget(title_label)
        titles_layout.addWidget(subtitle_label)

        top_layout.addWidget(logo_label)
        top_layout.addLayout(titles_layout)

        # --- Sección del Curso ---
        course_font = QFont()
        course_font.setPointSize(18)
        course_font.setItalic(True)

        course_label = QLabel("Curso: Programación No Lineal y Entera")
        course_label.setFont(course_font)
        course_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Sección de Botones ---
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(15)

        button_style = "QPushButton { font-size: 20px; padding: 15px 30px; }"

        self.nonlinear_button = QPushButton("Programación No Lineal")
        self.nonlinear_button.setStyleSheet(button_style)

        self.integer_button = QPushButton("Programación Entera")
        self.integer_button.setStyleSheet(button_style)
        self.integer_button.setEnabled(False)

        buttons_layout.addWidget(self.nonlinear_button)
        buttons_layout.addWidget(self.integer_button)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(course_label)
        main_layout.addLayout(buttons_layout)


class NonLinearMenuScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Parte Superior ---
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.setSpacing(20)

        title_label = QLabel("Programación No lineal")
        title_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; margin-bottom: 20px;"
        )

        self.button1 = QPushButton("Método Newton-Raphson")
        self.button2 = QPushButton("Método de la región dorada")
        self.button3 = QPushButton("Gradiente y Hessiano")

        top_layout.addWidget(title_label)
        top_layout.addWidget(self.button1)
        top_layout.addWidget(self.button2)
        top_layout.addWidget(self.button3)

        main_layout.addWidget(top_widget)
        main_layout.addStretch()

        bottom_layout = QHBoxLayout()

        logo_label = QLabel()
        logo_label.setFixedSize(95, 142)
        pixmap = QPixmap("static/img/logo.svg")
        logo_label.setPixmap(
            pixmap.scaled(
                95,
                142,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.back_button = QPushButton("Volver")
        self.back_button.setObjectName("greyVolverButton")

        bottom_layout.addWidget(logo_label, 0, Qt.AlignmentFlag.AlignBottom)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.back_button, 0, Qt.AlignmentFlag.AlignBottom)

        main_layout.addLayout(bottom_layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculadora de Métodos Iterativos")
        self.setGeometry(100, 100, 1400, 800)

        app_theme = """
            QMainWindow, QWidget {
                background-color: #F4F8F7;
                color: #212121;
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #00796B;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
            }
            QPushButton:hover {
                background-color: #00897B;
            }
            QPushButton:pressed {
                background-color: #00695C;
            }
            QPushButton:disabled {
                background-color: #B2DFDB;
                color: #757575;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #BDBDBD;
                padding: 8px;
                border-radius: 4px;
                font-size: 14px;
            }
            QTableWidget {
                border: 1px solid #BDBDBD;
                gridline-color: #E0E0E0;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #E0E8E7;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
            QLabel {
                background-color: transparent;
            }
        """
        self.setStyleSheet(app_theme)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.welcome_screen = WelcomeScreen()
        self.nonlinear_menu = NonLinearMenuScreen()
        self.screen1 = GoldenSectionScreen()
        self.screen2 = NewtonRaphsonScreen()
        self.screen3 = SymbolicCalcScreen()

        self.stacked_widget.addWidget(self.welcome_screen)  # Índice 0
        self.stacked_widget.addWidget(self.nonlinear_menu)  # Índice 1
        self.stacked_widget.addWidget(self.screen1)  # Índice 2
        self.stacked_widget.addWidget(self.screen2)  # Índice 3
        self.stacked_widget.addWidget(self.screen3)  # Índice 4

        self.welcome_screen.nonlinear_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.nonlinear_menu.button1.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(3)
        )
        self.nonlinear_menu.button2.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(2)
        )
        self.nonlinear_menu.button3.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(4)
        )

        self.nonlinear_menu.back_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(0)
        )

        self.screen1.back_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.screen2.back_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.screen3.back_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
