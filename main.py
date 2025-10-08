import sys
import matplotlib
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget,
)
from PySide6.QtCore import Qt
from modes.newton_raphson import NewtonRaphsonScreen

matplotlib.use("QtAgg")


class ParameterScreen(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        title_label = QLabel(f"Configuración de: {title}")
        self.back_button = QPushButton("Volver al Menú")
        left_layout.addWidget(title_label)
        left_layout.addStretch()
        left_layout.addWidget(self.back_button)
        image_label = QLabel(f"Pantalla para '{title}'")
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(image_label, 2)


class MainMenuScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button1 = QPushButton("Newton-Raphson")
        self.button2 = QPushButton("Opción 2")
        self.button3 = QPushButton("Opción 3")
        self.button4 = QPushButton("Opción 4")
        button_style = "QPushButton { font-size: 18px; padding: 15px; }"
        self.button1.setStyleSheet(button_style)
        self.button2.setStyleSheet(button_style)
        self.button3.setStyleSheet(button_style)
        self.button4.setStyleSheet(button_style)
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.button4)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplicación")
        self.setGeometry(100, 100, 1200, 700)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.main_menu = MainMenuScreen()
        self.screen1 = NewtonRaphsonScreen()
        self.screen2 = ParameterScreen("Opción 2")
        self.screen3 = ParameterScreen("Opción 3")
        self.screen4 = ParameterScreen("Opción 4")

        self.stacked_widget.addWidget(self.main_menu)  # Índice 0
        self.stacked_widget.addWidget(self.screen1)  # Índice 1
        self.stacked_widget.addWidget(self.screen2)  # Índice 2
        self.stacked_widget.addWidget(self.screen3)  # Índice 3
        self.stacked_widget.addWidget(self.screen4)  # Índice 4

        self.main_menu.button1.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.main_menu.button2.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(2)
        )
        self.main_menu.button3.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(3)
        )
        self.main_menu.button4.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(4)
        )

        self.screen1.back_button.clicked.connect(self.go_to_main_menu)
        self.screen2.back_button.clicked.connect(self.go_to_main_menu)
        self.screen3.back_button.clicked.connect(self.go_to_main_menu)
        self.screen4.back_button.clicked.connect(self.go_to_main_menu)

    def go_to_main_menu(self):
        self.stacked_widget.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
