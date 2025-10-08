from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
)
from PySide6.QtGui import QFont


class NewtonRaphsonScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Layout principal
        main_layout = QHBoxLayout(self)

        # --- Lado Izquierdo ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        title_label = QLabel("Método Newton-Raphson")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 10px;"
        )

        # --- Parámetros ---
        self.params: dict[str, QLineEdit] = {}
        param_font = QFont()
        param_font.setPointSize(10)

        # (nombre_interno, "Texto para mostrar", valor_por_defecto)
        param_list: list[tuple[str, str, str]] = [
            ("w", "Frecuencia Angular (w)", "376.9911"),
            ("L", "Inductancia (L)", "0.1"),
            ("Cn", "Capacitancia Inicial (Cn)", "1e-5"),
            ("tol", "Tolerancia (tol)", "1e-6"),
            ("maxiter", "Máx. Iteraciones", "8"),
        ]

        for name, text, default_val in param_list:
            label = QLabel(text)
            label.setFont(param_font)
            line_edit = QLineEdit(default_val)
            line_edit.setFont(param_font)
            self.params[name] = line_edit
            left_layout.addWidget(label)
            left_layout.addWidget(line_edit)

        self.calculate_button = QPushButton("Calcular")
        self.calculate_button.setStyleSheet(
            "font-size: 14px; padding: 8px; margin-top: 15px;"
        )

        self.back_button = QPushButton("Volver")

        left_layout.addWidget(self.calculate_button)
        left_layout.addStretch()
        left_layout.addWidget(self.back_button)

        # --- Lado Derecho ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Tabla de resultados
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(5)
        self.table_widget.setHorizontalHeaderLabels(
            ["Iteración", "C", "G(x)", "Error", "Porcentaje"]
        )
        self.table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table_widget.verticalHeader().setVisible(False)

        # Gráfico de Matplotlib
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)

        right_layout.addWidget(QLabel("Tabla de Iteraciones:"), 0)
        right_layout.addWidget(self.table_widget, 1)  # Proporción 1
        right_layout.addWidget(QLabel("Gráfico de Convergencia:"), 0)
        right_layout.addWidget(self.canvas, 2)  # Proporción 2

        # Main layout
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 3)

        self.calculate_button.clicked.connect(self.run_calculation)

    def run_calculation(self):
        try:
            p: dict[str, float] = {
                name: float(widget.text()) for name, widget in self.params.items()
            }
            p["maxiter"] = int(p["maxiter"])
        except ValueError:
            QMessageBox.critical(
                self,
                "Error de Entrada",
                "Por favor, ingresa solo valores numéricos válidos.",
            )
            return

        def f(C: float, w: float, L: float) -> float:
            return w * L - 1.0 / (w * C)

        def df(C: float, w: float) -> float:
            if w == 0 or C == 0:
                return float("inf")
            return 1.0 / (w * C * C)

        history: list[tuple[int, float, float, float, float]] = []
        C: float = p["Cn"]
        for k in range(p["maxiter"]):
            f_val: float = f(C, p["w"], p["L"])
            df_val: float = df(C, p["w"])

            if abs(df_val) < 1e-12:
                QMessageBox.warning(
                    self,
                    "Error de Cálculo",
                    f"La derivada es cero en la iteración {k}. No se puede continuar.",
                )
                break

            gx: float = C - f_val / df_val
            error: float = abs((gx - C) / gx) if gx != 0 else 0
            pct: float = error * 100
            history.append((k, C, gx, error, pct))

            if abs(f_val) < p["tol"]:
                break
            C = gx

        self.populate_table(history)

        if history:
            self.plot_convergence(history)

    def populate_table(self, history: list[tuple[int, float, float, float, float]]):
        self.table_widget.setRowCount(len(history))
        for row, item in enumerate(history):
            self.table_widget.setItem(row, 0, QTableWidgetItem(f"{item[0]}"))
            self.table_widget.setItem(row, 1, QTableWidgetItem(f"{item[1]:.8f}"))
            self.table_widget.setItem(row, 2, QTableWidgetItem(f"{item[2]:.8f}"))
            self.table_widget.setItem(row, 3, QTableWidgetItem(f"{item[3]:.6f}"))
            self.table_widget.setItem(row, 4, QTableWidgetItem(f"{item[4]:.6f}"))

    def plot_convergence(self, history):
        iterations = [h[0] for h in history]
        c_values = [h[1] for h in history]
        C_opt = history[-1][2]

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(iterations, c_values, marker="o", color="blue", label="C (iteraciones)")
        ax.axhline(
            C_opt, color="red", linestyle="--", label=f"C óptimo = {C_opt:.2e} F"
        )

        ax.set_xlabel("Iteración")
        ax.set_ylabel("C (Capacitancia en F)")
        ax.set_title("Convergencia de C por iteración (Newton-Raphson)")
        ax.legend()
        ax.grid(True)

        self.canvas.draw()
