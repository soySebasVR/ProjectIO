import sympy as sp
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
            ("f", "Función f(x)", "x**3 - 2*x - 5"),
            ("x0", "Valor Inicial (x0)", "2.0"),
        ]

        left_layout.addWidget(title_label)
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
            ["Iteración", "x", "x_next", "Error", "Porcentaje"]
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
            # Parse numeric parameters
            x0 = float(self.params["x0"].text())
            
            # Parse function
            f_str = self.params["f"].text()
            x_sym = sp.Symbol("x")
            f_expr = sp.sympify(f_str)
            df_expr = sp.diff(f_expr, x_sym)
            
            # Create callable functions
            f_func = sp.lambdify(x_sym, f_expr, "numpy")
            df_func = sp.lambdify(x_sym, df_expr, "numpy")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error de Entrada",
                f"Error al procesar la función o los parámetros:\n{str(e)}",
            )
            return

        history: list[tuple[int, float, float, float, float]] = []
        x_curr = x0
        k = 0
        max_safety_iter = 100  # Safety limit to prevent infinite freeze

        while True:
            try:
                f_val = float(f_func(x_curr))
                df_val = float(df_func(x_curr))
            except Exception as e:
                QMessageBox.warning(self, "Error de Evaluación", f"Error evaluando la función: {e}")
                break

            if abs(df_val) < 1e-12:
                QMessageBox.warning(
                    self,
                    "Error de Cálculo",
                    f"La derivada es cero en la iteración {k}. No se puede continuar.",
                )
                break

            # Newton-Raphson step: x_new = x_curr - f(x)/f'(x)
            x_next = x_curr - f_val / df_val
            
            error = abs((x_next - x_curr) / x_next) if x_next != 0 else 0.0
            pct = error * 100
            history.append((k, x_curr, x_next, error, pct))

            # Stop if error is effectively 0 (or extremely small float epsilon)
            if error == 0.0:
                break
            
            # Safety break
            if k >= max_safety_iter:
                QMessageBox.warning(self, "Límite de Seguridad", "Se alcanzaron 100 iteraciones sin llegar a error 0 absoluto.")
                break

            x_curr = x_next
            k += 1

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

        ax.plot(iterations, c_values, marker="o", color="blue", label="x (iteraciones)")
        ax.axhline(
            C_opt, color="red", linestyle="--", label=f"x óptimo = {C_opt:.6f}"
        )

        ax.set_xlabel("Iteración")
        ax.set_ylabel("x")
        ax.set_title("Convergencia de Newton-Raphson")
        ax.legend()
        ax.grid(True)

        self.canvas.draw()
