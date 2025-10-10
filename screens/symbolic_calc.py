import sympy as sp
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTextEdit,
)
from PySide6.QtGui import QFont


class SymbolicCalcScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)

        # --- Lado Izquierdo ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        title_label = QLabel("Cálculo Simbólico")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 10px;"
        )

        # left_layout.addWidget(QLabel("Número de variables (n):"))
        self.n_vars_input = QLineEdit("4")

        # left_layout.addWidget(QLabel("Función f(x1, x2, ...):"))
        self.f_input = QLineEdit("x1**2 + 2*x2*x3 + 3*x4**2")

        # left_layout.addWidget(QLabel("Punto de evaluación (separado por comas):"))
        self.point_input = QLineEdit("1, 2, 3, 4")

        self.calculate_button = QPushButton("Calcular")
        self.calculate_button.setStyleSheet(
            "font-size: 14px; padding: 8px; margin-top: 15px;"
        )
        self.back_button = QPushButton("Volver")

        # Añadir widgets al layout izquierdo
        left_layout.addWidget(title_label)
        left_layout.addWidget(QLabel("Número de variables (n):"))
        left_layout.addWidget(self.n_vars_input)
        left_layout.addWidget(QLabel("Función f(x1, x2, ...):"))
        left_layout.addWidget(self.f_input)
        left_layout.addWidget(QLabel("Punto de evaluación:"))
        left_layout.addWidget(self.point_input)
        left_layout.addWidget(self.calculate_button)
        left_layout.addStretch()
        left_layout.addWidget(self.back_button)

        # --- Lado Derecho ---
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)  # Dividido en texto y gráfico

        text_results_widget = QWidget()
        text_layout = QVBoxLayout(text_results_widget)

        self.f_output = QTextEdit()
        self.grad_output = QTextEdit()
        self.hess_output = QTextEdit()
        self.eval_output = QTextEdit()

        for widget in [
            self.f_output,
            self.grad_output,
            self.hess_output,
            self.eval_output,
        ]:
            widget.setReadOnly(True)
            widget.setFont(QFont("Courier New", 10))

        text_layout.addWidget(QLabel("Función Interpretada:"))
        text_layout.addWidget(self.f_output)
        text_layout.addWidget(QLabel("Gradiente:"))
        text_layout.addWidget(self.grad_output)
        text_layout.addWidget(QLabel("Hessiano:"))
        text_layout.addWidget(self.hess_output)
        text_layout.addWidget(QLabel("Resultados Evaluados:"))
        text_layout.addWidget(self.eval_output)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        right_layout.addWidget(text_results_widget, 1)
        right_layout.addWidget(self.canvas, 2)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 3)

        self.calculate_button.clicked.connect(self.run_calculation)

    def run_calculation(self):
        try:
            n = int(self.n_vars_input.text())
            if n <= 0:
                raise ValueError("El número de variables debe ser positivo.")

            f_str = self.f_input.text()
            point_str = self.point_input.text().split(",")
            point_vals = [float(p.strip()) for p in point_str]

            if len(point_vals) != n:
                raise ValueError(
                    f"El punto debe tener {n} coordenadas, pero tiene {len(point_vals)}."
                )

            vars = sp.symbols(f"x1:{n+1}")
            local_dict = {f"x{i+1}": vars[i] for i in range(n)}
            f = sp.sympify(f_str, locals=local_dict)

            gradiente = [sp.diff(f, v) for v in vars]
            hessiano = sp.hessian(f, vars)

            punto = {v: val for v, val in zip(vars, point_vals)}
            grad_eval = [g.evalf(subs=punto) for g in gradiente]
            hess_eval = hessiano.evalf(subs=punto)

            self.f_output.setText(sp.pretty(f, use_unicode=True))
            self.grad_output.setText(sp.pretty(gradiente, use_unicode=True))
            self.hess_output.setText(sp.pretty(hessiano, use_unicode=True))

            eval_text = f"Gradiente evaluado:\n{grad_eval}\n\nHessiano evaluado:\n{sp.pretty(hess_eval, use_unicode=True)}"
            self.eval_output.setText(eval_text)

            self.plot_function(f, vars, n)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error: {e}")
            self.clear_outputs()

    def plot_function(self, f, vars, n):
        self.figure.clear()

        if n < 2:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Se necesitan al menos 2 variables para graficar.",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.canvas.draw()
            return

        ax = self.figure.add_subplot(111, projection="3d")
        x_vals = np.linspace(-5, 5, 100)
        y_vals = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        f_to_plot = f
        title = f"Superficie de f({vars[0]}, {vars[1]})"

        if n > 2:
            subs_point = {vars[i]: 1 for i in range(2, n)}
            f_to_plot = f.subs(subs_point)
            title = f"Proyección de f (con x3...=1)"

        f_num = sp.lambdify((vars[0], vars[1]), f_to_plot, "numpy")
        Z = f_num(X, Y)

        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
        ax.set_xlabel(str(vars[0]))
        ax.set_ylabel(str(vars[1]))
        ax.set_zlabel("f")
        ax.set_title(title)

        self.canvas.draw()

    def clear_outputs(self):
        self.f_output.clear()
        self.grad_output.clear()
        self.hess_output.clear()
        self.eval_output.clear()
        self.figure.clear()
        self.canvas.draw()
