import sys
import math
import sympy as sp
import numpy as np
from typing import Callable
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QFrame,
    QTextEdit,
)
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import Qt

# Configuración de Matplotlib
matplotlib.use("QtAgg")


# =============================================================================
# PANTALLA: MÉTODO DE LA REGIÓN DORADA (Golden Section)
# =============================================================================
class GoldenSectionScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)

        # --- Lado Izquierdo ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        title_label = QLabel("Método de la Región Dorada")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 10px;"
        )

        self.params: dict[str, QLineEdit] = {}
        param_font = QFont()
        param_font.setPointSize(10)

        param_list: list[tuple[str, str, str]] = [
            ("f", "Función f(x)", "-x**2 + 4*x + 5"),
            ("a_int", "Límite inferior (a)", "3"),
            ("b_int", "Límite superior (b)", "7"),
            ("n_iter", "Número de iteraciones", "3"),
        ]

        left_layout.addWidget(title_label)
        for name, text, default_val in param_list:
            left_layout.addWidget(QLabel(text))
            line_edit = QLineEdit(default_val)
            line_edit.setFont(param_font)
            self.params[name] = line_edit
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

        # Resultados
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.Shape.StyledPanel)
        results_layout = QHBoxLayout(results_frame)
        self.result_p_label = QLabel("Óptimo: --")
        self.result_i_label = QLabel("Máximo: --")
        results_layout.addWidget(self.result_p_label)
        results_layout.addWidget(self.result_i_label)

        # Tabla de iteraciones
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(7)
        self.table_widget.setHorizontalHeaderLabels(
            ["Iteración", "Intervalo", "p1", "I(p1)", "p2", "I(p2)", "Nuevo Intervalo"]
        )
        self.table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table_widget.verticalHeader().setVisible(False)

        # Gráfico
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        right_layout.addWidget(QLabel("Resultados Finales:"))
        right_layout.addWidget(results_frame)
        right_layout.addWidget(QLabel("Tabla de Iteraciones:"))
        right_layout.addWidget(self.table_widget, 1)
        right_layout.addWidget(QLabel("Gráfico de la Función:"))
        right_layout.addWidget(self.canvas, 2)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 3)

        self.calculate_button.clicked.connect(self.run_calculation)

    def golden_section_max(
        self, f: Callable[[float], float], a: float, b: float, n_iter: int = 3
    ):
        phi = (math.sqrt(5) - 1) / 2
        results: list[dict] = []
        puntos: list[tuple[float, float]] = []
        for i in range(1, n_iter + 1):
            L = b - a
            p1 = a + (1 - phi) * L
            p2 = a + phi * L
            f1, f2 = f(p1), f(p2)
            puntos.extend([(p1, f1), (p2, f2)])
            new_interval_str = (
                f"[{p1:.4f}, {b:.4f}]" if f1 < f2 else f"[{a:.4f}, {p2:.4f}]"
            )
            results.append(
                {
                    "Iteración": i,
                    "Intervalo": f"[{a:.4f}, {b:.4f}]",
                    "p1": p1,
                    "I(p1)": f1,
                    "p2": p2,
                    "I(p2)": f2,
                    "Nuevo intervalo": new_interval_str,
                }
            )
            if f1 < f2:
                a = p1
            else:
                b = p2
        return (a + b) / 2, results, puntos

    def run_calculation(self):
        try:
            # Parse numeric parameters
            a_int = float(self.params["a_int"].text())
            b_int = float(self.params["b_int"].text())
            n_iter = int(self.params["n_iter"].text())
            
            # Parse function
            f_str = self.params["f"].text()
            x = sp.Symbol("x")
            f_expr = sp.sympify(f_str)
            f_lambda = sp.lambdify(x, f_expr, "numpy")
            
            # Wrapper to ensure float return and handle numpy types if needed
            def ingreso(val):
                return float(f_lambda(val))

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error de Entrada",
                f"Error al procesar la función o los parámetros:\n{str(e)}",
            )
            return

        p_opt, tabla, puntos = self.golden_section_max(
            ingreso, a_int, b_int, n_iter
        )
        I_opt = ingreso(p_opt)

        self.result_p_label.setText(f"<b>Óptimo:</b> {p_opt:.4f}")
        self.result_i_label.setText(f"<b>f(x):</b> {I_opt:.4f}")

        self.populate_table(tabla)
        self.plot_function(ingreso, a_int, b_int, puntos, p_opt, I_opt)

    def populate_table(self, tabla):
        self.table_widget.setRowCount(len(tabla))
        for row, item in enumerate(tabla):
            self.table_widget.setItem(row, 0, QTableWidgetItem(str(item["Iteración"])))
            self.table_widget.setItem(row, 1, QTableWidgetItem(item["Intervalo"]))
            self.table_widget.setItem(row, 2, QTableWidgetItem(f"{item['p1']:.4f}"))
            self.table_widget.setItem(row, 3, QTableWidgetItem(f"{item['I(p1)']:.4f}"))
            self.table_widget.setItem(row, 4, QTableWidgetItem(f"{item['p2']:.4f}"))
            self.table_widget.setItem(row, 5, QTableWidgetItem(f"{item['I(p2)']:.4f}"))
            self.table_widget.setItem(row, 6, QTableWidgetItem(item["Nuevo intervalo"]))

    def plot_function(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        puntos: list[tuple[float, float]],
        p_opt: float,
        I_opt: float,
    ):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Use numpy for smoother plotting
        x_vals = np.linspace(a, b, 200)
        try:
            # Handle if f can take array input directly (which lambdify usually supports)
            y_vals = f(x_vals)
        except:
            # Fallback for functions that might not vectorize well automatically
            y_vals = np.array([f(x) for x in x_vals])

        ax.plot(
            x_vals,
            y_vals,
            label="f(x)",
            color="blue",
        )

        for i, (px, Ix) in enumerate(puntos):
            label = f"Iter. {math.ceil((i+1)/2)}" if i % 2 == 0 else None
            ax.scatter(
                px, Ix, color="green", s=50, marker="x", label=label
            )

        ax.scatter(
            p_opt,
            I_opt,
            color="red",
            s=100,
            zorder=5,
            label=f"Optimum ({p_opt:.4f}, {I_opt:.4f})",
        )

        ax.set_title("Método de la Región Dorada")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()


# =============================================================================
# PANTALLA: MÉTODO NEWTON-RAPHSON
# =============================================================================
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


# =============================================================================
# PANTALLA: CÁLCULO SIMBÓLICO (Gradiente y Hessiano)
# =============================================================================
class SymbolicCalcScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)

        # --- Lado Izquierdo ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        title_label = QLabel("Gradiente y Hessiano")
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


# =============================================================================
# PANTALLA: BIENVENIDA (WelcomeScreen)
# =============================================================================
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


# =============================================================================
# PANTALLA: MENÚ DE PROGRAMACIÓN NO LINEAL
# =============================================================================
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
        self.button2 = QPushButton("Método de la Región Dorada")
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


# =============================================================================
# VENTANA PRINCIPAL (MainWindow)
# =============================================================================
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
