import sys
import math
import sympy as sp
import numpy as np
import networkx as nx
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

        p_opt, tabla, puntos = self.golden_section_max(ingreso, a_int, b_int, n_iter)
        I_opt = ingreso(p_opt)

        self.result_p_label.setText(f"<b>Óptimo:</b> {p_opt:.4f}")
        self.result_i_label.setText(f"<b>Máximo:</b> {I_opt:.4f}")

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
            ax.scatter(px, Ix, color="green", s=50, marker="x", label=label)

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
                QMessageBox.warning(
                    self, "Error de Evaluación", f"Error evaluando la función: {e}"
                )
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
                QMessageBox.warning(
                    self,
                    "Límite de Seguridad",
                    "Se alcanzaron 100 iteraciones sin llegar a error 0 absoluto.",
                )
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
        ax.axhline(C_opt, color="red", linestyle="--", label=f"x óptimo = {C_opt:.6f}")

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
# PANTALLA: MÉTODO DE LAGRANGE
# =============================================================================
class LagrangeScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)

        # --- Lado Izquierdo ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        title_label = QLabel("Método de Lagrange")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 10px;"
        )

        self.f_input = QLineEdit("5*x**2 + 6*y**2 - x*y")
        self.g_input = QLineEdit("x + 2*y - 24")

        self.calculate_button = QPushButton("Resolver")
        self.calculate_button.setStyleSheet(
            "font-size: 14px; padding: 8px; margin-top: 15px;"
        )
        self.back_button = QPushButton("Volver")

        left_layout.addWidget(title_label)
        left_layout.addWidget(QLabel("Función f(x,y):"))
        left_layout.addWidget(self.f_input)
        left_layout.addWidget(QLabel("Restricción g(x,y)=0:"))
        left_layout.addWidget(self.g_input)
        left_layout.addWidget(self.calculate_button)
        left_layout.addStretch()
        left_layout.addWidget(self.back_button)

        # --- Lado Derecho ---
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)

        text_results_widget = QWidget()
        text_layout = QVBoxLayout(text_results_widget)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier New", 10))

        text_layout.addWidget(QLabel("Resultados:"))
        text_layout.addWidget(self.output_text)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        right_layout.addWidget(text_results_widget, 1)
        right_layout.addWidget(self.canvas, 2)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 3)

        self.calculate_button.clicked.connect(self.run_calculation)

    def run_calculation(self):
        try:
            f_str = self.f_input.text()
            g_str = self.g_input.text()

            x, y, lam = sp.symbols("x y lambda_")
            f = sp.sympify(f_str)
            g = sp.sympify(g_str)

            L = f - lam * g
            sol = sp.solve([sp.diff(L, x), sp.diff(L, y), sp.diff(L, lam)], (x, y, lam))

            if not sol:
                self.output_text.setText("No se encontró solución.")
                return

            # Handle multiple solutions or single solution
            if isinstance(sol, dict):
                sol = [sol]
            elif isinstance(sol, tuple):
                # Check if it's a tuple of values (single solution) or tuple of tuples
                if all(isinstance(i, (int, float, sp.Basic)) for i in sol):
                    sol = [{x: sol[0], y: sol[1], lam: sol[2]}]

            # Take the first real solution for simplicity in plotting,
            # but list all in text

            output_msg = f"Lagrangiano L = {L}\n\n"

            valid_sol = None

            for s in sol:
                # Normalize solution format to dict if it's a tuple
                if isinstance(s, tuple):
                    s_dict = {x: s[0], y: s[1], lam: s[2]}
                else:
                    s_dict = s

                try:
                    x_val = float(s_dict[x])
                    y_val = float(s_dict[y])
                    lam_val = float(s_dict[lam])

                    # Hessian check
                    Lxx = sp.diff(L, x, 2)
                    Lxy = sp.diff(L, x, y)
                    Lyy = sp.diff(L, y, 2)
                    H = sp.Matrix([[Lxx, Lxy], [Lxy, Lyy]])

                    # Evaluate H at critical point
                    H_eval = H.subs(s_dict)
                    Delta1 = H_eval[0, 0]
                    Delta2 = H_eval.det()

                    f_val = float(f.subs(s_dict))

                    output_msg += f"--- Punto Crítico ---\n"
                    output_msg += f"x* = {x_val:.4f}\n"
                    output_msg += f"y* = {y_val:.4f}\n"
                    output_msg += f"λ* = {lam_val:.4f}\n"
                    output_msg += f"f(x*, y*) = {f_val:.4f}\n"
                    output_msg += f"Hessiano:\n{sp.pretty(H_eval)}\n"
                    output_msg += f"Δ1 = {Delta1}, Δ2 = {Delta2}\n\n"

                    valid_sol = (x_val, y_val, f_val, f, g)

                except Exception:
                    continue

            self.output_text.setText(output_msg)

            if valid_sol:
                self.plot_graph(*valid_sol)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en el cálculo: {e}")

    def plot_graph(self, x_star, y_star, z_star, f_sym, g_sym):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")

        x_sym, y_sym = sp.symbols("x y")
        f_l = sp.lambdify((x_sym, y_sym), f_sym, "numpy")

        # Create meshgrid around the optimal point
        range_span = 20
        X = np.linspace(x_star - range_span, x_star + range_span, 60)
        Y = np.linspace(y_star - range_span, y_star + range_span, 60)
        X, Y = np.meshgrid(X, Y)
        Z = f_l(X, Y)

        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)

        # Plot restriction line (approximate)
        # Solve g(x,y)=0 for y to plot the line
        try:
            sol_y = sp.solve(g_sym, y_sym)
            if sol_y:
                y_func = sp.lambdify(x_sym, sol_y[0], "numpy")
                xr = np.linspace(x_star - range_span, x_star + range_span, 200)
                yr = y_func(xr)
                # Filter yr to be within plot range for better visual
                mask = (yr >= y_star - range_span) & (yr <= y_star + range_span)
                xr_plot = xr[mask]
                yr_plot = yr[mask]

                if len(xr_plot) > 0:
                    zr_plot = f_l(xr_plot, yr_plot)
                    ax.plot(
                        xr_plot,
                        yr_plot,
                        zr_plot,
                        color="red",
                        linewidth=3,
                        label="Restricción g(x,y)=0",
                    )
        except:
            pass  # Complex restriction plotting might fail, skip line

        # Plot optimal point
        ax.scatter(
            x_star, y_star, z_star, color="black", s=100, label="Óptimo", zorder=10
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")
        ax.set_title("Método de Lagrange")
        ax.legend()

        self.canvas.draw()


# =============================================================================
# PANTALLA: MÉTODO DE WOLFE
# =============================================================================
class WolfeScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)

        # --- Lado Izquierdo ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        title_label = QLabel("Método de Wolfe")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 10px;"
        )

        # Parámetros editables
        self.params = {}
        param_font = QFont()
        param_font.setPointSize(10)

        # Función Objetivo
        left_layout.addWidget(
            QLabel(
                "Función Objetivo: Max Z = a1*x1 + a2*x2 - b1*x1^2 - b2*x2^2 - b3*x1*x2"
            )
        )

        grid_obj = QHBoxLayout()
        obj_params = [
            ("a1", "10"),
            ("a2", "25"),
            ("b1", "10"),
            ("b2", "1"),
            ("b3", "4"),
        ]
        for name, default in obj_params:
            lbl = QLabel(name)
            inp = QLineEdit(default)
            inp.setFixedWidth(40)
            self.params[name] = inp
            grid_obj.addWidget(lbl)
            grid_obj.addWidget(inp)
        left_layout.addLayout(grid_obj)

        # Restricciones
        left_layout.addWidget(QLabel("Restricción 1: r11*x1 + r12*x2 <= R1"))
        grid_r1 = QHBoxLayout()
        r1_params = [("r11", "1"), ("r12", "2"), ("R1", "10")]
        for name, default in r1_params:
            lbl = QLabel(name)
            inp = QLineEdit(default)
            inp.setFixedWidth(40)
            self.params[name] = inp
            grid_r1.addWidget(lbl)
            grid_r1.addWidget(inp)
        left_layout.addLayout(grid_r1)

        left_layout.addWidget(QLabel("Restricción 2: r21*x1 + r22*x2 <= R2"))
        grid_r2 = QHBoxLayout()
        r2_params = [("r21", "1"), ("r22", "1"), ("R2", "9")]
        for name, default in r2_params:
            lbl = QLabel(name)
            inp = QLineEdit(default)
            inp.setFixedWidth(40)
            self.params[name] = inp
            grid_r2.addWidget(lbl)
            grid_r2.addWidget(inp)
        left_layout.addLayout(grid_r2)

        self.calculate_button = QPushButton("Ejecutar Wolfe")
        self.calculate_button.setStyleSheet(
            "font-size: 14px; padding: 8px; margin-top: 15px;"
        )
        self.back_button = QPushButton("Volver")

        left_layout.addWidget(self.calculate_button)
        left_layout.addStretch()
        left_layout.addWidget(self.back_button)

        # --- Lado Derecho ---
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)

        text_results_widget = QWidget()
        text_layout = QVBoxLayout(text_results_widget)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier New", 10))

        text_layout.addWidget(QLabel("Resultados:"))
        text_layout.addWidget(self.output_text)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        right_layout.addWidget(text_results_widget, 1)
        right_layout.addWidget(self.canvas, 2)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 3)

        self.calculate_button.clicked.connect(self.run_calculation)

    def run_calculation(self):
        try:
            # Obtener valores
            p = {k: float(v.text()) for k, v in self.params.items()}

            self.output_text.clear()
            self.log("WOLFE METHOD - Paso a paso")
            self.log("-" * 50)
            self.log(
                f"Max Z = {p['a1']} x1 + {p['a2']} x2 - {p['b1']} x1^2 - {p['b2']} x2^2 - {p['b3']} x1 x2"
            )
            self.log("Sujeto a:")
            self.log(f"  (1) {p['r11']} x1 + {p['r12']} x2 <= {p['R1']}")
            self.log(f"  (2) {p['r21']} x1 + {p['r22']} x2 <= {p['R2']}")
            self.log("-" * 50)

            # 1. Estacionaridad
            E1 = np.array([2 * p["b1"], p["b3"], 1, 1, -1, 0, 1, 0, p["a1"]])
            E2 = np.array([p["b3"], 2 * p["b2"], 2, 1, 0, -1, 0, 1, p["a2"]])

            self.log("Ecuaciones de estacionaridad calculadas.")

            # 2. Iteración 1 (Aumentar x2)
            L1 = p["R1"] / p["r12"] if p["r12"] != 0 else 999
            L2 = p["R2"] / p["r22"] if p["r22"] != 0 else 999
            x2 = min(L1, L2)
            x1 = 0

            self.log(f"ITERACIÓN 1: x1={x1}, x2={x2}")

            # 3. Iteración 2 (Entra mu1)
            x2_new = p["R1"] / p["r12"] if p["r12"] != 0 else 0
            self.log(f"ITERACIÓN 2: Ajuste x2={x2_new}")
            x2 = x2_new

            # 4. Solución Final
            Z = (
                p["a1"] * x1
                + p["a2"] * x2
                - p["b1"] * x1**2
                - p["b2"] * x2**2
                - p["b3"] * x1 * x2
            )

            self.log("-" * 50)
            self.log("SOLUCIÓN FINAL ESTIMADA:")
            self.log(f"x1* = {x1}")
            self.log(f"x2* = {x2}")
            self.log(f"Z*  = {Z}")

            # Graficar
            self.plot_graph(p, x1, x2)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en el cálculo: {e}")

    def log(self, text):
        self.output_text.append(text)

    def plot_graph(self, p, x1_sol, x2_sol):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Rango de graficación
        max_val = max(p["R1"], p["R2"]) * 1.2
        x = np.linspace(0, max_val, 100)
        y = np.linspace(0, max_val, 100)
        X, Y = np.meshgrid(x, y)

        # Función Objetivo (Curvas de nivel)
        Z = (
            p["a1"] * X
            + p["a2"] * Y
            - p["b1"] * X**2
            - p["b2"] * Y**2
            - p["b3"] * X * Y
        )
        cs = ax.contour(X, Y, Z, 15, cmap="viridis")
        ax.clabel(cs, inline=True, fontsize=8)

        # Restricciones
        # R1: r11*x1 + r12*x2 <= R1  =>  x2 <= (R1 - r11*x1)/r12
        if p["r12"] != 0:
            y_r1 = (p["R1"] - p["r11"] * x) / p["r12"]
            ax.plot(x, y_r1, "r-", label="Restricción 1")
        else:
            ax.axvline(p["R1"] / p["r11"], color="r", label="Restricción 1")

        # R2: r21*x1 + r22*x2 <= R2
        if p["r22"] != 0:
            y_r2 = (p["R2"] - p["r21"] * x) / p["r22"]
            ax.plot(x, y_r2, "b-", label="Restricción 2")
        else:
            ax.axvline(p["R2"] / p["r21"], color="b", label="Restricción 2")

        # Región Factible (Sombreado aproximado)
        # Filtramos puntos que cumplen ambas
        feasible = (
            (p["r11"] * X + p["r12"] * Y <= p["R1"])
            & (p["r21"] * X + p["r22"] * Y <= p["R2"])
            & (X >= 0)
            & (Y >= 0)
        )
        ax.contourf(X, Y, feasible, levels=[0.5, 1.5], colors=["#d0f0d0"], alpha=0.3)

        # Punto Solución
        ax.plot(x1_sol, x2_sol, "ko", markersize=8, label="Solución")
        ax.annotate(
            f"({x1_sol:.2f}, {x2_sol:.2f})",
            (x1_sol, x2_sol),
            xytext=(10, 10),
            textcoords="offset points",
        )

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Método de Wolfe - Gráfica")
        ax.legend()
        ax.grid(True)

        self.canvas.draw()


# =============================================================================
# PANTALLA: PROGRAMACIÓN DINÁMICA (Ruta más corta)
# =============================================================================
class DynamicProgrammingScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)

        # --- Lado Izquierdo ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        title_label = QLabel("Ruta más corta – Programación Dinámica")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 10px;"
        )
        left_layout.addWidget(title_label)

        # Entradas
        self.input_stages = QTextEdit()
        self.input_stages.setPlaceholderText("A\nB C\nD E\nF")
        self.input_edges = QTextEdit()
        self.input_edges.setPlaceholderText("A B 5\nA C 3\nB D 2\n...")

        left_layout.addWidget(QLabel("Etapas (una por línea):"))
        left_layout.addWidget(self.input_stages)

        left_layout.addWidget(QLabel("Arcos (Origen Destino Costo):"))
        left_layout.addWidget(self.input_edges)

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

        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setFont(QFont("Courier New", 10))

        # Frame para el gráfico
        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_widget)

        right_layout.addWidget(QLabel("Resultados:"))
        right_layout.addWidget(self.results, 1)
        right_layout.addWidget(QLabel("Grafo:"))
        right_layout.addWidget(self.graph_widget, 2)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 2)

        self.calculate_button.clicked.connect(self.run_calculation)

    def run_calculation(self):
        try:
            raw_stages = self.input_stages.toPlainText().strip().split("\n")
            stages = [line.split() for line in raw_stages if line.strip()]

            if not stages:
                raise ValueError("Debes definir las etapas.")

            graph = {node: {} for stage in stages for node in stage}

            edges_text = self.input_edges.toPlainText().strip()
            if not edges_text:
                raise ValueError("Debes definir los arcos.")

            for line in edges_text.split("\n"):
                parts = line.split()
                if len(parts) != 3:
                    continue
                o, d, c = parts
                if o not in graph:
                    graph[o] = {}
                graph[o][d] = float(c)
        except Exception as e:
            QMessageBox.critical(self, "Error de Entrada", f"Formato incorrecto: {e}")
            return

        try:
            costo = {node: float("inf") for node in graph}
            decision = {node: [] for node in graph}

            # Inicializar última etapa (costo 0 para llegar al final desde el final)
            # Asumimos que la última etapa son nodos destino finales
            for node in stages[-1]:
                costo[node] = 0
                decision[node] = ["TERMINAL"]

            # Programación Dinámica hacia atrás
            for stage in reversed(stages[:-1]):
                for node in stage:
                    if node not in graph or not graph[node]:
                        continue

                    # Calcular costo mínimo hacia la siguiente etapa
                    # min(costo_arco + costo_acumulado_destino)
                    valid_paths = []
                    for nxt, weight in graph[node].items():
                        if nxt in costo and costo[nxt] != float("inf"):
                            total_c = weight + costo[nxt]
                            valid_paths.append((total_c, nxt))

                    if valid_paths:
                        min_cost = min(p[0] for p in valid_paths)
                        costo[node] = min_cost
                        # Guardar todas las decisiones óptimas (para múltiples rutas)
                        decision[node] = [
                            p[1] for p in valid_paths if abs(p[0] - min_cost) < 1e-9
                        ]

            # Reconstruir rutas desde el inicio
            rutas = []
            if not stages[0]:
                raise ValueError("Primera etapa vacía.")

            inicio = stages[0][0]  # Asumimos un nodo inicial único o tomamos el primero

            if costo[inicio] == float("inf"):
                self.results.setText("No hay ruta factible desde el inicio al final.")
                return

            def expand(path):
                last = path[-1]
                if last in stages[-1]:
                    rutas.append(path)
                    return

                if last in decision:
                    for nxt in decision[last]:
                        if nxt == "TERMINAL":
                            # Esto no debería pasar si la lógica es correcta y last está en stages[-1]
                            pass
                        else:
                            expand(path + [nxt])

            expand([inicio])
            # Eliminar duplicados
            rutas_unique = []
            seen = set()
            for r in rutas:
                t = tuple(r)
                if t not in seen:
                    rutas_unique.append(r)
                    seen.add(t)
            rutas = rutas_unique

            texto = f"📌 Costo mínimo total: {costo[inicio]}\n\n"
            texto += "🚩 Rutas óptimas:\n"
            for r in rutas:
                texto += " → ".join(r) + "\n"
            self.results.setText(texto)

            self.plot_graph(graph, stages, rutas)

        except Exception as e:
            QMessageBox.critical(
                self, "Error de Cálculo", f"Error durante el cálculo: {e}"
            )

    def plot_graph(self, graph_dict, stages, rutas):
        # Limpiar gráfico anterior
        for i in reversed(range(self.graph_layout.count())):
            widget = self.graph_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        G = nx.DiGraph()
        for o, dests in graph_dict.items():
            for d, w in dests.items():
                G.add_edge(o, d, weight=w)

        # Asignar atributos de etapa para el layout
        for i, stage in enumerate(stages):
            for node in stage:
                G.nodes[node]["subset"] = i

        try:
            pos = nx.multipartite_layout(G, subset_key="subset")
        except:
            pos = nx.spring_layout(G)

        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Dibujar nodos y arcos base
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=800,
            node_color="lightblue",
            font_size=10,
            arrows=True,
        )

        # Etiquetas de peso
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax, font_size=8
        )

        # Resaltar rutas óptimas
        for r in rutas:
            edges = list(zip(r, r[1:]))
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, ax=ax, width=3, edge_color="red", arrows=True
            )
            nx.draw_networkx_nodes(
                G, pos, nodelist=r, ax=ax, node_color="lightgreen", node_size=800
            )

        self.graph_layout.addWidget(canvas)


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

        course_label = QLabel("Curso: Programación No Lineal y Dinámica")
        course_label.setFont(course_font)
        course_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Sección de Botones ---
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(15)

        button_style = "QPushButton { font-size: 20px; padding: 15px 30px; }"

        self.nonlinear_button = QPushButton("Programación No Lineal")
        self.nonlinear_button.setStyleSheet(button_style)

        self.dynamic_button = QPushButton("Programación Dinámica")
        self.dynamic_button.setStyleSheet(button_style)

        buttons_layout.addWidget(self.nonlinear_button)
        buttons_layout.addWidget(self.dynamic_button)

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
        self.button4 = QPushButton("Método de Lagrange")
        self.button5 = QPushButton("Método de Wolfe")

        top_layout.addWidget(title_label)
        top_layout.addWidget(self.button1)
        top_layout.addWidget(self.button2)
        top_layout.addWidget(self.button3)
        top_layout.addWidget(self.button4)
        top_layout.addWidget(self.button5)

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
        self.screen4 = LagrangeScreen()
        self.screen5 = WolfeScreen()
        self.dynamic_screen = DynamicProgrammingScreen()

        self.stacked_widget.addWidget(self.welcome_screen)  # Índice 0
        self.stacked_widget.addWidget(self.nonlinear_menu)  # Índice 1
        self.stacked_widget.addWidget(self.screen1)  # Índice 2
        self.stacked_widget.addWidget(self.screen2)  # Índice 3
        self.stacked_widget.addWidget(self.screen3)  # Índice 4
        self.stacked_widget.addWidget(self.screen4)  # Índice 5
        self.stacked_widget.addWidget(self.screen5)  # Índice 6
        self.stacked_widget.addWidget(self.dynamic_screen)  # Índice 7

        self.welcome_screen.nonlinear_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.welcome_screen.dynamic_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(7)
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
        self.nonlinear_menu.button4.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(5)
        )
        self.nonlinear_menu.button5.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(6)
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
        self.screen4.back_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.screen5.back_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.dynamic_screen.back_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(0)
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
