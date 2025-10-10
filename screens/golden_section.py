import math
from typing import Callable
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
    QFrame,
)
from PySide6.QtGui import QFont


class GoldenSectionScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)

        # --- Lado Izquierdo ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        title_label = QLabel("Método de la Sección Áurea")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 10px;"
        )

        self.params: dict[str, QLineEdit] = {}
        param_font = QFont()
        param_font.setPointSize(10)

        param_list: list[tuple[str, str, str]] = [
            ("a_coef", "Coeficiente a", "-1.5"),
            ("b_coef", "Coeficiente b", "15"),
            ("c_coef", "Coeficiente c", "-20"),
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
        self.result_p_label = QLabel("Precio óptimo: --")
        self.result_i_label = QLabel("Ingreso máximo: --")
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
            p = {name: float(widget.text()) for name, widget in self.params.items()}
            p["n_iter"] = int(p["n_iter"])
        except ValueError:
            QMessageBox.critical(
                self,
                "Error de Entrada",
                "Por favor, ingresa valores numéricos válidos.",
            )
            return

        def ingreso(pr: float) -> float:
            return p["a_coef"] * pr**2 + p["b_coef"] * pr + p["c_coef"]

        p_opt, tabla, puntos = self.golden_section_max(
            ingreso, p["a_int"], p["b_int"], p["n_iter"]
        )
        I_opt = ingreso(p_opt)

        self.result_p_label.setText(f"<b>Precio óptimo:</b> {p_opt*100:.2f} $")
        self.result_i_label.setText(f"<b>Ingreso máximo:</b> {I_opt*1000:.2f} $")

        self.populate_table(tabla)
        self.plot_function(ingreso, p["a_int"], p["b_int"], puntos, p_opt, I_opt)

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

        p_values: list[float] = [a + i * (b - a) / 200 for i in range(201)]
        I_values: list[float] = [f(p) for p in p_values]
        ax.plot(
            [p * 100 for p in p_values],
            [I * 1000 for I in I_values],
            label="Ingreso ($)",
            color="blue",
        )

        for i, (px, Ix) in enumerate(puntos):
            label = f"Iter. {math.ceil((i+1)/2)}" if i % 2 == 0 else None
            ax.scatter(
                px * 100, Ix * 1000, color="green", s=50, marker="x", label=label
            )

        ax.scatter(
            p_opt * 100,
            I_opt * 1000,
            color="red",
            s=100,
            zorder=5,
            label=f"Máximo ({p_opt*100:.2f}$, {I_opt*1000:.2f}$)",
        )

        ax.set_title("Maximización de Ingresos - Sección Áurea")
        ax.set_xlabel("Precio ($)")
        ax.set_ylabel("Ingreso ($)")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()
