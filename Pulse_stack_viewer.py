import sys
import glob
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# --- Global Configuration ---
DRIFT_COLORS = ["#00a4e4", "red", "#34bf49"]

# --- 1. Helper Functions ---
flat_huslmap = matplotlib.colors.ListedColormap(sns.color_palette("husl", 256))  # type: ignore


def L_debias(Q, U, off_pulse_mask, nsigma=1.0):
    Q = np.array(Q)
    U = np.array(U)
    L_meas = np.sqrt(Q**2 + U**2)

    if Q.ndim == 1:
        sigma_q = np.std(Q[off_pulse_mask])
        sigma_u = np.std(U[off_pulse_mask])
    else:
        sigma_q = np.std(Q[:, off_pulse_mask], axis=1, keepdims=True)
        sigma_u = np.std(U[:, off_pulse_mask], axis=1, keepdims=True)

    sigma = np.sqrt(0.5 * (sigma_q**2 + sigma_u**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_calc = np.sqrt((sigma_q**2 * Q**2 + sigma_u**2 * U**2) / L_meas**2)
        sigma = np.where(L_meas > 0, sigma_calc, sigma)

    L_debiased = np.full_like(L_meas, np.nan)
    valid_mask = L_meas > sigma * nsigma
    L_debiased[valid_mask] = np.sqrt(L_meas[valid_mask] ** 2 - sigma[valid_mask] ** 2)
    return L_debiased, sigma


def mpl_to_pg_cmap(mpl_cmap, n_colors=256, add_gray_for_nan=False):
    if isinstance(mpl_cmap, str):
        cmap = matplotlib.colormaps[mpl_cmap]
    else:
        cmap = mpl_cmap

    if add_gray_for_nan:
        gray = np.array([[128, 128, 128, 255]])
        mpl_colors = cmap(np.linspace(0, 1, n_colors)) * 255
        mpl_colors = mpl_colors.astype(np.uint8)
        colors = np.vstack([gray, mpl_colors])
        pos = np.concatenate([[0.0], np.linspace(0.001, 1.0, n_colors)])
    else:
        colors = (cmap(np.linspace(0, 1, n_colors)) * 255).astype(np.uint8)
        pos = np.linspace(0.0, 1.0, n_colors)

    return pg.ColorMap(pos, colors)


# --- 2. Advanced Panel Class ---
class PolPanel(QtWidgets.QWidget):
    def __init__(
        self,
        data,
        title,
        cmap,
        symmetric=False,
        fixed_range=None,
        parent=None,
    ):
        super().__init__(parent)
        self.data = data
        self.title = title
        self.symmetric = symmetric
        self.fixed_range = fixed_range

        finite_vals = data[np.isfinite(data)]
        if len(finite_vals) > 0:
            self.d_min, self.d_max = float(np.min(finite_vals)), float(
                np.max(finite_vals)
            )
        else:
            self.d_min, self.d_max = 0.0, 1.0

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(2)

        self.lbl_title = QtWidgets.QLabel(title)
        self.lbl_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.layout.addWidget(self.lbl_title)

        self.gv = pg.GraphicsView()
        self.plot_item = pg.PlotItem()
        self.plot_item.setLabel("top", "")

        n_pulses, nbin = self.data.shape
        self.plot_item.setXRange(0, nbin - 1, padding=0)
        self.plot_item.setYRange(0, n_pulses - 1, padding=0)
        self.plot_item.setMenuEnabled(False)
        self.gv.setCentralItem(self.plot_item)

        self.layout.addWidget(self.gv, 1)

        self.img = pg.ImageItem(self.data.T)
        self.plot_item.addItem(self.img)

        # ** NEW: Double-layered curves for outline effect **
        self.drift_styles = [
            {"color": DRIFT_COLORS[0], "style": QtCore.Qt.SolidLine, "width": 2.0},
            {"color": DRIFT_COLORS[1], "style": QtCore.Qt.DashLine, "width": 2.0},
            {"color": DRIFT_COLORS[2], "style": QtCore.Qt.DotLine, "width": 2.0},
        ]

        # drift_curves stores tuples: (outline_curve, main_curve)
        self.drift_curves = []

        for style in self.drift_styles:
            # 1. Create Outline Curve (Thicker, White)
            outline_pen = pg.mkPen(
                color="w", width=style["width"] + 1, style=style["style"]
            )
            outline_curve = pg.PlotCurveItem(pen=outline_pen, connect="finite")

            # --- MODIFICATION: Only add outline for PA panel ---
            if self.title == "PA":
                self.plot_item.addItem(outline_curve)

            # 2. Create Main Curve (Colored)
            main_pen = pg.mkPen(
                color=style["color"], width=style["width"], style=style["style"]
            )
            main_curve = pg.PlotCurveItem(pen=main_pen, connect="finite")
            self.plot_item.addItem(main_curve)

            self.drift_curves.append((outline_curve, main_curve))

        if fixed_range:
            values = fixed_range
            self.cbar = pg.ColorBarItem(values=values, interactive=False)
        else:
            if self.symmetric:
                lim = np.round(max(abs(self.d_min), abs(self.d_max)) * 0.1, 3)
                if lim == 0:
                    lim = 1.0
                values = (-lim, lim)
            else:
                values = np.round((self.d_min, self.d_max * 0.3), 3)
            self.cbar = pg.ColorBarItem(values=values, interactive=True)

        self.cbar.setImageItem(self.img, insert_in=self.plot_item)
        self.cbar.setColorMap(cmap)
        self.cbar.setFixedWidth(55)

        small_font = QtGui.QFont()
        small_font.setPointSize(12)
        self.cbar.axis.setStyle(tickFont=small_font, tickTextOffset=5)

        for child in self.cbar.childItems():
            if isinstance(child, pg.GradientEditorItem):
                child.setMaxWidth(14)

        self.ctrl_widget = QtWidgets.QWidget()
        self.ctrl_layout = QtWidgets.QHBoxLayout(self.ctrl_widget)
        self.ctrl_layout.setContentsMargins(0, 2, 0, 0)
        self.ctrl_layout.setSpacing(2)

        self.ctrl_widget.setFixedHeight(35)
        self.ctrl_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )

        control_font = QtGui.QFont("Arial", 14)

        self.spin_min = pg.SpinBox(value=values[0], dec=True)
        self.spin_min.setFont(control_font)
        self.spin_min.setFixedWidth(55)
        self.spin_min.setToolTip("Min Level")

        self.spin_max = pg.SpinBox(value=values[1], dec=True)
        self.spin_max.setFont(control_font)
        self.spin_max.setFixedWidth(55)
        self.spin_max.setToolTip("Max Level")

        self.btn_auto = QtWidgets.QPushButton("Auto")
        self.btn_auto.setToolTip("Auto Scale")
        self.btn_auto.setFixedWidth(60)
        self.btn_auto.setFont(QtGui.QFont("Arial", 14))
        self.btn_auto.clicked.connect(self.auto_scale)

        self.ctrl_layout.addStretch()
        lbl_min = QtWidgets.QLabel("Min:")
        lbl_min.setFont(QtGui.QFont("Arial", 14))
        self.ctrl_layout.addWidget(lbl_min)
        self.ctrl_layout.addWidget(self.spin_min)
        self.ctrl_layout.addSpacing(10)
        lbl_max = QtWidgets.QLabel("Max:")
        lbl_max.setFont(QtGui.QFont("Arial", 14))
        self.ctrl_layout.addWidget(lbl_max)
        self.ctrl_layout.addWidget(self.spin_max)
        self.ctrl_layout.addSpacing(10)
        self.ctrl_layout.addWidget(self.btn_auto)
        self.ctrl_layout.addStretch()

        self.layout.addWidget(self.ctrl_widget, 0)

        if fixed_range is not None:
            self.spin_min.setReadOnly(True)
            self.spin_max.setReadOnly(True)
            self.btn_auto.setEnabled(False)
        else:
            self.spin_min.sigValueChanged.connect(self.on_spin_changed)
            self.spin_max.sigValueChanged.connect(self.on_spin_changed)
            self.cbar.sigLevelsChanged.connect(self.on_cbar_changed)
            self.img.setLevels(values)

        self.cursor_v = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(color="r", width=1.0)
        )
        self.cursor_h = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(color="r", width=1.0)
        )
        self.plot_item.addItem(self.cursor_v)
        self.plot_item.addItem(self.cursor_h)
        self.set_cursor(None, None)

    def auto_scale(self):
        if self.symmetric:
            lim = max(abs(self.d_min), abs(self.d_max))
            if lim == 0:
                lim = 1.0
            self.set_levels(-lim, lim)
        else:
            self.set_levels(self.d_min, self.d_max)

    def set_levels(self, mn, mx):
        self.spin_min.blockSignals(True)
        self.spin_max.blockSignals(True)
        self.cbar.blockSignals(True)
        self.spin_min.setValue(mn)
        self.spin_max.setValue(mx)
        self.cbar.setLevels((mn, mx))
        self.spin_min.blockSignals(False)
        self.spin_max.blockSignals(False)
        self.cbar.blockSignals(False)

    def on_spin_changed(self):
        mn = self.spin_min.value()
        mx = self.spin_max.value()
        if self.symmetric:
            sender = self.sender()
            limit = abs(mn) if sender == self.spin_min else abs(mx)
            mn, mx = -limit, limit
            self.spin_min.blockSignals(True)
            self.spin_max.blockSignals(True)
            self.spin_min.setValue(mn)
            self.spin_max.setValue(mx)
            self.spin_min.blockSignals(False)
            self.spin_max.blockSignals(False)
        self.cbar.setLevels((mn, mx))

    def on_cbar_changed(self):
        mn, mx = self.cbar.levels()
        if self.symmetric:
            limit = max(abs(mn), abs(mx))
            if abs(limit - abs(mn)) > 1e-5 or abs(limit - abs(mx)) > 1e-5:
                mn, mx = -limit, limit
                self.cbar.setLevels((mn, mx))
        self.spin_min.blockSignals(True)
        self.spin_max.blockSignals(True)
        self.spin_min.setValue(mn)
        self.spin_max.setValue(mx)
        self.spin_min.blockSignals(False)
        self.spin_max.blockSignals(False)

    def set_cursor(self, x, y):
        if x is None or y is None:
            self.cursor_v.hide()
            self.cursor_h.hide()
            return
        self.cursor_v.setPos(x)
        self.cursor_h.setPos(y)
        self.cursor_v.show()
        self.cursor_h.show()

    def update_drift_bands(self, drift_params_list, visibilities):
        """
        visibilities: list of bools [True, False, True] indicating if band i is visible
        """
        n_pulses, n_bins = self.data.shape
        phi_per_bin = 1 / n_bins
        phi_bins = np.array([0, n_bins - 1])
        phi_deg = phi_bins * phi_per_bin

        for i, (fp2, fp3, theta0) in enumerate(drift_params_list):
            outline_item, main_item = self.drift_curves[i]

            # Check visibility
            is_visible = visibilities[i]
            outline_item.setVisible(is_visible)
            main_item.setVisible(is_visible)

            # If invisible or invalid parameters, clear data and skip
            if not is_visible or abs(fp3) < 1e-5:
                outline_item.setData([], [])
                main_item.setData([], [])
                continue

            shift = -theta0 / (2 * np.pi)
            k_center = int(-shift)
            k_range = int(n_pulses * abs(fp3)) + int(1 * abs(fp2)) + 5

            ks = np.arange(k_center - k_range, k_center + k_range + 1)

            x_plot = []
            y_plot = []

            for k in ks:
                n_vals = 1 / fp3 * (phi_deg * fp2 + k + shift)
                x_plot.extend(phi_bins)
                x_plot.append(np.nan)
                y_plot.extend(n_vals)
                y_plot.append(np.nan)

            # Update both curves with the same data
            x_arr = np.array(x_plot)
            y_arr = np.array(y_plot)
            outline_item.setData(x_arr, y_arr)
            main_item.setData(x_arr, y_arr)


# --- 3. Main Logic ---
def main():
    app = pg.mkQApp()

    # Try to load existing files on startup, but don't error out
    npy_files = sorted(glob.glob("*.npy"))
    current_idx = 0

    win = QtWidgets.QMainWindow()
    central = QtWidgets.QWidget()
    main_layout = QtWidgets.QHBoxLayout(central)
    main_layout.setContentsMargins(5, 5, 5, 5)
    main_layout.setSpacing(5)

    panels_layout = QtWidgets.QHBoxLayout()
    panels = []

    ctrl = QtWidgets.QFrame()
    ctrl.setFrameShape(QtWidgets.QFrame.StyledPanel)
    ctrl.setMaximumWidth(170)
    ctrl.setMinimumWidth(150)
    ctrl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

    ctrl_layout = QtWidgets.QVBoxLayout(ctrl)
    ctrl_layout.setContentsMargins(5, 10, 5, 10)
    ctrl_layout.setSpacing(10)
    ctrl_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)

    def add_section_label(text, layout, color=None):
        lbl = QtWidgets.QLabel(text)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        if color:
            lbl.setStyleSheet(f"font-weight: bold; margin-top: 5px; color: {color};")
        else:
            lbl.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(lbl)

    # --- NEW: Import Button ---
    btn_import = QtWidgets.QPushButton("Import Data")
    btn_import.setToolTip("Open .npy files")
    btn_import.setFixedHeight(35)
    #
    btn_import.setStyleSheet(
        """
        QPushButton {
            background-color: #0078d7; 
            color: white; 
            font-weight: bold; 
            font-size: 14px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #1084e3;
        }
    """
    )
    ctrl_layout.addWidget(btn_import)

    line_import = QtWidgets.QFrame()
    line_import.setFrameShape(QtWidgets.QFrame.HLine)
    line_import.setFrameShadow(QtWidgets.QFrame.Sunken)
    ctrl_layout.addWidget(line_import)
    # --------------------------

    # New helper to create a header row with a checkbox
    def add_checkable_section(text, layout, color=None):
        container = QtWidgets.QWidget()
        h_layout = QtWidgets.QHBoxLayout(container)
        h_layout.setContentsMargins(0, 5, 0, 0)
        # --- MODIFICATION: Increased spacing between checkbox and label ---
        h_layout.setSpacing(15)
        h_layout.setAlignment(QtCore.Qt.AlignCenter)

        chk = QtWidgets.QCheckBox()
        chk.setChecked(True)  # Default on

        lbl = QtWidgets.QLabel(text)
        if color:
            lbl.setStyleSheet(f"font-weight: bold; color: {color};")
        else:
            lbl.setStyleSheet("font-weight: bold;")

        h_layout.addWidget(chk)
        h_layout.addWidget(lbl)

        layout.addWidget(container)
        return chk

    def add_spin_row(label_text, spinbox, layout):
        container = QtWidgets.QWidget()
        h_layout = QtWidgets.QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(5)
        lbl = QtWidgets.QLabel(label_text)
        lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        spinbox.setFixedWidth(60)
        spinbox.setAlignment(QtCore.Qt.AlignCenter)
        h_layout.addWidget(lbl)
        h_layout.addWidget(spinbox)
        layout.addWidget(container, 0, QtCore.Qt.AlignHCenter)

    add_section_label("File", ctrl_layout)

    btn_container = QtWidgets.QWidget()
    btn_layout = QtWidgets.QHBoxLayout(btn_container)
    btn_layout.setContentsMargins(0, 0, 0, 0)
    btn_layout.setSpacing(5)
    prev_btn = QtWidgets.QPushButton("<")
    next_btn = QtWidgets.QPushButton(">")
    prev_btn.setFixedWidth(40)
    next_btn.setFixedWidth(40)
    btn_layout.addWidget(prev_btn)
    btn_layout.addWidget(next_btn)
    ctrl_layout.addWidget(btn_container, 0, QtCore.Qt.AlignHCenter)

    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    ctrl_layout.addWidget(line)

    add_section_label("Bin Range", ctrl_layout)
    min_spin = pg.SpinBox(value=490, int=True)
    max_spin = pg.SpinBox(value=530, int=True)
    add_spin_row("Min:", min_spin, ctrl_layout)
    add_spin_row("Max:", max_spin, ctrl_layout)

    line2 = QtWidgets.QFrame()
    line2.setFrameShape(QtWidgets.QFrame.HLine)
    line2.setFrameShadow(QtWidgets.QFrame.Sunken)
    ctrl_layout.addWidget(line2)

    add_section_label("Pulse Range", ctrl_layout)
    y_min_spin = pg.SpinBox(value=0, int=True)
    y_max_spin = pg.SpinBox(value=0, int=True)
    add_spin_row("Min:", y_min_spin, ctrl_layout)
    add_spin_row("Max:", y_max_spin, ctrl_layout)

    line3 = QtWidgets.QFrame()
    line3.setFrameShape(QtWidgets.QFrame.HLine)
    line3.setFrameShadow(QtWidgets.QFrame.Sunken)
    ctrl_layout.addWidget(line3)

    def add_drift_control(label, spin_val, spin_range, step):
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)

        lbl = QtWidgets.QLabel(label)
        spin = pg.SpinBox(value=spin_val, bounds=spin_range, step=step)
        spin.setFixedWidth(80)

        row_layout.addWidget(lbl)
        row_layout.addWidget(spin)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(spin_range[0] * 100), int(spin_range[1] * 100))
        slider.setValue(int(spin_val * 100))
        slider.setFixedHeight(12)
        slider.setStyleSheet(
            """
            QSlider::groove:horizontal { height: 4px; background: #bfbfbf; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #6aa9ff; height: 4px; border-radius: 2px; }
            QSlider::add-page:horizontal { background: #bfbfbf; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #8a8a8a; border: 1px solid #6e6e6e; width: 10px; height: 10px; margin: -4px 0; border-radius: 5px; }
            """
        )

        ctrl_layout.addWidget(row_widget, 0, QtCore.Qt.AlignHCenter)
        ctrl_layout.addWidget(slider)
        ctrl_layout.addSpacing(2)
        return spin, slider

    drift_labels = ["Drift Band 1", "Drift Band 2", "Drift Band 3"]
    drift_widgets = []

    for i in range(3):
        # Use add_checkable_section instead of add_section_label
        chk_box = add_checkable_section(
            drift_labels[i], ctrl_layout, color=DRIFT_COLORS[i]
        )

        fp2_s, fp2_sl = add_drift_control(f"1/P2:", 0, (-200, 200), 1)
        fp3_s, fp3_sl = add_drift_control(f"1/P3:", 0.1, (-0.5, 0.5), 0.002)
        th_s, th_sl = add_drift_control(f"θ [rad]:", i, (0.0, 2 * np.pi), 0.02)

        drift_widgets.append(
            {
                "checkbox": chk_box,
                "fp2_spin": fp2_s,
                "fp2_slider": fp2_sl,
                "fp3_spin": fp3_s,
                "fp3_slider": fp3_sl,
                "th_spin": th_s,
                "th_slider": th_sl,
            }
        )

        if i < 2:
            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.HLine)
            sep.setFrameShadow(QtWidgets.QFrame.Sunken)
            ctrl_layout.addWidget(sep)

    ctrl_layout.addStretch()

    hover_connections = []

    def clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def disconnect_hover():
        for conn in hover_connections:
            try:
                conn.disconnect()
            except Exception:
                pass
        hover_connections.clear()

    def make_hover_handler(panel_idx, panels_ref):
        panel = panels_ref[panel_idx]

        def on_mouse_moved(pos):
            vb = panel.plot_item.getViewBox()
            view_pos = vb.mapSceneToView(pos)
            x, y = view_pos.x(), view_pos.y()
            for p in panels_ref:
                p.set_cursor(x, y)

        return on_mouse_moved

    def update_range(nbin, n_pulse, panels_ref, master_plot):
        min_spin.setRange(0, nbin)
        max_spin.setRange(0, nbin)
        y_min_spin.setRange(0, n_pulse)
        y_max_spin.setRange(0, n_pulse)

        mn, mx = min_spin.value(), max_spin.value()
        if mn >= mx:
            mx = min(nbin, mn + 1)
            max_spin.setValue(mx)
        master_plot.setXRange(mn, mx, padding=0)

        ymn, ymx = y_min_spin.value(), y_max_spin.value()
        if ymn >= ymx:
            ymx = min(n_pulse - 1, ymn + 1)
            y_max_spin.setValue(ymx)
        master_plot.setYRange(ymn, ymx, padding=0)

    def update_drift_lines():
        drift_params = []
        visibilities = []

        for d in drift_widgets:
            drift_params.append(
                (d["fp2_spin"].value(), d["fp3_spin"].value(), d["th_spin"].value())
            )
            visibilities.append(d["checkbox"].isChecked())

        for p in panels:
            p.update_drift_bands(drift_params, visibilities)

    def sync_spin_slider(spin, slider):
        def spin_changed():
            slider.blockSignals(True)
            slider.setValue(int(spin.value() * 100))
            slider.blockSignals(False)
            update_drift_lines()

        def slider_changed():
            spin.blockSignals(True)
            spin.setValue(slider.value() / 100.0)
            spin.blockSignals(False)
            update_drift_lines()

        spin.sigValueChanged.connect(spin_changed)
        slider.valueChanged.connect(slider_changed)

    for d in drift_widgets:
        sync_spin_slider(d["fp2_spin"], d["fp2_slider"])
        sync_spin_slider(d["fp3_spin"], d["fp3_slider"])
        sync_spin_slider(d["th_spin"], d["th_slider"])
        # Connect checkbox toggle
        d["checkbox"].stateChanged.connect(update_drift_lines)

    def load_file(idx):
        nonlocal panels, panels_layout, hover_connections

        if not npy_files:
            return

        fname = npy_files[idx]
        try:
            data = np.load(fname)
        except Exception as e:
            QtWidgets.QMessageBox.critical(win, "Error", f"Failed to load {fname}\n{e}")
            return

        if data.ndim != 3 or data.shape[0] < 4:
            return

        I, Q, U, V = data
        n_pulse, nbin = I.shape
        L, _ = L_debias(Q, U, off_pulse_mask=np.r_[0:400, 600:nbin], nsigma=1.0)
        with np.errstate(invalid="ignore"):
            PA = 0.5 * np.degrees(np.arctan2(U, Q))
        PA = np.where(np.isnan(L), np.nan, PA)

        cm_husl = mpl_to_pg_cmap(flat_huslmap)
        cm_afmhot = mpl_to_pg_cmap("afmhot_r")
        cm_bwr = mpl_to_pg_cmap("bwr")

        disconnect_hover()
        clear_layout(panels_layout)
        panels = []

        p_I = PolPanel(I, "I", cm_afmhot)
        p_L = PolPanel(L, "L", cm_afmhot)
        p_PA = PolPanel(PA, "PA", cm_husl, fixed_range=(-90, 90))
        p_V = PolPanel(V, "V", cm_bwr, symmetric=True)

        for p in (p_I, p_L, p_PA, p_V):
            panels.append(p)
            panels_layout.addWidget(p)

        master = panels[0].plot_item
        for p in panels[1:]:
            p.plot_item.setXLink(master)
            p.plot_item.setYLink(master)

        for idx_p, p in enumerate(panels):
            conn = p.gv.scene().sigMouseMoved.connect(make_hover_handler(idx_p, panels))
            hover_connections.append(conn)

        y_min_spin.blockSignals(True)
        y_max_spin.blockSignals(True)
        y_min_spin.setValue(0)
        y_max_spin.setValue(n_pulse - 1)
        y_min_spin.blockSignals(False)
        y_max_spin.blockSignals(False)

        update_range(nbin, n_pulse, panels, master)
        update_drift_lines()

        win.setWindowTitle(f"Pulse Stack Viewer - {os.path.basename(fname)}")

    def import_data():
        nonlocal npy_files, current_idx
        # Open File Dialog
        fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            win, "Import .npy Files", "", "NumPy files (*.npy);;All Files (*)"
        )

        if fnames:
            # Add unique files to the list
            new_files = [f for f in fnames if f not in npy_files]
            if new_files:
                npy_files.extend(sorted(new_files))
                # Load the first of the newly imported files
                # If list was empty, this starts from index 0
                # If list had 3 items and we add more, we might want to jump to the new one
                # For simplicity, if we were at "no data" state, load index 0.
                # Otherwise, keep looking at current file or jump to new?
                # Let's jump to the first new file.
                current_idx = npy_files.index(new_files[0])
                load_file(current_idx)
            else:
                QtWidgets.QMessageBox.information(
                    win, "Import", "Selected files are already in the list."
                )

    btn_import.clicked.connect(import_data)

    def on_next():
        nonlocal current_idx
        if not npy_files:
            return
        current_idx = (current_idx + 1) % len(npy_files)
        load_file(current_idx)

    def on_prev():
        nonlocal current_idx
        if not npy_files:
            return
        current_idx = (current_idx - 1) % len(npy_files)
        load_file(current_idx)

    next_btn.clicked.connect(on_next)
    prev_btn.clicked.connect(on_prev)

    def apply_range_signal():
        if panels:
            master = panels[0].plot_item
            nbin = panels[0].data.shape[1]
            n_pulse = panels[0].data.shape[0]
            update_range(nbin, n_pulse, panels, master)

    min_spin.sigValueChanged.connect(apply_range_signal)
    max_spin.sigValueChanged.connect(apply_range_signal)
    y_min_spin.sigValueChanged.connect(apply_range_signal)
    y_max_spin.sigValueChanged.connect(apply_range_signal)

    main_layout.addLayout(panels_layout, 1)
    main_layout.addWidget(ctrl, 0)

    win.setCentralWidget(central)

    screen = app.primaryScreen()
    if screen:
        avail = screen.availableGeometry()
        win.resize(int(avail.width() * 0.95), int(avail.height() * 0.95))
    else:
        win.resize(1200, 700)

    # Initial load if files exist in directory
    if npy_files:
        load_file(current_idx)

    win.show()
    app.exec()


if __name__ == "__main__":
    main()
