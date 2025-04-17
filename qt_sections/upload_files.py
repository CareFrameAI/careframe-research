import sys
import difflib
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QMessageBox,
    QProgressDialog, QInputDialog, QTableView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDateTime, QAbstractTableModel, QModelIndex

# ----------------------------------------------------------------
# Candidate key names for fuzzy matching.
candidate_keys = {
    "patient_id": ['patient id', 'patient_id', 'patientid', 'pat_id', 'pt_id', 'subject_id', 'subjectid', 'pid'],
    "patient_mrn": ['mrn', 'medical record number', 'medical_record_number', 'patient_mrn', 'pat_mrn', 'patient mrn'],
    "encounter_id": ['encounter id', 'enc_id', 'enc_csn_id', 'pat_enc_csn_id', 'encounter_id', 'encounterid', 'visit id', 'visit_id'],
    "visit_id": ['visit id', 'visit_id', 'visitid', 'enc_visit_id', 'enc_visitid', 'encounter id', 'encounter_id'],
    "episode_id": ['episode id', 'episode_id', 'episodeid', 'ep_id', 'epid'],
    "date_of_birth": ['date of birth', 'date_of_birth', 'dob', 'birth date', 'birth_date', 'birthdate'],
    "sex": ['sex', 'gender', 'biological sex'],
    "race": ['race', 'ethnicity', 'racial group'],
    "address": ['address', 'patient address', 'home address', 'street address'],
    "zip_code": ['zip code', 'zip', 'postal code', 'zipcode'],
    "phone_number": ['phone number', 'phone', 'telephone', 'contact number', 'cell phone', 'mobile number'],
}

def detect_key(df: pd.DataFrame, key: str, candidates: list, threshold: float = 0.6):
    """Detects key column using fuzzy matching."""
    candidate_matches = []
    for col in df.columns:
        norm = col.lower().strip().replace(" ", "_")
        for cand in candidates:
            sim = difflib.SequenceMatcher(None, norm, cand).ratio()
            if sim >= threshold:
                candidate_matches.append((col, sim))
    if not candidate_matches:
        return None, []
    candidate_matches.sort(key=lambda x: x[1], reverse=True)
    best = candidate_matches[0]
    if len(candidate_matches) > 2 and candidate_matches[2][1] >= best[1] - 0.3:
        return None, [col for col, sim in candidate_matches]
    return best[0], [col for col, sim in candidate_matches]

# ----------------------------------------------------------------
class DataMergeWorker(QThread):
    """Worker thread that preprocesses each file, concatenates them, and deduplicates."""
    finished = pyqtSignal(pd.DataFrame, str)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, dataframes, file_paths):
        super().__init__()
        self.dataframes = dataframes
        self.file_paths = file_paths
        self._is_running = True  # Cancellation flag

    def cancel(self):
        self._is_running = False

    def run(self):
        try:
            if not self.dataframes:
                self.error.emit("No data loaded to merge.")
                return

            processed_dfs = []
            total = len(self.dataframes)
            # Preprocess each dataframe independently.
            for i, df in enumerate(self.dataframes):
                if not self._is_running:
                    self.error.emit("Merge cancelled by user.")
                    return

                # Determine all candidate keys present in this dataframe.
                keys = []
                for key_list in candidate_keys.values():
                    keys.extend([k for k in key_list if k in df.columns])
                keys = list(set(keys))
                if keys:
                    df = self.coalesce_dataframe(df, keys)
                processed_dfs.append(df)
                self.progress.emit(int((i + 1) / total * 30))

            # Concatenate all processed dataframes.
            combined = pd.concat(processed_dfs, ignore_index=True, sort=False)
            self.progress.emit(40)

            # Determine a global key for deduplication (using preference order).
            global_key = None
            for key in ["patient_id", "patient_mrn", "encounter_id", "visit_id", "episode_id"]:
                if all(key in df.columns for df in processed_dfs):
                    global_key = key
                    break

            # Use groupby to coalesce duplicate rows.
            if global_key:
                final_df = combined.groupby(global_key, dropna=False).apply(self.coalesce_group)
                # Reset index after groupby – drop the extra grouping index.
                final_df.reset_index(drop=True, inplace=True)
            else:
                final_df = combined.drop_duplicates()

            self.progress.emit(80)
            final_df = self.standardize_data_types(final_df)
            self.progress.emit(100)
            self.finished.emit(final_df, "")

        except Exception as e:
            self.error.emit(str(e))

    def coalesce_dataframe(self, df: pd.DataFrame, common_keys: list) -> pd.DataFrame:
        """Coalesces data within a single dataframe based on common keys."""
        if not common_keys:
            return df
        valid_keys = [key for key in common_keys if key in df.columns and df[key].notna().any()]
        if not valid_keys:
            return df
        grouped = df.groupby(valid_keys, dropna=False)
        coalesced_df = grouped.apply(self.coalesce_group)
        return coalesced_df.reset_index(drop=True)

    def coalesce_group(self, group: pd.DataFrame) -> pd.Series:
        """Coalesces data for one group by selecting the first non-null (or non-empty) value per column."""
        coalesced_row = {}
        for col in group.columns:
            non_null_values = group[col].dropna()
            if non_null_values.empty:
                coalesced_row[col] = pd.NA
            elif non_null_values.dtype == 'object':
                non_empty = non_null_values[non_null_values.str.strip() != ""]
                if not non_empty.empty:
                    coalesced_row[col] = non_empty.iloc[0]
                else:
                    coalesced_row[col] = non_null_values.iloc[0]
            else:
                coalesced_row[col] = non_null_values.iloc[0]
        return pd.Series(coalesced_row)

    def standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes certain data types (e.g., date and numeric columns)."""
        date_cols = ["date_of_birth", "encounter_date", "visit_date"]
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass
        numeric_cols = ["zip_code"]
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass
        return df

# ----------------------------------------------------------------
class DataFrameModel(QAbstractTableModel):
    """A Qt model to efficiently display a pandas DataFrame."""
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            if pd.isna(value):
                return ""
            if isinstance(value, pd.Timestamp):
                return value.strftime('%Y-%m-%d %H:%M:%S')
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(section)
        return None

# ----------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMR Data Alignment and Deduplication")
        self.setMinimumSize(800, 600)

        self.dataframes = []
        self.file_paths = []

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.stats_label = QLabel("Merged dataset stats will appear here.", self)
        self.stats_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.stats_label)

        self.file_list = QListWidget(self)
        main_layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload TSV Files", self)
        self.upload_button.clicked.connect(self.load_files)
        btn_layout.addWidget(self.upload_button)

        self.merge_button = QPushButton("Merge Data", self)
        self.merge_button.clicked.connect(self.start_merge)
        self.merge_button.setEnabled(False)
        btn_layout.addWidget(self.merge_button)
        main_layout.addLayout(btn_layout)

        self.tableView = QTableView(self)
        main_layout.addWidget(self.tableView)

        self.progress_dialog = QProgressDialog("Merging Data...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.hide()
        self.progress_dialog.canceled.connect(self.cancel_merge)

        self.merge_worker = None

    def load_files(self):
        options = QFileDialog.Option.ReadOnly | QFileDialog.Option.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select TSV Files",
            "",
            "TSV Files (*.tsv);;All Files (*)",
            options=options
        )
        if files:
            for file in files:
                self.file_paths.append(file)
                self.file_list.addItem(file)
                try:
                    df = pd.read_csv(file, sep="\t")
                    df_std = self.standardize_keys(df)
                    self.dataframes.append(df_std)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load file:\n{file}\n{str(e)}")
                    continue
            if self.dataframes:
                self.merge_button.setEnabled(True)

    def standardize_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        df_std = df.copy()
        for key, cand_list in candidate_keys.items():
            detected, candidates = detect_key(df_std, key, cand_list)
            if detected is None and candidates:
                chosen, ok = QInputDialog.getItem(
                    self,
                    f"Select column for {key}",
                    f"Multiple columns appear to match '{key}'.\nPlease choose one:",
                    candidates,
                    0,
                    False
                )
                if ok and chosen:
                    df_std.rename(columns={chosen: key}, inplace=True)
            elif detected is not None:
                df_std.rename(columns={detected: key}, inplace=True)
        return df_std

    def start_merge(self):
        if self.merge_worker is not None:
            QMessageBox.warning(self, "Warning", "Merge already in progress.")
            return

        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        self.merge_worker = DataMergeWorker(self.dataframes, self.file_paths)
        self.merge_worker.progress.connect(self.progress_dialog.setValue)
        self.merge_worker.finished.connect(self.merge_finished)
        self.merge_worker.error.connect(self.merge_error)
        self.merge_worker.start()

    def cancel_merge(self):
        if self.merge_worker:
            self.merge_worker.cancel()
            self.merge_worker.wait()  # Wait for graceful termination
            self.merge_worker = None
            self.progress_dialog.hide()
            QMessageBox.information(self, "Cancelled", "Data merging cancelled.")

    def merge_finished(self, master_df, error_message):
        self.merge_worker = None
        self.progress_dialog.hide()
        if error_message:
            QMessageBox.critical(self, "Error", error_message)
            return
        self.master_df = master_df
        self.update_stats(master_df)
        self.update_table(master_df)

    def merge_error(self, error_message):
        self.merge_worker = None
        self.progress_dialog.hide()
        QMessageBox.critical(self, "Error", f"Merge error: {error_message}")

    def update_stats(self, df: pd.DataFrame):
        stats_parts = []
        if "patient_id" in df.columns:
            stats_parts.append(f"Total Patients (by ID): {df['patient_id'].nunique()}")
        if "patient_mrn" in df.columns:
            stats_parts.append(f"Total Patients (by MRN): {df['patient_mrn'].nunique()}")
        if "encounter_id" in df.columns:
            stats_parts.append(f"Total Encounters: {df['encounter_id'].nunique()}")
        if "visit_id" in df.columns:
            stats_parts.append(f"Total Visits: {df['visit_id'].nunique()}")
        if "episode_id" in df.columns:
            stats_parts.append(f"Total Episodes: {df['episode_id'].nunique()}")
        stats_text = "   ".join(stats_parts) if stats_parts else "No key stats identified."
        self.stats_label.setText(stats_text)

    def update_table(self, df: pd.DataFrame):
        model = DataFrameModel(df)
        self.tableView.setModel(model)
        self.tableView.resizeColumnsToContents()

# ----------------------------------------------------------------
# def main():
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())

# if __name__ == "__main__":
#     main()
