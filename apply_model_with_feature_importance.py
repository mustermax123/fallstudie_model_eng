import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


def prepare_data(df):
    """
    Bereitet die Daten für die Modellierung vor. Dazu gehört:
    - Trennen der Zielvariable ('success') von den Features.
    - Aufteilen in Trainings- und Testdaten.
    - One-Hot-Encoding für kategoriale Merkmale.
    
    Parameters:
        df (pd.DataFrame): Der DataFrame mit den Rohdaten.

    Returns:
        X_train_encoded (pd.DataFrame): Encodierte Trainingsdaten.
        X_test_encoded (pd.DataFrame): Encodierte Testdaten.
        y_train (pd.Series): Zielvariable für die Trainingsdaten.
        y_test (pd.Series): Zielvariable für die Testdaten.
        encoder (OneHotEncoder): Der trainierte OneHotEncoder für zukünftige Transformationen.
        X_test (pd.DataFrame): Originaler Test-DataFrame (nicht encodiert).
    """

    # Trennung der Features und der Zielvariable
    X = df.drop(columns=['success'])  # Features (alle außer 'success')
    y = df['success']  # Zielvariable (0 oder 1: Misserfolg oder Erfolg)

    # Aufteilen in Trainings- (80%) und Testdaten (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Stratifikation sorgt für gleiche Klassenverteilung
    )

    # One-Hot-Encoding für kategoriale Merkmale
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Transformation der Trainings- und Testdaten
    X_train_encoded = pd.DataFrame(
        encoder.fit_transform(X_train.select_dtypes(include=['object', 'category'])),  # Nur kategoriale Features encodieren
        columns=encoder.get_feature_names_out(X_train.select_dtypes(include=['object', 'category']).columns)
    )
    X_test_encoded = pd.DataFrame(
        encoder.transform(X_test.select_dtypes(include=['object', 'category'])),  # Testdaten mit gleichem Encoder transformieren
        columns=encoder.get_feature_names_out(X_test.select_dtypes(include=['object', 'category']).columns)
    )

    # Numerische Features beibehalten und mit den encodierten Features kombinieren
    X_train_encoded = pd.concat([X_train_encoded.reset_index(drop=True), X_train.select_dtypes(include=['number']).reset_index(drop=True)], axis=1)
    X_test_encoded = pd.concat([X_test_encoded.reset_index(drop=True), X_test.select_dtypes(include=['number']).reset_index(drop=True)], axis=1)

    return X_train_encoded, X_test_encoded, y_train, y_test, encoder, X_test


def generate_psp_variants(row, psps):
    """
    Erstellt vier Varianten einer Transaktion, wobei jeweils ein anderer PSP (Payment Service Provider) verwendet wird.

    Parameters:
        row (pd.Series): Eine einzelne Zeile aus X_test.
        psps (list): Liste der möglichen Zahlungsdienstleister (PSPs).

    Returns:
        pd.DataFrame: Ein DataFrame mit vier Varianten der ursprünglichen Zeile, jede mit einem anderen PSP.
    """

    rows = []
    for psp in psps:
        row_copy = row.copy()  # Erstelle eine Kopie der ursprünglichen Zeile
        row_copy['PSP'] = psp  # Setze einen anderen PSP für diese Variante
        rows.append(row_copy)

    # Die Liste der Zeilen wird in einen DataFrame umgewandelt
    return pd.DataFrame(rows)


def encode_psp_variants(psp_variants, encoder, feature_columns):
    """
    Encodiert die PSP-Varianten einer Transaktion, sodass sie als Eingabe für ein Modell verwendet werden können.

    Parameters:
        psp_variants (pd.DataFrame): DataFrame mit vier Zeilen, jede für einen PSP.
        encoder (OneHotEncoder): Der trainierte Encoder für kategoriale Merkmale.
        feature_columns (list): Liste aller Feature-Spalten aus dem Trainingsset.

    Returns:
        pd.DataFrame: Ein DataFrame mit den encodierten Varianten, bereit für die Modellvorhersage.
    """

    # Ermittlung der kategorialen und numerischen Features
    categorical_features = encoder.feature_names_in_  # Kategorische Features aus dem Encoder extrahieren
    numerical_features = [col for col in feature_columns if col not in encoder.get_feature_names_out(categorical_features)]

    # Anwendung des Encoders auf die kategorialen Features
    psp_variants_categorical = pd.DataFrame(
        encoder.transform(psp_variants[categorical_features]),  # Transformation der PSP-Varianten
        columns=encoder.get_feature_names_out(categorical_features)  # Spaltennamen beibehalten
    )

    # Numerische Features aus psp_variants übernehmen
    psp_variants_numerical = psp_variants[numerical_features].reset_index(drop=True)

    # Sicherstellen, dass alle numerischen Features vorhanden sind
    for col in numerical_features:
        if col not in psp_variants_numerical.columns:
            psp_variants_numerical[col] = 0  # Fehlende Spalten mit 0 füllen

    # Kombination von encodierten kategorialen und numerischen Features
    psp_variants_encoded = pd.concat([psp_variants_categorical, psp_variants_numerical], axis=1)

    # Spalten in der gleichen Reihenfolge wie im Trainingsset anordnen
    psp_variants_encoded = psp_variants_encoded.reindex(columns=feature_columns, fill_value=0)

    return psp_variants_encoded

# 2. Modelltraining und Evaluation
def train_dt(X_train, y_train):
    """
    Trainiert ein Decision Tree Modell, um die Erfolgswahrscheinlichkeit vorherzusagen.

    Parameters:
        X_train (pd.DataFrame): Trainingsdaten mit Features.
        y_train (pd.Series): Zielvariable für das Training.

    Returns:
        dt_model (DecisionTreeClassifier): Trainiertes Decision Tree Modell.
    """

    dt_model = DecisionTreeClassifier(
        max_depth=5,               # Maximale Tiefe des Baumes, um Overfitting zu reduzieren.
        min_samples_split=5,        # Minimale Anzahl an Samples, um einen Knoten zu teilen.
        min_samples_leaf=3,         # Minimale Anzahl an Samples pro Blattknoten.
        class_weight='balanced',    # Gewichtung für das Klassenungleichgewicht.
        random_state=42             # Setzt eine feste Zufallszahl für Reproduzierbarkeit.
    )
        
    # Modell mit den Trainingsdaten trainieren
    dt_model.fit(X_train, y_train)
    
    return dt_model


def select_psp_with_dataset(success_probabilities, psp_variants_encoded, cost_dict, threshold=0.8):
    """
    Wählt den besten Zahlungsdienstleister basierend auf Erfolgswahrscheinlichkeiten und Kosten.

    Parameters:
        success_probabilities (list): Erfolgswahrscheinlichkeiten für die PSPs.
        psp_variants_encoded (pd.DataFrame): Encodierter DataFrame mit den 4 PSP-Varianten.
        cost_dict (dict): Dictionary mit den Kosten pro PSP.
        threshold (float): Mindestschwelle für die Auswahl eines PSPs.

    Returns:
        str: Der ausgewählte PSP.
    """

    # PSPs aus dem DataFrame extrahieren
    psps = psp_variants_encoded['PSP'].values

    # Ermittlung des PSPs mit der höchsten Erfolgswahrscheinlichkeit
    max_proba = max(success_probabilities)
    best_psps = [psp for psp, proba in zip(psps, success_probabilities) if proba == max_proba]

    # Falls eine PSP-Wahrscheinlichkeit die Schwelle überschreitet, wird der günstigste PSP gewählt
    if max_proba > threshold:
        candidates = [psp for psp, proba in zip(psps, success_probabilities) if proba > threshold]
        best_psp = min(candidates, key=lambda psp: cost_dict[psp])
    else:
        # Falls keine PSP die Schwelle überschreitet, wird der mit der höchsten Wahrscheinlichkeit gewählt
        best_psp = best_psps[0]

    return best_psp


def predict_psp_probabilities(model, X_test, psps):
    """
    Sagt für jede Instanz in X_test die Erfolgswahrscheinlichkeiten für verschiedene Zahlungsdienstleister voraus.

    Parameters:
        model: Das trainierte DecisionTreeClassifier Modell.
        X_test (pd.DataFrame): Test-Datensatz mit Eingabewerten.
        psps (list): Liste der verfügbaren Zahlungsdienstleister.

    Returns:
        pd.DataFrame: DataFrame mit Erfolgswahrscheinlichkeiten und den zugehörigen Transaktionsdetails.
    """

    results = []
    for index, row in X_test.iterrows():
        # Generiere vier Varianten der Transaktion mit verschiedenen PSPs
        psp_variants = generate_psp_variants(row, psps)

        # Encodiere die PSP-Varianten für die Modellvorhersage
        psp_variants_encoded = encode_psp_variants(psp_variants, encoder, X_train_encoded.columns)

        # Erfolgswahrscheinlichkeiten für jede PSP ermitteln
        probabilities = model.predict_proba(psp_variants_encoded)[:, 1]

        # Speichere die Ergebnisse
        for psp, prob in zip(psps, probabilities):
            results.append({
                'Index': index,
                'amount': row['amount'],
                '3D_secured': row['3D_secured'],
                'PSP': psp,
                'Success Probability': prob
            })
    
    return pd.DataFrame(results)


def execute_predict_psp(row, encoder, X_train_encoded, dt_model, psps):
    """
    Führt eine Vorhersage für eine einzelne Transaktion durch, indem verschiedene PSPs simuliert werden.

    Parameters:
        row (pd.Series): Einzelne Transaktion aus X_test.
        encoder (OneHotEncoder): Der trainierte OneHotEncoder für das Encoding.
        X_train_encoded (pd.DataFrame): Der Trainingsdatensatz für die Spaltenstruktur.
        dt_model: Das trainierte Decision Tree Modell.
        psps (list): Liste der verfügbaren PSPs.

    Returns:
        str: Der ausgewählte beste PSP.
    """

    # Schritt 1: Varianten der Transaktion mit verschiedenen PSPs generieren
    psp_variants = generate_psp_variants(row, psps)
    
    # Schritt 2: Encodieren der PSP-Varianten für das Modell
    psp_variants_encoded = encode_psp_variants(psp_variants, encoder, X_train_encoded.columns)
    
    # Schritt 3: Erfolgswahrscheinlichkeiten mit dem Modell vorhersagen
    y_pred_proba = dt_model.predict_proba(psp_variants_encoded)
    success_probabilities = y_pred_proba[:, 1] 

    # Schritt 4: Auswahl des besten PSP basierend auf Erfolgswahrscheinlichkeiten und Kosten
    cost_dict = {
        'Moneycard': 5,
        'Goldcard': 10,
        'UK_Card': 3,
        'Simplecard': 1
    }
    
    threshold = 0.8  # Mindestschwelle für die Erfolgswahrscheinlichkeit
    best_psp = select_psp_with_dataset(success_probabilities, psp_variants, cost_dict, threshold)

    return best_psp


def plot_feature_importance(dt_model, feature_names):
    """
    Plottet die Wichtigkeit der Features im trainierten Decision Tree Modell.

    Parameters:
        dt_model: Das trainierte DecisionTreeClassifier Modell.
        feature_names (list): Spaltennamen des Trainingsdatensatzes.
    """

    # Extrahiere die Feature-Wichtigkeiten aus dem Modell
    importances = dt_model.feature_importances_

    # Sortiere die Features basierend auf ihrer Wichtigkeit
    indices = np.argsort(importances)[::-1]
    
    # Erstelle eine geordnete Liste der Feature-Namen und ihrer Wichtigkeiten
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Erzeuge das Diagramm zur Visualisierung der Feature-Wichtigkeiten
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color='blue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance im Decision Tree")
    plt.gca().invert_yaxis()  # Größte Werte oben anzeigen
    plt.show()


def plot_psp_performance(results_df):
    """
    Plottet die Erfolgswahrscheinlichkeiten der PSPs in Bezug auf den Transaktionsbetrag und den 3D-Sicherheitsstatus.

    Parameters:
        results_df (pd.DataFrame): DataFrame mit PSP-Vorhersagen und zugehörigen Instanzen.
    """

    # Scatter-Plot für Erfolgswahrscheinlichkeit vs. Transaktionsbetrag
    plt.figure(figsize=(10, 6))
    for psp in results_df['PSP'].unique():
        subset = results_df[results_df['PSP'] == psp]
        plt.scatter(subset['amount'], subset['Success Probability'], label=psp, alpha=0.6)
    plt.xlabel("Transaction Amount")
    plt.ylabel("Success Probability")
    plt.title("Success Probability over Transaction Amount")
    plt.legend()
    plt.show()

    # Scatter-Plot für Erfolgswahrscheinlichkeit vs. 3D-Sicherheitsstatus
    plt.figure(figsize=(10, 6))
    for psp in results_df['PSP'].unique():
        subset = results_df[results_df['PSP'] == psp]
        plt.scatter(subset['3D_secured'], subset['Success Probability'], label=psp, alpha=0.6)
    plt.xlabel("3D Secured")
    plt.ylabel("Success Probability")
    plt.title("Success Probability over 3D Secured Status")
    plt.legend()
    plt.show()

# Hauptprogramm: Ausführung der gesamten Pipeline
if __name__ == "__main__":

    # Datei mit Transaktionsdaten laden
    file = r"C:\Users\czind\Desktop\fernstudium\Fallstudie\use_case_1\use_case_1\PSP_Jan_Feb_2019.xlsx"
    df = pd.read_excel(file)

    # Speichern der Spaltennamen als Liste für spätere Analysen
    parameter_list = list(df)

    # Entfernen der unnötigen Spalte "Unnamed: 0", falls vorhanden
    df = df.drop("Unnamed: 0", axis=1)
    
    # Definierte Liste der verfügbaren Zahlungsdienstleister (PSPs)
    psps = ['Moneycard', 'Goldcard', 'UK_Card', 'Simplecard']
    
    # Vorbereitung der Daten: Encoding, Train-Test-Split und Modelltraining
    X_train_encoded, X_test_encoded, y_train, y_test, encoder, X_test = prepare_data(df)

    # Decision Tree Modell trainieren
    dt_model = train_dt(X_train_encoded, y_train)
    
    # Beispielhafte Simulation einer neuen Transaktion basierend auf einer Stichprobe aus X_test
    row = X_test.iloc[10]  # Auswahl der 11. Zeile aus X_test
    
    # Vorhersage des besten Zahlungsdienstleisters für diese Transaktion
    best_psp = execute_predict_psp(row, encoder, X_train_encoded, dt_model, psps)
    print(f"Der ausgewählte PSP ist: {best_psp}")
    
    # Visualisierung der Feature-Wichtigkeit des trainierten Modells
    plot_feature_importance(dt_model, X_train_encoded.columns)
    
    # Berechnung der Erfolgswahrscheinlichkeiten für alle PSPs über den gesamten Testdatensatz
    psp_results_df = predict_psp_probabilities(dt_model, X_test, psps)
    
    # Analyse der PSP-Performance durch Visualisierung der Erfolgswahrscheinlichkeiten
    plot_psp_performance(psp_results_df)
