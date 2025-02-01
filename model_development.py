import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# 1. Datenvorbereitung
def prepare_data(df):
    """
    Bereitet die Daten für das Training von Machine-Learning-Modellen vor.
    Dazu gehört das One-Hot-Encoding kategorialer Merkmale sowie die Aufteilung in Trainings- und Testdaten.

    Parameters:
        df (pd.DataFrame): Der ursprüngliche DataFrame mit den Rohdaten.

    Returns:
        X_train_encoded (pd.DataFrame): Features für das Training.
        X_test_encoded (pd.DataFrame): Features für den Test.
        y_train (pd.Series): Zielvariable für das Training.
        y_test (pd.Series): Zielvariable für den Test.
    """

    # Zielvariable (success) von den Features trennen
    X = df.drop(columns=['success'])  # Features ohne die Zielvariable
    y = df['success']  # Zielvariable (0 oder 1 für Misserfolg oder Erfolg)

    # Aufteilen der Daten in Trainings- und Testsets (80% Training, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Stratify zur Beibehaltung der Klassenverteilung
    )

    # One-Hot-Encoding für kategoriale Merkmale (z.B. 'country', 'PSP', 'card' etc.)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Transformation der Trainings- und Testdaten
    X_train_encoded = pd.DataFrame(
        encoder.fit_transform(X_train.select_dtypes(include=['object', 'category'])),
        columns=encoder.get_feature_names_out(X_train.select_dtypes(include=['object', 'category']).columns)
    )
    X_test_encoded = pd.DataFrame(
        encoder.transform(X_test.select_dtypes(include=['object', 'category'])),
        columns=encoder.get_feature_names_out(X_test.select_dtypes(include=['object', 'category']).columns)
    )

    # Numerische Merkmale beibehalten und mit den encodierten Daten kombinieren
    X_train_encoded = pd.concat([X_train_encoded.reset_index(drop=True), X_train.select_dtypes(include=['number']).reset_index(drop=True)], axis=1)
    X_test_encoded = pd.concat([X_test_encoded.reset_index(drop=True), X_test.select_dtypes(include=['number']).reset_index(drop=True)], axis=1)

    return X_train_encoded, X_test_encoded, y_train, y_test

# 2. Modelltraining und Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trainiert verschiedene Modelle (Decision Tree, Random Forest, XGBoost) 
    und bewertet deren Leistung auf den Testdaten.

    Parameters:
        X_train (pd.DataFrame): Trainingsfeatures.
        X_test (pd.DataFrame): Testfeatures.
        y_train (pd.Series): Trainingszielvariable.
        y_test (pd.Series): Testzielvariable.

    Returns:
        trained_models (dict): Ein Dictionary mit trainierten Modellen.
    """

    # Definition der Modelle mit optimierten Hyperparametern
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,               # Maximale Tiefe des Baums
            min_samples_split=5,        # Mindestens 5 Samples für eine Teilung
            min_samples_leaf=3,         # Mindestens 3 Samples pro Blattknoten
            class_weight='balanced',    # Berücksichtigung des Klassenungleichgewichts
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,           # Anzahl der Bäume im Wald
            max_depth=15,               # Maximale Tiefe der Bäume
            min_samples_split=5,        # Mindestens 5 Samples für eine Teilung
            min_samples_leaf=3,         # Mindestens 3 Samples pro Blattknoten
            class_weight='balanced',    # Berücksichtigung des Klassenungleichgewichts
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,           # Anzahl der Bäume
            max_depth=10,               # Maximale Tiefe der Bäume
            learning_rate=0.1,          # Lernrate des Modells
            subsample=0.8,              # Anteil der Stichprobe pro Baum
            colsample_bytree=0.8,       # Anteil der Features pro Baum
            scale_pos_weight=10,        # Gewichtung der positiven Klasse zur Balance
            eval_metric='logloss',      # Evaluierungsmetrik zur Minimierung des Log-Loss
            use_label_encoder=False,    # Deaktivierung des veralteten Label-Encoders
            random_state=42
        )
    }

    trained_models = {}

    # Iteration über die Modelle zum Training und zur Evaluation
    for name, model in models.items():
        # Modell trainieren
        model.fit(X_train, y_train)

        # Vorhersagen für die Testdaten
        y_pred = model.predict(X_test)

        # Modellbewertung anhand der Accuracy und des Klassifikationsberichts
        print(f"=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("\n")

        trained_models[name] = model  # Modell speichern
    
    return trained_models

# 3. ROC-Kurve und AUC anzeigen
def plot_roc_auc(models, X_test, y_test):
    """
    Erstellt ROC-Kurven für alle trainierten Modelle, um deren Diskriminierungskraft zu bewerten.

    Parameters:
        models (dict): Dictionary mit den trainierten Modellen.
        X_test (pd.DataFrame): Testfeatures.
        y_test (pd.Series): Wahre Klassenlabels der Testdaten.
    """
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        # Berechnung der Wahrscheinlichkeiten für die positive Klasse (Erfolg)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Berechnung der False-Positive-Rate (FPR) und True-Positive-Rate (TPR)
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        
        # Berechnung des AUC-Wertes (Fläche unter der ROC-Kurve)
        auc_score = auc(fpr, tpr)
        
        # Darstellung der ROC-Kurve für das Modell
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')

    # Diagonale Linie für den Zufallsfall
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Kurve für die Modelle')
    plt.legend(loc='lower right')
    plt.show()

# Hauptprogramm
if __name__ == "__main__":

    # Definierter Dateipfad für die Excel-Datei mit den Transaktionsdaten
    path_to_data = r"Fallstudie\use_case_1\use_case_1\PSP_Jan_Feb_2019.xlsx"
    
    # Laden der Excel-Datei in einen Pandas DataFrame
    df = pd.read_excel(path_to_data)

    # Speichern der ursprünglichen Spaltennamen
    parameter_list = list(df)

    # Entfernen unnötiger Spalte "Unnamed: 0"
    df = df.drop("Unnamed: 0", axis=1)

    # Daten vorbereiten (Encoding, Train-Test-Split)
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Training und Evaluation der Modelle
    trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # ROC-Kurven für die trainierten Modelle visualisieren
    plot_roc_auc(trained_models, X_test, y_test)
