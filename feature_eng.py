import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def add_attempt_count_feature(df, time_window_minutes=2):
    """
    Fügt dem DataFrame ein neues Merkmal 'attempt_count' hinzu, das die Anzahl der Versuche
    für Transaktionen innerhalb eines bestimmten Zeitfensters ermittelt.

    Parameters:
        df (pd.DataFrame): Der ursprüngliche DataFrame mit Transaktionsdaten.
        time_window_minutes (int): Zeitfenster in Minuten, um Transaktionen als denselben Versuch zu betrachten.

    Returns:
        pd.DataFrame: DataFrame mit der zusätzlichen Spalte 'attempt_count'.
    """

    # Sicherstellen, dass die Zeitstempel korrekt als datetime interpretiert werden
    df['tmsp'] = pd.to_datetime(df['tmsp'])
    
    # Sortierung des DataFrames nach den relevanten Merkmalen
    df = df.sort_values(by=['country', 'amount', '3D_secured', 'card', 'PSP', 'tmsp'])
    
    # Gruppieren der Daten nach bestimmten Merkmalen, um Transaktionen mit ähnlichen Eigenschaften zusammenzufassen
    grouped = df.groupby(['country', 'amount', '3D_secured', 'card', 'PSP'])
    
    # Initialisierung der neuen Spalte für die Anzahl der Versuche mit Standardwert 1
    df['attempt_count'] = 1  
    
    # Iteration durch jede Gruppe, um die Anzahl der Versuche pro Transaktionsgruppe zu ermitteln
    for group_name, group in grouped:
        group = group.sort_values(by='tmsp')  # Innerhalb der Gruppe nach Zeitstempel sortieren
        attempt_count = 1  # Zähler für die Anzahl der Versuche
        
        # Iteration durch die einzelnen Zeilen der Gruppe
        for i in range(1, len(group)):
            # Zeitdifferenz zur vorherigen Transaktion innerhalb der Gruppe berechnen
            time_diff = (group.iloc[i]['tmsp'] - group.iloc[i - 1]['tmsp']).total_seconds()
            
            # Wenn die Transaktion innerhalb des definierten Zeitfensters liegt, wird der Versuchszähler erhöht
            if time_diff <= time_window_minutes * 60:
                attempt_count += 1
            else:
                attempt_count = 1  # Zähler zurücksetzen, wenn das Zeitfenster überschritten wird
            
            # Aktualisierung der Spalte 'attempt_count' mit der berechneten Anzahl der Versuche
            df.loc[group.index[i], 'attempt_count'] = attempt_count
    
    return df


def plot_success_rate_and_attempts(df):
    """
    Erstellt drei Plots:
    1. Erfolgsquote pro Versuch (attempt_count).
    2. Häufigkeit der Versuche (attempt_count).
    3. Häufigkeit der Versuche mit logarithmischer Skalierung.

    Parameters:
        df (pd.DataFrame): Der DataFrame mit den Spalten 'attempt_count' und 'success'.
    """

    # Berechnung der Erfolgsquote pro Versuch
    success_rate = df.groupby('attempt_count')['success'].mean()
    
    # Berechnung der Anzahl der Versuche
    attempt_counts = df['attempt_count'].value_counts().sort_index()

    # Erstellen des Plots für die Erfolgsquote pro Versuch
    plt.figure(figsize=(8, 4))
    success_rate.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Erfolgsquote pro Versuch (attempt_count)", fontsize=14)
    plt.xlabel("Versuch (attempt_count)", fontsize=12)
    plt.ylabel("Erfolgsquote", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Erstellen des Plots für die Häufigkeit der Versuche
    plt.figure(figsize=(8, 4))
    attempt_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title("Häufigkeit der Versuche (attempt_count)", fontsize=14)
    plt.xlabel("Versuch (attempt_count)", fontsize=12)
    plt.ylabel("Häufigkeit", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Berechnung der Häufigkeit der Versuche für den logarithmischen Plot
    attempt_counts = df['attempt_count'].value_counts().sort_index()

    # Erstellen des Plots mit logarithmischer Skalierung
    plt.figure(figsize=(8, 4))
    attempt_counts.plot(kind='bar', color='lightgreen', edgecolor='black', log=True)  # Log-Skalierung aktivieren
    plt.title("Häufigkeit der Versuche (Log-Skala)", fontsize=14)
    plt.xlabel("Versuch (attempt_count)", fontsize=12)
    plt.ylabel("Häufigkeit (logarithmisch)", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Hauptprogramm
if __name__ == "__main__":

    # Definierter Dateipfad für die Excel-Datei mit den Transaktionsdaten
    path_to_data = r"Fallstudie\use_case_1\use_case_1\PSP_Jan_Feb_2019.xlsx"
    
    # Laden der Excel-Datei in einen Pandas DataFrame
    df = pd.read_excel(path_to_data)
    
    # Speichern der Spaltennamen als Liste
    parameter_list = list(df)
    
    # Entfernen der überflüssigen Spalte "Unnamed: 0"
    df = df.drop("Unnamed: 0", axis=1)
    
    # Hinzufügen der Spalte 'attempt_count' zum DataFrame
    df_with_attempts = add_attempt_count_feature(df)

    # Ermitteln der maximalen Anzahl an Versuchen innerhalb des Datensatzes
    max_attempt = df_with_attempts["attempt_count"].max()

    # Visualisierung der Erfolgsquote und der Verteilung der Versuche
    plot_success_rate_and_attempts(df_with_attempts)
