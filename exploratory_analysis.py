import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_value_counts(df, columns):
    """
    Erstellt Barplots für die Häufigkeiten der Werte in den angegebenen Spalten.
    
    Parameters:
        df (DataFrame): Der DataFrame mit den Daten.
        columns (list): Liste der Spalten, die geplottet werden sollen.
    """
    for column in columns:
        plt.figure(figsize=(8, 4))  # Größe des Plots festlegen
        
        # Balkendiagramm der Werteanzahl erstellen
        sns.countplot(data=df, x=column, order=df[column].value_counts().index, palette="viridis")
        
        # Achsen und Titel anpassen
        plt.title(f"Häufigkeit der Werte in '{column}'", fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Anzahl", fontsize=12)
        plt.xticks(rotation=45)  # X-Achsenbeschriftung um 45 Grad drehen
        plt.tight_layout()  # Layout optimieren
        plt.show()


def plot_value_counts_with_percentages(df, columns):
    """
    Erstellt Barplots für die Häufigkeiten der Werte in den angegebenen Spalten
    und fügt Prozentzahlen über die Balken.
    
    Parameters:
        df (DataFrame): Der DataFrame mit den Daten.
        columns (list): Liste der Spalten, die geplottet werden sollen.
    """
    for column in columns:
        plt.figure(figsize=(8, 4))  # Größe des Plots festlegen
        
        # Häufigkeiten und Prozentwerte berechnen
        value_counts = df[column].value_counts()
        total = len(df)  # Gesamtanzahl der Werte
        percentages = (value_counts / total * 100).round(2)  # Prozentzahlen runden
        
        # Barplot erstellen
        sns.barplot(
            x=value_counts.index,  # X-Achse: Kategorien
            y=value_counts.values,  # Y-Achse: Anzahl der Werte
            order=value_counts.index,  # Reihenfolge basierend auf der Häufigkeit
            palette="viridis"
        )
        
        # Prozentzahlen über die Balken schreiben
        for i, value in enumerate(value_counts.values):
            plt.text(i, value + 100, f"{percentages.iloc[i]}%", ha='center', fontsize=10)
        
        # Achsen und Titel anpassen
        plt.title(f"Häufigkeit der Werte in '{column}'", fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Anzahl", fontsize=12)
        plt.xticks(rotation=45)  # X-Achsenbeschriftung um 45 Grad drehen
        plt.tight_layout()  # Layout optimieren
        plt.show()


def plot_boxplots_with_mean(df):
    """
    Erstellt einen vertikalen Boxplot für die Spalte 'amount' mit Matplotlib
    und zeichnet den Mittelwert ein.
    
    Parameters:
        df (DataFrame): Der DataFrame mit den Daten.
    """
    # Daten für den Boxplot, fehlende Werte entfernen
    data = df['amount'].dropna()
    
    plt.figure(figsize=(6, 8))  # Größe des Plots festlegen
    
    # Boxplot mit Mittelwert erstellen
    box = plt.boxplot(data, vert=True, patch_artist=True, showmeans=True, 
                      flierprops=dict(marker='o', color='red', markersize=5))
    
    # Achsentitel setzen
    plt.title("Boxplot für 'amount' mit Mittelwert", fontsize=14)
    plt.ylabel("Betrag (amount)", fontsize=12)
    plt.xticks([1], ['amount'])  # X-Achse beschriften
    
    # Farbe der Box anpassen
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')  # Box-Farbe setzen
    
    # Gitterlinien auf der y-Achse hinzufügen
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Layout optimieren
    plt.show()


# 1. Überblick über die Datenstruktur
def data_overview(df):
    """
    Gibt einen Überblick über die Struktur und den Inhalt des DataFrames aus.
    
    Parameters:
        df (DataFrame): Der DataFrame mit den Daten.
    """
    print("Datenüberblick:")
    print(df.info())  # Anzeige der Spalten, Datentypen und Nicht-Null-Werte
    
    print("\nStatistische Zusammenfassung:")
    print(df.describe(include='all'))  # Statistische Übersicht aller Spalten


# 2. Fehlende Werte prüfen
def missing_values_analysis(df):
    """
    Prüft und visualisiert fehlende Werte im DataFrame.
    
    Parameters:
        df (DataFrame): Der DataFrame mit den Daten.
    """
    missing_values = df.isnull().sum()  # Anzahl fehlender Werte pro Spalte berechnen
    
    print("\nFehlende Werte pro Spalte:")
    print(missing_values)  # Ergebnisse ausgeben
    
    # Balkendiagramm der fehlenden Werte erstellen
    plt.figure(figsize=(8, 4))
    missing_values.plot(kind='bar', color='skyblue')
    plt.title("Fehlende Werte pro Spalte")
    plt.ylabel("Anzahl fehlender Werte")
    plt.xticks(rotation=45)  # X-Achsenbeschriftung um 45 Grad drehen
    plt.show()


# 3. Datenqualität und Verteilung prüfen
def analyze_distributions(df):
    """
    Analysiert die Erfolgsquoten basierend auf verschiedenen Merkmalen
    und visualisiert die Verteilung der Beträge.

    Parameters:
        df (DataFrame): Der DataFrame mit den zu analysierenden Daten.
    """

    # Erfolgsquote pro PSP (Payment Service Provider) berechnen und ausgeben
    success_rate = df.groupby('PSP')['success'].mean()
    print("\nErfolgsquote pro PSP:")
    print(success_rate)

    # Erfolgsquote pro PSP als Balkendiagramm visualisieren
    plt.figure(figsize=(8, 4))
    success_rate.sort_values().plot(kind='bar', color='lightgreen')
    plt.title("Durchschnittliche Erfolgsquote pro PSP")
    plt.ylabel("Erfolgsquote")
    plt.xticks(rotation=45)  # X-Achsenbeschriftung um 45 Grad drehen
    plt.tight_layout()
    plt.show()

    # Erfolgsquote pro Land berechnen und ausgeben
    success_rate_country = df.groupby('country')['success'].mean()
    print("\nErfolgsquote pro Country:")
    print(success_rate_country)

    # Erfolgsquote pro Land als Balkendiagramm visualisieren
    plt.figure(figsize=(8, 4))
    success_rate_country.sort_values().plot(kind='bar', color='lightblue')
    plt.title("Durchschnittliche Erfolgsquote pro Country")
    plt.ylabel("Erfolgsquote")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Erfolgsquote pro Kartentyp berechnen und ausgeben
    success_rate_card = df.groupby('card')['success'].mean()
    print("\nErfolgsquote pro Card:")
    print(success_rate_card)

    # Erfolgsquote pro Kartentyp als Balkendiagramm visualisieren
    plt.figure(figsize=(8, 4))
    success_rate_card.sort_values().plot(kind='bar', color='orange')
    plt.title("Durchschnittliche Erfolgsquote pro Card")
    plt.ylabel("Erfolgsquote")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Erfolgsquote basierend auf 3D-Sicherheitsstatus berechnen und ausgeben
    success_rate_3d_secured = df.groupby('3D_secured')['success'].mean()
    print("\nErfolgsquote für 3D_secured:")
    print(success_rate_3d_secured)

    # Erfolgsquote für 3D-Sicherheitsstatus als Balkendiagramm visualisieren
    plt.figure(figsize=(8, 4))
    success_rate_3d_secured.sort_values().plot(kind='bar', color='purple')
    plt.title("Durchschnittliche Erfolgsquote für 3D-Sicherheitsstatus")
    plt.ylabel("Erfolgsquote")
    plt.xticks(ticks=[0, 1], labels=['Nicht gesichert', '3D-gesichert'], rotation=0)
    plt.tight_layout()
    plt.show()

    # Verteilung der Beträge analysieren
    plt.figure(figsize=(8, 4))
    sns.histplot(df['amount'], bins=30, kde=True, color='orange')
    plt.title("Verteilung des Betrags")
    plt.xlabel("Betrag")
    plt.ylabel("Häufigkeit")
    plt.show()


def analyze_relationships(df):
    """
    Analysiert die Korrelationen ausschließlich zwischen numerischen Spalten.

    Parameters:
        df (DataFrame): Der DataFrame mit den zu analysierenden Daten.
    """

    # Auswahl nur numerischer Spalten
    numeric_df = df.select_dtypes(include=['number'])

    # Berechnung der Korrelationen zwischen numerischen Variablen
    correlation_matrix = numeric_df.corr()
    print("\nKorrelationsmatrix der numerischen Variablen:")
    print(correlation_matrix)

    # Visualisierung der Korrelationsmatrix als Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Korrelationsmatrix der numerischen Variablen")
    plt.show()

# Funktionen zur Extraktion spezifischer Werte aus dem DataFrame
def get_min_max_amount(df):
    """
    Ermittelt den minimalen und maximalen Betrag aus der 'amount'-Spalte.

    Parameters:
        df (DataFrame): Der DataFrame mit den zu analysierenden Daten.

    Returns:
        tuple: Minimum- und Maximum-Wert des Betrags.
    """
    min_amount = df['amount'].min()
    max_amount = df['amount'].max()
    return min_amount, max_amount

def get_success_total(df):
    """
    Teilt den DataFrame in erfolgreiche und fehlgeschlagene Transaktionen auf.

    Parameters:
        df (DataFrame): Der DataFrame mit den zu analysierenden Daten.

    Returns:
        tuple: DataFrames für erfolgreiche und fehlgeschlagene Transaktionen.
    """
    success_df = df[df["success"] == 1]
    fail_df = df[df["success"] == 0]
    return success_df, fail_df

def get_3d_total(df):
    """
    Teilt den DataFrame in 3D-gesicherte und nicht gesicherte Transaktionen auf.

    Parameters:
        df (DataFrame): Der DataFrame mit den zu analysierenden Daten.

    Returns:
        tuple: DataFrames für 3D-gesicherte und nicht gesicherte Transaktionen.
    """
    drei_d_df = df[df["3D_secured"] == 1]
    no_3d_df = df[df["3D_secured"] == 0]
    return drei_d_df, no_3d_df

def get_success_3d(drei_d_df):
    """
    Teilt den DataFrame mit 3D-gesicherten Transaktionen weiter in erfolgreiche und fehlgeschlagene.

    Parameters:
        drei_d_df (DataFrame): DataFrame mit 3D-gesicherten Transaktionen.

    Returns:
        tuple: DataFrames für erfolgreiche und fehlgeschlagene 3D-gesicherte Transaktionen.
    """
    success_df_3d = drei_d_df[drei_d_df["success"] == 1]
    fail_df_3d = drei_d_df[drei_d_df["success"] == 0]
    return success_df_3d, fail_df_3d

# Hauptprogramm
if __name__ == "__main__":

    # Pfad zur Excel-Datei mit den Daten
    path_to_data = r"Fallstudie\use_case_1\use_case_1\PSP_Jan_Feb_2019.xlsx"
    
    # Laden der Excel-Datei in einen DataFrame
    df = pd.read_excel(path_to_data)
    
    # Liste aller Spalten des DataFrames speichern
    parameter_list = list(df)
    
    # Entfernen der unnötigen Spalte "Unnamed: 0"
    df = df.drop("Unnamed: 0", axis=1)
    
    # Aufteilung des DataFrames in erfolgreiche und fehlgeschlagene Transaktionen
    success_df, fail_df = get_success_total(df)
    
    # Aufteilung des DataFrames in 3D-gesicherte und nicht gesicherte Transaktionen
    drei_d_df, no_3d_df = get_3d_total(df)
    
    # Aufteilung der 3D-gesicherten Transaktionen in erfolgreiche und fehlgeschlagene
    success_df_3d, fail_df_3d = get_success_3d(drei_d_df)
    
    # Definition der Spalten, für die Häufigkeitsanalysen durchgeführt werden sollen
    columns_to_plot = ['country', 'PSP', 'card']
    
    # Erstellung von Barplots für die Werteverteilung in den definierten Spalten
    plot_value_counts_with_percentages(df, columns_to_plot)
    
    # Erstellung eines Boxplots zur Analyse der Betragsverteilung
    plot_boxplots_with_mean(df)
    
    # Überblick über den DataFrame und Prüfung auf fehlende Werte
    data_overview(df)
    missing_values_analysis(df)
    
    # Analyse der Erfolgsquoten basierend auf verschiedenen Faktoren
    analyze_distributions(df)
    
    # Analyse der Korrelationen zwischen numerischen Variablen
    analyze_relationships(df)
    
    # Umwandlung der 'tmsp'-Spalte in ein Datumsformat
    df['tmsp'] = pd.to_datetime(df['tmsp'])
    
    # Visualisierung der Erfolgsrate über die Zeit
    plt.figure()
    plt.plot(df['tmsp'], df['success'], ".")
