Dieses Repository dient der Bearbeitung der Fallstudie aus Aufgabenstellung 1: Erstellen eines Prognosemodells für den Kreditkartenzahlungsverkehr bei Online-Einkäufen im Rahmen des Kurses Model Engineering.

Die Skripte dieses Repositories sind synchron zur schriftlichen Ausarbeitung aufgebaut.

1. exploratory_analysis:
Dieses Skript führt die explorative Datenanalyse aus, wie sie in der schriftlichen Arbeit beschrieben ist. Ziel ist es, einen Überblick über die Daten zu erhalten, Verteilungen zu analysieren und erste Muster zu erkennen.

2. feature_eng:
Hier werden wiederholte Transaktionen modelliert und untersucht.

3. model_development:
Dieses Skript beinhaltet das Training und die Evaluation verschiedener Machine-Learning-Modelle, die für die Vorhersage der Erfolgswahrscheinlichkeit von Transaktionen genutzt werden.

4. apply_model_with_feature_importance:
Hier wird das endgültige Modell angewendet, das den besten Zahlungsdienstleister für eine Transaktion vorhersagt.

Das Modell besteht aus zwei Komponenten:

  -Vorhersagemodell: Der beste Klassifikator wird verwendet, um die Erfolgswahrscheinlichkeit für eine neue Instanz aus unbekannten Testdaten vorherzusagen.

  -Kostenoptimierung: Wenn die Erfolgswahrscheinlichkeit eine flexible Schwelle überschreitet, wird der kostengünstigste Zahlungsdienstleister ausgewählt.

Zusätzlich berechnet und visualisiert dieses Skript die Feature Importance. Die wichtigsten Merkmale werden detailliert analysiert und ausgegeben, um die Entscheidungsfindung des Modells nachvollziehbar zu machen.

Um die schriftliche Arbeit kombiniert mit den Skripten einfach und schnell nachvollziehen zu können, sind die Skripte direkt ausführbar, wenn sie als main-Datei ausgeführt werden.
