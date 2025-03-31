# credit-risk-analysis
Analiza zdolności kredytowej z użyciem regresji logistycznej, Random Forest i XGBoost
Opis projektu: Analiza zdolności kredytowej (German Credit Risk)
Ten projekt analizuje zbiór danych "German Credit Risk" (960 rekordów po czyszczeniu) w celu przewidywania zdolności kredytowej (Creditability: 0 = "bad", 1 = "good") za pomocą trzech modeli: regresji logistycznej, Random Forest i XGBoost.

Przetwarzanie danych:
Usunięto outliery: rekordy z Credit Amount > 10,000 (4% danych).
Dodano cechę Monthly Burden = Credit Amount / Duration of Credit (month) jako wskaźnik obciążenia miesięcznego.

Modele i wyniki:
Regresja logistyczna (AUC 0.80, Accuracy 0.73):
Najlepszy recall dla klasy "bad" (0.79).
Odpowiednia dla minimalizacji ryzyka kredytowego.

Random Forest (tuned) (AUC 0.80, Accuracy 0.79):
Najwyższe accuracy i recall dla "good" (0.95).
Optymalny do maksymalizacji liczby przyznanych kredytów.

XGBoost (AUC 0.79, Accuracy 0.77):
Najbardziej zrównoważony (recall: 0.57 dla "bad", 0.84 dla "good").
Dobra opcja ogólnego zastosowania.

Wnioski:
Czyszczenie danych i nowa cecha poprawiły wydajność modeli (AUC ~0.80).
Wybór modelu zależy od priorytetów biznesowych: wykrywanie ryzyka (regresja logistyczna) vs. maksymalizacja kredytów (Random Forest).
