# TP 2 : K-means Clustering 

---

## 📁 Structure du Projet

```
votre_projet/
│
├── data/                                      # 📂 Données sources
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
│
├── resultats/                                 # 📂 Tous les résultats
│   ├── visualisations/                        # 🖼️ Graphiques PNG
│   │   ├── distribution_variable_cible.png
│   │   ├── distributions_numeriques.png
│   │   ├── matrice_correlation.png
│   │   ├── methode_coude.png
│   │   ├── score_silhouette.png
│   │   ├── davies_bouldin.png
│   │   ├── distribution_clusters.png
│   │   ├── matrice_confusion.png
│   │   ├── profils_clusters_heatmap.png
│   │   ├── pca_2d.png
│   │   └── tsne_2d.png
│   │
│   ├── obesity_with_clusters.csv             # 📊 Dataset avec clusters
│   ├── profils_clusters.csv                  # 📊 Profils moyens
│   └── cluster_profiles_report.txt           # 📄 Rapport détaillé
│
└── TP2_KMeans_Obesite_OPTIMISE.ipynb         # 📓 Ce notebook
```

---

## 🚀 Installation Rapide

### 1️⃣ Créer la structure

```bash
# Créer les dossiers (ou le notebook le fera automatiquement)
mkdir -p data resultats/visualisations
```

### 2️⃣ Placer le dataset

Téléchargez le fichier CSV et placez-le dans le dossier `data/` :

```
data/
└── ObesityDataSet_raw_and_data_sinthetic.csv
```

### 3️⃣ Installer les dépendances

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 4️⃣ Lancer le notebook

```bash
jupyter notebook TP2_KMeans_Obesite_OPTIMISE.ipynb
```

### 5️⃣ Exécuter

- Exécutez toutes les cellules : `Cell → Run All`
- Ou cellule par cellule : `Shift + Enter`

---

## 🎯 Améliorations par Rapport à la Version Précédente

### 🗂️ Organisation

| Avant | Après |
|-------|-------|
| Fichiers dans le dossier courant | Structure en dossiers claire |
| Graphiques non sauvegardés | Tous les graphiques exportés en PNG |
| Pas de rapport automatique | Rapport TXT détaillé généré |

### 💻 Code

| Avant | Après |
|-------|-------|
| Chemins en dur | Utilisation de `pathlib.Path` |
| `random_state` répété | Constante `RANDOM_STATE = 42` |
| Pas de vérification fichier | Vérification avec message d'erreur clair |
| Commentaires techniques | Commentaires naturels et pédagogiques |

### 📊 Visualisations

| Avant | Après |
|-------|-------|
| Affichage uniquement | Sauvegarde automatique (DPI 150) |
| Pas de récapitulatif | Liste des fichiers générés |
| Graphiques basiques | Graphiques améliorés (annotations, grille) |

### 📝 Rapports

| Avant | Après |
|-------|-------|
| Rapport simple | Rapport complet avec timestamp |
| Pas de résumé fichiers | Résumé automatique avec tailles |
| CSV basique | CSV + profils moyens séparés |

---

## 📚 Sections du Notebook

### 1. Configuration (Nouveau ✨)
- Import des bibliothèques
- Création automatique des dossiers
- Configuration des constantes globales

### 2. Chargement des Données
- Vérification de l'existence du fichier
- Message d'erreur clair si fichier manquant
- Exploration initiale complète

### 3. Prétraitement
- Encodage des variables catégorielles
- Normalisation StandardScaler
- Validation des transformations

### 4. Détermination du K Optimal
- Méthode du coude
- Score de silhouette
- Indice de Davies-Bouldin
- **Nouveau** : Sauvegarde automatique de chaque graphique

### 5. Entraînement K-means
- Configuration optimale
- Distribution des clusters
- Visualisation sauvegardée

### 6. Évaluation
- Métriques multiples
- Interprétation automatique (Nouveau ✨)
- Matrice de confusion sauvegardée

### 7. Analyse des Clusters
- Profils moyens exportés en CSV
- Heatmap des profils normalisés
- Caractérisation textuelle détaillée

### 8. Visualisations Avancées
- PCA 2D avec centroïdes
- t-SNE 2D
- **Nouveau** : Toutes sauvegardées automatiquement

### 9. Sauvegarde (Amélioré ✨)
- Dataset enrichi
- Rapport complet avec timestamp
- Résumé des fichiers générés

### 10. Synthèse Finale (Nouveau ✨)
- Configuration récapitulative
- Métriques de performance
- Liste des livrables

---

## 🎨 Qualité du Code

### Conventions Respectées

✅ **PEP 8** - Style Python standard  
✅ **Noms explicites** - Variables claires (`K_OPTIMAL`, `RESULTS_DIR`)  
✅ **Commentaires** - Explications naturelles  
✅ **Organisation** - Sections logiques et progressives  
✅ **Reproductibilité** - `RANDOM_STATE` utilisé partout  
✅ **Robustesse** - Vérification des erreurs  

### Formatage

- **Constantes** : `MAJUSCULES_AVEC_UNDERSCORES`
- **Variables** : `snake_case_descriptif`
- **Spacing** : Lignes vides pour la lisibilité
- **Docstrings** : Markdown pour la documentation

---

## 📊 Résultats Attendus

À la fin de l'exécution, vous obtiendrez :

### Fichiers CSV (3)
- `obesity_with_clusters.csv` - Dataset original + colonne Cluster
- `profils_clusters.csv` - Moyennes par cluster
- (Le dataset original reste intact dans `data/`)

### Visualisations PNG (11+)
Tous les graphiques en haute résolution (150 DPI) :
- Distribution de la variable cible
- Distributions des variables numériques
- Matrice de corrélation
- Méthode du coude
- Score de silhouette
- Davies-Bouldin
- Distribution des clusters
- Matrice de confusion
- Heatmap des profils
- PCA 2D
- t-SNE 2D

### Rapports (1)
- `cluster_profiles_report.txt` - Rapport complet avec :
  - Configuration du modèle
  - Métriques de performance
  - Profils détaillés de chaque cluster
  - Timestamp

---

## ⏱️ Temps d'Exécution

| Section | Temps Estimé |
|---------|--------------|
| Configuration | < 1 sec |
| Chargement données | < 5 sec |
| Prétraitement | < 5 sec |
| Détermination K | ~3 min |
| Entraînement | < 30 sec |
| Évaluation | < 10 sec |
| Analyse | < 30 sec |
| PCA | < 10 sec |
| t-SNE | 1-2 min |
| Sauvegarde | < 10 sec |
| **TOTAL** | **~6 minutes** |

---

## 🔧 Configuration Avancée

### Modifier le Nombre de Clusters

```python
# Après la section 4.4, vous pouvez forcer un K spécifique
K_OPTIMAL = 7  # Par exemple pour comparer aux 7 catégories originales
```

### Changer les Dossiers

```python
# Section 1, cellule 2
DATA_DIR = Path('mes_donnees')
RESULTS_DIR = Path('mes_resultats')
```

### Ajuster les Paramètres K-means

```python
# Section 5
kmeans_final = KMeans(
    n_clusters=K_OPTIMAL,
    init='k-means++',    # Ou 'random'
    n_init=20,           # Augmenter pour plus de stabilité
    max_iter=500,        # Augmenter si non convergence
    random_state=RANDOM_STATE
)
```

### Modifier la Résolution des Images

```python
# Dans chaque plt.savefig()
plt.savefig(fichier, dpi=300, bbox_inches='tight')  # Haute résolution
```

---

## ❓ FAQ

### Q : Puis-je utiliser mes propres données ?
**R :** Oui ! Placez votre CSV dans `data/` et modifiez `DATA_FILE` en conséquence.

### Q : Les graphiques sont-ils modifiables ?
**R :** Oui ! Ils sont en format PNG. Vous pouvez aussi les regénérer avec d'autres paramètres.

### Q : Comment changer le K optimal ?
**R :** Modifiez `K_OPTIMAL` après la section 4.4 et réexécutez à partir de la section 5.

### Q : Le notebook crée-t-il les dossiers automatiquement ?
**R :** Oui ! Si `data/` et `resultats/` n'existent pas, ils seront créés automatiquement.

### Q : Puis-je désactiver la sauvegarde des graphiques ?
**R :** Oui, commentez les lignes `plt.savefig()` dans chaque cellule de visualisation.

---

## 🎓 Pour Aller Plus Loin

### Algorithmes Alternatifs
```python
from sklearn.cluster import DBSCAN, AgglomerativeClustering

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

# Hierarchical
hierarchical = AgglomerativeClustering(n_clusters=K_OPTIMAL)
labels_hier = hierarchical.fit_predict(X_scaled)
```

### Feature Engineering
```python
# Créer l'IMC (Indice de Masse Corporelle)
df['IMC'] = df['Weight'] / (df['Height'] ** 2)

# Ratio activité / calories
df['Ratio_Sante'] = df['FAF'] / (df['FAVC'].map({'yes': 1, 'no': 0}) + 1)
```

### Validation Croisée
```python
from sklearn.model_selection import cross_val_score

# Tester la stabilité avec différentes graines
scores = []
for seed in range(10):
    kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=seed)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)
    
print(f"Score moyen : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

---

## 📞 Support

Pour toute question :
1. Consultez le TROUBLESHOOTING.md
2. Vérifiez que la structure des dossiers est correcte
3. Vérifiez que le dataset est au bon endroit

---


**🎉 Prête pour la Production !**

*Dernière mise à jour : Octobre 2025*
*Version : 2.0*
