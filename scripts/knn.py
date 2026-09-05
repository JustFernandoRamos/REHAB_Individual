from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

#---------------------------Carga de datos Raw--------------------

X_PATH = os.getenv('X_PATH')
Y_PATH = os.getenv('Y_PATH')
X = np.load(X_PATH)
y = np.load(Y_PATH)

# Tomamos los primeros 6 ejercicios para reducir el tiempo de evaluación del modelo

ejercicios_seleccionados = [0, 1, 2, 3, 4, 5, 6]  # o los nombres/códigos que correspondan

mask = np.isin(y, ejercicios_seleccionados)

X = X[mask]
y = y[mask]

print("Nuevo shape de X:", X.shape)
print("Distribución de clases:", np.unique(y, return_counts=True))

def train_test_split_manual(X, y, test_ratio=0.2, seed=42):
    """
    Funcion divisora de train y test sets
    """
    np.random.seed(seed)
    n = X.shape[0]
    indices = np.arange(n)
    # Mezclamos los indices de las entradas
    np.random.shuffle(indices)

    #definimos el tamaño del test con el test ratio
    n_test = int(n * test_ratio)
    #Tomamos desde la primera entrada hasta el tamaño del tes
    test_idx = indices[:n_test]
    #el resto de entradas van al train
    train_idx = indices[n_test:]

    #regresa los sets utilizando los ids tanto en x como en y
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split_manual(X, y)


def fit_scaler(X_train):
    mean = X_train.mean(axis=(0, 1))  
    std = X_train.std(axis=(0, 1)) + 1e-8
    return mean, std

def apply_scaler(X, mean, std):
    return (X - mean) / std

mean, std = fit_scaler(X_train)
X_train_scaled = apply_scaler(X_train, mean, std)
X_test_scaled = apply_scaler(X_test, mean, std)


def extract_features(X):
    n_samples, n_timesteps, n_channels = X.shape
    features = []

    for i in range(n_samples):
        sample_feats = []
        for c in range(n_channels):
            señal = X[i, :, c]
            media = np.mean(señal)
            desv = np.std(señal)
            minimo = np.min(señal)
            maximo = np.max(señal)
            cruces = np.sum(np.diff(np.sign(señal)) != 0)

            sample_feats.extend([media, desv, minimo, maximo, cruces])
        features.append(sample_feats)

    return np.array(features)


X_train_feats = extract_features(X_train_scaled)
X_test_feats = extract_features(X_test_scaled)

print(f"las dimensiones del nuevo training set son: {X_train_feats.shape}")
print(f"las dimensiones del nuevo test set son: {X_train_feats.shape}")

def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_query, k=5):
    distancias = [distancia_euclidiana(x_query, x_train_i) for x_train_i in X_train]
    k_indices = np.argsort(distancias)[:k]
    k_labels = y_train[k_indices]
    valores, conteos = np.unique(k_labels, return_counts=True)
    return valores[np.argmax(conteos)]

def matriz_confusion(y_true, y_pred, clases):
    n_clases = len(clases)
    clase_a_idx = {c: i for i, c in enumerate(clases)}
    matriz = np.zeros((n_clases, n_clases), dtype=int)

    for real, pred in zip(y_true, y_pred):
        i = clase_a_idx[real]
        j = clase_a_idx[pred]
        matriz[i, j] += 1

    return matriz  # filas = clase real, columnas = clase predicha

def metricas_por_clase(matriz, clases):
    n_clases = len(clases)
    precision = np.zeros(n_clases)
    recall = np.zeros(n_clases)
    f1 = np.zeros(n_clases)

    for i in range(n_clases):
        vp = matriz[i, i]
        fp = np.sum(matriz[:, i]) - vp
        fn = np.sum(matriz[i, :]) - vp

        precision[i] = vp / (vp + fp) if (vp + fp) > 0 else 0.0
        recall[i] = vp / (vp + fn) if (vp + fn) > 0 else 0.0
        f1[i] = (2 * precision[i] * recall[i] / (precision[i] + recall[i])
                 if (precision[i] + recall[i]) > 0 else 0.0)

    return precision, recall, f1

def knn_evaluate(X_train, y_train, X_test, y_test, k=5):
    predicciones = [knn_predict(X_train, y_train, x, k) for x in X_test]
    predicciones = np.array(predicciones, dtype=y_train.dtype)

    # --- Métricas ---
    accuracy = np.mean(predicciones == y_test)
    clases = np.unique(np.concatenate([y_train, y_test]))
    matriz = matriz_confusion(y_test, predicciones, clases)
    precision, recall, f1 = metricas_por_clase(matriz, clases)

    resultados = {
        "accuracy": accuracy,
        "matriz_confusion": matriz,
        "clases": clases,
        "precision_por_clase": precision,
        "recall_por_clase": recall,
        "f1_por_clase": f1,
        "precision_macro": np.mean(precision),
        "recall_macro": np.mean(recall),
        "f1_macro": np.mean(f1),
        "predicciones": predicciones,
    }

    return resultados


# --- Uso ---
resultados = knn_evaluate(X_train_feats, y_train, X_test_feats, y_test, k=3)

print("Accuracy:", resultados["accuracy"])
print("F1 macro:", resultados["f1_macro"])
print("\nMatriz de confusión (filas=real, columnas=predicho):")
print(resultados["matriz_confusion"])

print("\nPor clase:")
for i, c in enumerate(resultados["clases"]):
    print(f"Clase {c}: precision={resultados['precision_por_clase'][i]:.3f}, "
          f"recall={resultados['recall_por_clase'][i]:.3f}, "
          f"f1={resultados['f1_por_clase'][i]:.3f}")
