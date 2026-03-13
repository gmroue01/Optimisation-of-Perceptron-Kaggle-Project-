import numpy as np
from scipy.sparse import rand
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
import time


def computeConditionNumber(A):
    # =====================================================================
    # 1. Création d'une matrice (très) grande et creuse
    # =====================================================================
    print("1. Création d'une grande matrice creuse (10 000 x 10 000)...")
    n = 10000




    # =====================================================================
    # 2. Méthode 3 : Randomized SVD pour la plus GRANDE valeur singulière
    # =====================================================================
    print("\n2. Calcul de la plus GRANDE valeur singulière (Randomized SVD)...")
    start_time = time.time()


    U, Sigma_max_array, VT = randomized_svd(A, n_components=1, random_state=42)

    sigma_max = Sigma_max_array[0]
    print(f"   -> sigma_max = {sigma_max:.4f}")
    print(f"   -> Temps d'exécution : {time.time() - start_time:.4f} secondes")


    # =====================================================================
    # 3. Trouver la plus PETITE valeur singulière (Shift-and-Invert)
    # =====================================================================
    print("\n3. Calcul de la plus PETITE valeur singulière (Krylov/SciPy)...")
    start_time = time.time()

    u, Sigma_min_array, vt = svds(A, k=1, which='SM')

    sigma_min = Sigma_min_array[0]
    print(f"   -> sigma_min = {sigma_min:.4e}")
    print(f"   -> Temps d'exécution : {time.time() - start_time:.4f} secondes")


    # =====================================================================
    # 4. Calcul du conditionnement
    # =====================================================================
    print("\n" + "="*50)
    if sigma_min == 0:
        print("La matrice est singulière (sigma_min = 0). Conditionnement infini.")
    else:
        kappa = sigma_max / sigma_min
        print(f"CONDITIONNEMENT ESTIMÉ (Norme 2) : {kappa:.4e}")
    print("="*50)

    return kappa
