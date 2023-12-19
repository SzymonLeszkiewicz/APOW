# Projekt Image Processing z wykorzystaniem SIFT i Image Stitching

## Opis projektu

Ten projekt skupia się na wykorzystaniu algorytmu SIFT (Scale-Invariant Feature Transform) do detekcji i opisu punktów kluczowych w obrazach, a także na technice image stitching do łączenia dwóch obrazów.





## Notebooks

1. **notebooks/:** Katalog zawierający notebooki Jupyter z realizacją poszczególnych etapów projektu.

   - `Lab6.1_zadanie.ipynb`: przegląd parametrów SIFT i ich wpływu na wyniki detekcji i dopasowania punktów kluczowych.
        - <img src="media/lena_kp.png" alt="lena_kp.png" style="zoom:50%;" />

   - `Lab6.2_zadanie.ipynb`: 
     - dopasowywanie punktów kluczowych:
        - <img src="media/coke_matching.png" alt="lena_kp.png" style="zoom:30%;" />
     - Detekcja i dopasowanie punktów kluczowych na obrazach z różnymi perspektywami:
        - <img src="media/shelf_matching.png" alt="lena_stitched.png" style="zoom:41.5%;" />
     - Transformacja perspektywiczna, na podstawie dopasowanych punktów kluczowych:
        - <img src="media/shelf_transform.png" alt="lena_kp.png" style="zoom:41.5%;" />




2. **photos/:** Katalog zawierający przykładowe obrazy wykorzystane w projektach.


## Jak uruchomić notebooki

1. `pip install -r requirements.txt`.

