# import required libraries
import batch.module2
import batch.functions


# batch 1 - lo usamos para hacer el crawling y el scraping
def batchOne():
    print("\n############ Ejecutando Batch 1: Descarga de ficheros #############\n")
    # TODO: CONECTAR CON GOOGLE DRIVE Y OBTENER LOS DATOS CON GDOWN, LUEGO PARSEAMOS A JSON

    archivos = [
        ('https://drive.google.com/file/d/1e_G-9a66AryHxBOwGWhriePYCCa4_29e/view', 'subtaskA_dev_monolingual.jsonl'),
        ('https://drive.google.com/file/d/1HeCgnLuDoUHhP-2OsTSSC3FXRLVoI6OG/view', 'subtaskA_train_monolingual.jsonl'),
        ('https://drive.google.com/file/d/1oh9c-d0fo3NtETNySmCNLUc6H1j4dSWE/view', 'subtaskB_dev.jsonl'),
        ('https://drive.google.com/file/d/1k5LMwmYF7PF-BzYQNE2ULBae79nbM268/view', 'subtaskB_train.jsonl'),
        ('https://drive.google.com/file/d/1ZEmXha1_apQlu3il2fSbE6CyQzNFm3di/view', 'DataSetFinal.tsv')
    ]
    # descargamos los archivos
    batch.functions.descarga_archivos(archivos)

    batch.module2.batchTwo()
